import argparse
import os

import numpy as np
from loguru import logger

from data_loading.loo_data_loader import DesignatedDataLoader
from metrics.metrics import ndcg_at_k
from models.bpr_recommender import BPRRecommender
from models.item_knn_recommender import ItemKNNRecommender
from models.pagerank.collaborative_pagerank_recommender import CollaborativePageRankRecommender
from models.pagerank.kg_pagerank_recommender import KnowledgeGraphPageRankRecommender
from models.randumb import RandomRecommender
from models.svd_recommender import SVDRecommender
from models.top_pop_recommender import TopPopRecommender
from models.user_knn_recommender import UserKNNRecommender

models = {
    'transe': {
        'descending': False
    },
    'transe-kg': {
        'descending': False
    },
    'user-knn': {
        'class': UserKNNRecommender,
        'data_loader': True
    },
    'item-knn': {
        'class': ItemKNNRecommender,
        'data_loader': True
    },
    'svd': {
        'class': SVDRecommender
    },
    'bpr': {
        'class': BPRRecommender
    },
    'pr-collab': {
        'class': CollaborativePageRankRecommender
    },
    'pr-kg': {
        'class': KnowledgeGraphPageRankRecommender,
        'data_loader': True
    },
    'pr-joint': None,
    'mf': None,
    'joint-mf': None,
    'top-pop': {
        'class': TopPopRecommender
    },
    'random': {
        'class': RandomRecommender
    }
}

parser = argparse.ArgumentParser()
parser.add_argument('--include', nargs='*', type=str, choices=models.keys(), help='models to include')
parser.add_argument('--exclude', nargs='*', type=str, choices=models.keys(), help='models to exclude')


def instantiate(parameters, loader):
    if not parameters or 'class' not in parameters:
        return None

    kwargs = dict()
    if parameters.get('data_loader', False):
        kwargs['data_loader'] = loader

    return parameters['class'](**kwargs)


if __name__ == '__main__':
    logger.info(f'Working directory: {os.getcwd()}')

    # Filter models
    args = parser.parse_args()
    model_selection = set(models.keys()) if not args.include else set(args.include)
    if args.exclude:
        model_selection = model_selection.difference(set(args.exclude))

    for v in [True, False]:
        logger.info(f'Movies only: {v}')

        # Load data
        data_loader = DesignatedDataLoader.load_from(
            path='./data_loading/mindreader',
            movies_only=v,
            min_num_entity_ratings=1
        )

        train, validation, test = data_loader.make(
            movie_to_entity_ratio=1,
            keep_all_ratings=True
        )

        # Run models
        for model in model_selection:
            recommender = instantiate(models[model], data_loader)
            if not recommender:
                logger.error(f'No parameters specified for {model}')

                continue

            logger.info(f'Fitting {model}')
            recommender.fit(train, validation)

            hits = list()
            ndcg_sum = list()

            k = 10
            for u, (pos_sample, neg_samples) in test:
                predictions = recommender.predict(u, neg_samples + [pos_sample]).items()
                # TODO: Consider descending parameter for sort order
                predictions = sorted(predictions, key=lambda item: item[1], reverse=True)[:k]

                relevance = [1 if item == pos_sample else 0 for item, score in predictions]
                top_k = [item[0] for item in predictions]

                hits.append(pos_sample in top_k)
                ndcg_sum.append(ndcg_at_k(relevance, 10))

            logger.info(f'{model} HR: {np.mean(hits) * 100:.2f}')
            logger.info(f'{model} NDCG: {np.mean(ndcg_sum) * 100:.2f}')
