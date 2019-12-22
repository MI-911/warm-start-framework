import argparse
import os
from collections import defaultdict

import numpy as np
from loguru import logger

from data_loading.loo_data_loader import DesignatedDataLoader
from experiments.experiment import Experiment, Dataset
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

upper_cutoff = 50

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


def test_model(name, model, test, reverse):
    hits = defaultdict(list)
    ndcgs = defaultdict(list)

    for u, (pos_sample, neg_samples) in test:
        predictions = model.predict(u, neg_samples + [pos_sample]).items()

        # Check that the model produced as many predictions as we requested
        if len(predictions) != len(neg_samples) + 1:
            logger.error(f'{name} only produced {len(predictions)} prediction scores')

            continue

        predictions = sorted(predictions, key=lambda item: item[1], reverse=reverse)
        relevance = [1 if item == pos_sample else 0 for item, score in predictions]

        # Append hits and NDCG for various cutoffs
        for k in range(1, upper_cutoff + 1):
            cutoff = relevance[:k]

            hits[k].append(1 in cutoff)
            ndcgs[k].append(ndcg_at_k(cutoff, k))

    # Transform lists to means
    hr = dict()
    ndcg = dict()

    for k in range(1, upper_cutoff + 1):
        hr[k] = np.mean(hits[k])
        ndcg[k] = np.mean(ndcgs[k])

    return hr, ndcg


def run():
    # Filter models
    args = parser.parse_args()
    model_selection = set(models.keys()) if not args.include else set(args.include)
    if args.exclude:
        model_selection = model_selection.difference(set(args.exclude))

    dataset = Dataset('data')
    for experiment in dataset.experiments():
        for split in experiment.splits():
            # Run models
            for model in model_selection:
                model_parameters = models[model]
                recommender = instantiate(model_parameters, None)
                if not recommender:
                    logger.error(f'No parameters specified for {model}')

                    continue

                logger.info(f'Fitting {model}')
                recommender.fit(split.training, split.validation)

                hr, ndcg = test_model(model, recommender, split.testing, model_parameters.get('descending', True))

                for k in [1, 5, 10]:
                    logger.info(f'{model} HR@{k}: {hr[k] * 100:.2f}')
                    logger.info(f'{model} NDCG@{k}: {ndcg[k] * 100:.2f}')


if __name__ == '__main__':
    logger.info(f'Working directory: {os.getcwd()}')

    run()
