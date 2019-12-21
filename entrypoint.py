import argparse
import os

import numpy as np
from loguru import logger

from data_loading.loo_data_loader import DesignatedDataLoader
from metrics.metrics import ndcg_at_k
from models.bpr_recommender import BPRRecommender
from models.randumb import RandomRecommender
from models.svd_recommender import SVDRecommender
from models.top_pop_recommender import TopPopRecommender

models = {
    'transe': None,
    'transe-kg': None,
    'user-knn': None,
    'item-knn': None,
    'svd': {
        'constructor': SVDRecommender
    },
    'bpr': {
        'constructor': BPRRecommender
    },
    'pr-collab': None,
    'pr-kg': None,
    'pr-joint': None,
    'mf': None,
    'joint-mf': None,
    'top-pop': {
        'constructor': TopPopRecommender
    },
    'random': {
        'constructor': RandomRecommender
    }
}

parser = argparse.ArgumentParser()
parser.add_argument('--include', nargs='*', type=str, choices=models.keys(), help='models to include')
parser.add_argument('--exclude', nargs='*', type=str, choices=models.keys(), help='models to exclude')

if __name__ == '__main__':
    logger.info(f'Working directory: {os.getcwd()}')

    # Filter models
    args = parser.parse_args()
    model_selection = set(models.keys()) if not args.include else set(args.include)
    if args.exclude:
        model_selection = model_selection.difference(set(args.exclude))

    # Load data
    data_loader = DesignatedDataLoader.load_from(
        path='data_loading/mindreader',
        movies_only=False,
        min_num_entity_ratings=1
    )

    train, validation, test = data_loader.make(
        movie_to_entity_ratio=0.25,
        replace_movies_with_descriptive_entities=True
    )

    # Run models
    for model in model_selection:
        recommender = models[model]['constructor']()

        recommender.fit(train, validation)

        hits, count = 0, 0
        ndcg_sum = list()

        k = 10
        for u, (pos_sample, neg_samples) in test:
            predictions = recommender.predict(u, neg_samples + [pos_sample]).items()
            predictions = sorted(predictions, key=lambda item: item[1], reverse=True)

            relevance = [1 if item == pos_sample else 0 for item, score in predictions]
            top_k = [item[0] for item in predictions][:k]

            if pos_sample in top_k:
                hits += 1

            ndcg_sum.append(ndcg_at_k(relevance, 10))

            count += 1

        logger.info(f'{model} HR: {hits / count * 100:.2f}')
        logger.info(f'{model} NDCG: {np.mean(ndcg_sum) * 100:.2f}')
