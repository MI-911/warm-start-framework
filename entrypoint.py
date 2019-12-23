import argparse
import json
import os
import traceback
from collections import defaultdict

import numpy as np
from loguru import logger

from experiments.experiment import Dataset
from metrics.metrics import ndcg_at_k
from models.bpr_recommender import BPRRecommender
from models.item_knn_recommender import ItemKNNRecommender
from models.pagerank.collaborative_pagerank_recommender import CollaborativePageRankRecommender
from models.pagerank.joint_pagerank_recommender import JointPageRankRecommender
from models.pagerank.kg_pagerank_recommender import KnowledgeGraphPageRankRecommender
from models.randumb import RandomRecommender
from models.svd_recommender import SVDRecommender
from models.top_pop_recommender import TopPopRecommender
from models.user_knn_recommender import UserKNNRecommender
from models.trans_e_recommender import CollabTransERecommender, KGTransERecommender

models = {
    'transe': {
        'class': CollabTransERecommender,
        'descending': False
    },
    'transe-kg': {
        'class': KGTransERecommender,
        'descending': False
    },
    'user-knn': {
        'class': UserKNNRecommender,
        'split': True
    },
    'item-knn': {
        'class': ItemKNNRecommender,
        'split': True
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
        'split': True
    },
    'pr-joint': {
        'class': JointPageRankRecommender,
        'split': True
    },
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
parser.add_argument('--experiments', nargs='*', type=str, help='experiments to run')


def instantiate(parameters, split):
    if not parameters or 'class' not in parameters:
        return None

    kwargs = dict()
    if parameters.get('split', False):
        kwargs['split'] = split

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


def summarise(experiment_base):
    model_directories = []
    for directory in os.listdir(experiment_base):
        if not os.path.isdir(os.path.join(experiment_base, directory)):
            continue

        model_directories.append(directory)

    if not model_directories:
        return {}

    results = dict()

    for model in model_directories:
        # Get split results
        model_base = os.path.join(experiment_base, model)
        model_splits = {}

        # Load all splits for this model
        for file in os.listdir(model_base):
            if not file.endswith('.json'):
                continue

            with open(os.path.join(model_base, file), 'r') as fp:
                model_splits[file] = json.load(fp)

        # Get all HRs and NDCGs for the different cutoffs
        hrs = defaultdict(list)
        ndcgs = defaultdict(list)

        for contents in model_splits.values():
            for k in range(1, upper_cutoff + 1):
                k = str(k)

                hrs[k].append(contents['hr'][k])
                ndcgs[k].append(contents['ndcg'][k])

        # Get mean and std for HR and NDCG at each cutoff
        hr = dict()
        ndcg = dict()

        for k in range(1, upper_cutoff + 1):
            k = str(k)

            hr[k] = {
                'mean': np.mean(hrs[k]),
                'std': np.std(hrs[k])
            }

            ndcg[k] = {
                'mean': np.mean(ndcgs[k]),
                'std': np.std(ndcgs[k])
            }

        results[model] = {'hr': hr, 'ndcg': ndcg}

    return results


def run():
    # Filter models
    args = parser.parse_args()
    model_selection = set(models.keys()) if not args.include else set(args.include)
    if args.exclude:
        model_selection = model_selection.difference(set(args.exclude))

    # Initialize dataset
    dataset = Dataset('data', args.experiments)

    # Create results folder
    results_base = 'results'
    if not os.path.exists(results_base):
        os.mkdir(results_base)

    # Run experiments
    for experiment in dataset.experiments():
        logger.info(f'Experiment: {experiment.name}')

        # Create experiment directory
        experiment_base = os.path.join(results_base, experiment.name)
        if not os.path.exists(experiment_base):
            os.mkdir(experiment_base)

        # Run all splits
        for split in experiment.splits():
            # Run models
            for model in model_selection:
                # Instantiate model
                model_parameters = models[model]
                recommender = instantiate(model_parameters, split)
                if not recommender:
                    logger.error(f'No parameters specified for {model}')

                    continue

                # Create directory for model
                model_base = os.path.join(experiment_base, model)
                if not os.path.exists(model_base):
                    os.mkdir(model_base)

                # Fit and test
                logger.info(f'Fitting {model}')
                try:
                    recommender.fit(split.training, split.validation)
                    hr, ndcg = test_model(model, recommender, split.testing, model_parameters.get('descending', True))
                except Exception as e:
                    logger.error(f'{model} failed during {split} due to {e}')
                    traceback.print_exc()

                    break

                # Save results to split file
                with open(os.path.join(model_base, split.name), 'w') as fp:
                    json.dump({'hr': hr, 'ndcg': ndcg}, fp)

                # Debug
                for k in [10]:
                    logger.info(f'{model} HR@{k}: {hr[k] * 100:.2f}')
                    logger.info(f'{model} NDCG@{k}: {ndcg[k] * 100:.2f}')

        # Summarise the experiment in a single file
        with open(os.path.join(experiment_base, 'summary.json'), 'w') as fp:
            json.dump(summarise(experiment_base), fp)


if __name__ == '__main__':
    logger.info(f'Working directory: {os.getcwd()}')

    run()
