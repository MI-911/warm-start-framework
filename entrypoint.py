import argparse
import json
import os
import sys
import traceback
from collections import defaultdict

import numpy as np
from loguru import logger

from experiments.experiment import Dataset
from experiments.metrics import ndcg_at_k
from models.bpr_recommender import BPRRecommender
from models.item_knn_recommender import ItemKNNRecommender
from models.melu_recommender import MeLURecommender
from models.pagerank.collaborative_pagerank_recommender import CollaborativePageRankRecommender
from models.pagerank.joint_pagerank_recommender import JointPageRankRecommender
from models.pagerank.kg_pagerank_recommender import KnowledgeGraphPageRankRecommender
from models.randumb import RandomRecommender
from models.svd_recommender import SVDRecommender
from models.top_pop_recommender import TopPopRecommender
from models.user_knn_recommender import UserKNNRecommender
from models.trans_e_recommender import CollabTransERecommender, KGTransERecommender
from models.mf_numpy_recommender import MatrixFactorisationRecommender
from models.mf_joint_numpy_recommender import JointMatrixFactorizaionRecommender
from time import time

from utility.table_generator import generate_table

models = {
    'transe': {
        'class': CollabTransERecommender,
        'split': True,
        'descending': False
    },
    'transe-kg': {
        'class': KGTransERecommender,
        'split': True,
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
    'ppr-collab': {
        'class': CollaborativePageRankRecommender
    },
    'ppr-kg': {
        'class': KnowledgeGraphPageRankRecommender,
        'split': True
    },
    'ppr-joint': {
        'class': JointPageRankRecommender,
        'split': True
    },
    'top-pop': {
        'class': TopPopRecommender
    },
    'random': {
        'class': RandomRecommender
    },
    'mf': {
        'class': MatrixFactorisationRecommender,
        'split': True
    },
    'joint-mf': {
        'class': JointMatrixFactorizaionRecommender,
        'split': True
    },
    'melu': {
        'class': MeLURecommender,
        'split': True
    },
}

upper_cutoff = 50

parser = argparse.ArgumentParser()
parser.add_argument('--include', nargs='*', type=str, choices=models.keys(), help='models to include')
parser.add_argument('--exclude', nargs='*', type=str, choices=models.keys(), help='models to exclude')
parser.add_argument('--debug', action='store_true', help='enable debug mode')
parser.add_argument('--summary', action='store_true', help='generate summaries for experiments')
parser.add_argument('--table', action='store_true', help='generate table for experiments')
parser.add_argument('--experiments', nargs='*', type=str, help='experiments to run')
parser.add_argument('--test', nargs='?', type=str, help='experiment to two-sample t-test on')


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


def get_summary(experiment_base):
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
            if (not file.endswith('.json')) or file == 'params.json':
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

            if k not in hrs or k not in ndcgs:
                logger.warning(f'Skipping {k} for {model} due to missing data')
                continue

            hr[k] = {
                'mean': np.mean(hrs[k]) if hrs[k] else np.nan,
                'std': np.std(hrs[k]) if hrs[k] else np.nan
            }

            ndcg[k] = {
                'mean': np.mean(ndcgs[k]) if ndcgs[k] else np.nan,
                'std': np.std(ndcgs[k]) if ndcgs[k] else np.nan
            }

        if hr and ndcg:
            results[model] = {'hr': hr, 'ndcg': ndcg}

    return results


def summarise(experiment_base):
    summary_path = os.path.join(experiment_base, 'summary.json')
    with open(summary_path, 'w') as fp:
        json.dump(get_summary(experiment_base), fp)

        logger.debug(f'Wrote summary to {summary_path}')


def run():
    # Filter models
    args = parser.parse_args()
    model_selection = set(models.keys()) if not args.include else set(args.include)
    if args.exclude:
        model_selection = model_selection.difference(set(args.exclude))

    if not args.debug:
        logger.remove()
        logger.add(sys.stderr, level='INFO')

    # Initialize dataset
    dataset = Dataset('data', args.experiments)

    # Create results folder
    results_base = 'results'
    if not os.path.exists(results_base):
        os.mkdir(results_base)

    # If summary, then create summaries
    if args.summary:
        for experiment in dataset.experiments():
            summarise(os.path.join(results_base, experiment.name))

        if not args.table:
            return

    # If table, then generate table here
    if args.table:
        if not args.experiments:
            logger.error('Must specify experiments to generate a table')

            return

        for metric in ['hr', 'ndcg']:
            table = generate_table(results_base, args.experiments, metric, test=args.test)
            if table:
                print(table)
            else:
                logger.error(f'Failed to generate {metric} table')

        return
    elif args.test:
        logger.error('Cannot specify test without generating table')

        return

    # Run experiments
    for experiment in dataset.experiments():
        logger.info(f'Starting experiment {experiment.name}')
        experiment_start = time()

        # Create experiment directory
        experiment_base = os.path.join(results_base, experiment.name)
        if not os.path.exists(experiment_base):
            os.mkdir(experiment_base)

        # Run all splits
        c = 0
        for split in experiment.splits():
            logger.info(f'Starting split {split.name}')
            split_start = time()

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
                logger.info(f'Starting {model}')
                start_time = time()
                try:
                    params = get_params(model_base)
                    if not params:
                        logger.debug(f'Tuning hyper parameters for {model}')
                    else:
                        logger.debug(f'Reusing optimal parameters for {model}: {params}')
                        recommender.optimal_params = params

                    recommender.fit(split.training, split.validation)
                    hr, ndcg = test_model(model, recommender, split.testing, model_parameters.get('descending', True))
                except Exception as e:
                    logger.error(f'{model} failed during {split} due to {e}')
                    traceback.print_exc()

                    break

                # Save results to split file
                with open(os.path.join(model_base, split.name), 'w') as fp:
                    json.dump({'hr': hr, 'ndcg': ndcg}, fp)
                with open(os.path.join(model_base, 'params.json'), 'w') as fp:
                    json.dump(recommender.optimal_params, fp)

                # Debug
                logger.info(f'{model} ({time() - start_time:.2f}s): {hr[10] * 100:.2f}% HR, {ndcg[10] * 100:.2f}% NDCG')

                c += 1

            logger.info(f'Split {split.name} took {time() - split_start:.2f}s')
        logger.info(f'Experiment {experiment.name} took {time() - experiment_start:.2f}s')

        # Summarise the experiment in a single file
        summarise(experiment_base)


def get_params(model_base):
    path = os.path.join(model_base, 'params.json')
    if os.path.exists(path):
        with open(path) as fp:
            return json.load(fp)

    return None


if __name__ == '__main__':
    logger.info(f'Working directory: {os.getcwd()}')

    run()
