from data_loading.loo_data_loader import LeaveOneOutDataLoader
from os.path import join
import json
import os
from multiprocessing.pool import Pool
from time import time
from shutil import copyfile

from loguru import logger

experiments = [
    ['all_movies', {
        'movie_to_entity_ratio': 1,
        'keep_all_ratings': True,
        'replace_movies_with_descriptive_entities': False,
        'n_negative_samples': 100,
        'movies_only': True
    }],
    ['all_entities', {
        'movie_to_entity_ratio': 1,
        'keep_all_ratings': True,
        'replace_movies_with_descriptive_entities': False,
        'n_negative_samples': 100,
        'movies_only': False
    }],
    ['substituting-3-4', {
        'movie_to_entity_ratio': 3/4,
        'keep_all_ratings': False,
        'replace_movies_with_descriptive_entities': True,
        'n_negative_samples': 100,
        'movies_only': False
    }],
    ['substituting-2-4', {
        'movie_to_entity_ratio': 2/4,
        'keep_all_ratings': False,
        'replace_movies_with_descriptive_entities': True,
        'n_negative_samples': 100,
        'movies_only': False
    }],
    ['substituting-1-4', {
        'movie_to_entity_ratio': 1/4,
        'keep_all_ratings': False,
        'replace_movies_with_descriptive_entities': True,
        'n_negative_samples': 100,
        'movies_only': False
    }]
]


def generate_with_top_pop(filter_unkowns=False):

    n_experiments = 10
    n_attempts = 0

    for i in range(n_experiments):
        for experiment, args in experiments:
            experiment_dir = join(f'datasets_with_top_pop{"" if filter_unkowns else "_with_unknowns"}', experiment)
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)

            successfully_created_experiment = False
            while not successfully_created_experiment:
                try:
                    filename = join(experiment_dir, f'{i}.json')
                    random_seed = n_attempts + i
                    logger.info(f'Attempting to create {filename} with random seed {random_seed}...')

                    args['random_seed'] = random_seed
                    loader = LeaveOneOutDataLoader.load_from(
                        join('../data_loading', 'mindreader'),
                        min_num_entity_ratings=1,
                        filter_unknowns=filter_unkowns,
                        unify_user_indices=False,
                        random_seed=random_seed
                    )
                    train, val, test = loader.make(**args)
                    dict_train = []
                    for u, ratings in train:
                        dict_train.append((u, [r.__dict__ for r in ratings]))

                    with open(filename, 'w') as fp:
                        json.dump({
                            'training': dict_train,
                            'validation': val,
                            'testing': test
                        }, fp)
                    successfully_created_experiment = True

                    logger.info(f'Successfully wrote {filename} to disk.')
                except AssertionError as e:
                    print(e)
                    n_attempts += 1


def generate_without_top_pop(filter_unkowns=False):

    n_experiments = 10
    n_attempts = 0

    for i in range(n_experiments):
        for experiment, args in experiments:
            experiment_dir = join(f'datasets_no_top_pop{"" if filter_unkowns else "_with_unknowns"}', experiment)
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)

            successfully_created_experiment = False
            while not successfully_created_experiment:
                try:
                    filename = join(experiment_dir, f'{i}.json')
                    random_seed = n_attempts + i
                    logger.info(f'Attempting to create {filename} with random seed {random_seed}...')
                    args['random_seed'] = random_seed
                    args['without_top_pop'] = True
                    loader = LeaveOneOutDataLoader.load_from(
                        join('./data_loading', 'mindreader'),
                        min_num_entity_ratings=1,
                        filter_unknowns=filter_unkowns,
                        unify_user_indices=False,
                        random_seed=random_seed
                    )
                    train, val, test = loader.make(**args)
                    dict_train = []
                    for u, ratings in train:
                        dict_train.append((u, [r.__dict__ for r in ratings]))

                    with open(filename, 'w') as fp:
                        json.dump({
                            'training': dict_train,
                            'validation': val,
                            'testing': test
                        }, fp)
                    successfully_created_experiment = True

                    logger.info(f'Successfully wrote {filename} to disk.')
                except AssertionError as e:
                    print(e)
                    n_attempts += 1


def _generate_dataset(args): 
    (experiment, args), (filter_unknowns, without_top_pop, i, base_dir) = args
    experiment_dir = join(base_dir, experiment)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    successfully_created_experiment = False
    while not successfully_created_experiment:
        try:
            filename = join(experiment_dir, f'{"ntp" if without_top_pop else "wtp"}-{i}.json')
            random_seed = time()
            logger.info(f'Attempting to create {filename} with random seed {random_seed}...')
            args['random_seed'] = random_seed
            args['without_top_pop'] = without_top_pop
            loader = LeaveOneOutDataLoader.load_from(
                join('./data_loading', 'mindreader'),
                min_num_entity_ratings=1,
                filter_unknowns=filter_unknowns,
                unify_user_indices=False,
                random_seed=random_seed
            )
            train, val, test = loader.make(**args)
            dict_train = []
            for u, ratings in train:
                dict_train.append((u, [r.__dict__ for r in ratings]))

            with open(filename, 'w') as fp:
                json.dump({
                    'training': dict_train,
                    'validation': val,
                    'testing': test
                }, fp)
            successfully_created_experiment = True

            logger.info(f'Successfully wrote {filename} to disk.')
        except AssertionError as e:
            print(e)


def prepare(datasets_dir='./datasets', mindreader_dir='./data_loading/mindreader/'):
    """ Creates a datasets_dir directory and adds KG triples and meta.json"""
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)

    # Copy triples
    copyfile(join(mindreader_dir, 'triples.csv'), join(datasets_dir, 'triples.csv'))
    logger.info(f'Copied triples (from {mindreader_dir})')

    # Write meta.json
    loader = LeaveOneOutDataLoader.load_from(
        join('./data_loading', 'mindreader'),
        min_num_entity_ratings=1,
        filter_unknowns=True,
        unify_user_indices=False,
        random_seed=42
    )

    with open(join(datasets_dir, 'meta.json'), 'w') as fp:
        json.dump({
            'e_idx_map': loader.e_idx_map
        }, fp)

    logger.info(f'Wrote meta.json')


def generate(filter_unknowns=False, without_top_pop=False, base_dir='./results', n_experiments=10):
    for i in range(n_experiments):
        with Pool(8) as p: 
            p.map(_generate_dataset, [(args, (filter_unknowns, without_top_pop, i, base_dir)) for args in experiments])

