from data_loading.loo_data_loader import DesignatedDataLoader
from os.path import join
import json
import os

from loguru import logger

experiments = [
    ['substituting-4-4', {
        'movie_to_entity_ratio': 4 / 4,
        'keep_all_ratings': False,
        'replace_movies_with_descriptive_entities': True,
        'n_negative_samples': 100
    }],
    ['substituting-3-4', {
        'movie_to_entity_ratio': 3 / 4,
        'keep_all_ratings': False,
        'replace_movies_with_descriptive_entities': True,
        'n_negative_samples': 100
    }],
    ['substituting-2-4', {
        'movie_to_entity_ratio': 2 / 4,
        'keep_all_ratings': False,
        'replace_movies_with_descriptive_entities': True,
        'n_negative_samples': 100
    }],
    ['substituting-1-4', {
        'movie_to_entity_ratio': 1 / 4,
        'keep_all_ratings': False,
        'replace_movies_with_descriptive_entities': True,
        'n_negative_samples': 100
    }],

    ['substituting-4-0', {
        'movie_to_entity_ratio': 4 / 4,
        'keep_all_ratings': False,
        'replace_movies_with_descriptive_entities': False,
        'n_negative_samples': 100
    }],
    ['substituting-3-0', {
        'movie_to_entity_ratio': 3 / 4,
        'keep_all_ratings': False,
        'replace_movies_with_descriptive_entities': False,
        'n_negative_samples': 100
    }],
    ['substituting-2-0', {
        'movie_to_entity_ratio': 2 / 4,
        'keep_all_ratings': False,
        'replace_movies_with_descriptive_entities': False,
        'n_negative_samples': 100
    }],
    ['substituting-1-0', {
        'movie_to_entity_ratio': 1 / 4,
        'keep_all_ratings': False,
        'replace_movies_with_descriptive_entities': False,
        'n_negative_samples': 100
    }],

    ['all_movies', {
        'movie_to_entity_ratio': 4 / 4,
        'keep_all_ratings': True,
        'replace_movies_with_descriptive_entities': False,
        'n_negative_samples': 100,
        'movies_only': True
    }],
    ['all_entities', {
        'movie_to_entity_ratio': 4 / 4,
        'keep_all_ratings': True,
        'replace_movies_with_descriptive_entities': False,
        'n_negative_samples': 100,
        'movies_only': False
    }],
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
                    loader = DesignatedDataLoader.load_from(
                        join('../data_loading', 'mindreader'),
                        min_num_entity_ratings=2,
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
                    loader = DesignatedDataLoader.load_from(
                        join('../data_loading', 'mindreader'),
                        min_num_entity_ratings=2,
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


if __name__ == '__main__':
    generate_with_top_pop(filter_unkowns=True)
    generate_without_top_pop(filter_unkowns=True)
    generate_with_top_pop(filter_unkowns=False)
    generate_without_top_pop(filter_unkowns=False)