import json
import numpy as np
import os
from loguru import logger


def n_not_in_train(xs, train_ratings):
    train_movies = set([r['e_idx'] for r in train_ratings if r['is_movie_rating']])
    n = 0
    for o in xs:
        if o not in train_movies:
            n += 1

    return n


if __name__ == '__main__':
    base_dir = 'datasets_no_top_pop'
    experiments = os.listdir(base_dir)

    for experiment in experiments:
        files = os.listdir(os.path.join(base_dir, experiment))
        last_val_pos = set()
        last_test_pos = set()
        logger.info(f'Statistics for {experiment}...')
        logger.info('--------------------------------------------------------------------')
        for file in files:
            logger.info(f'                                                                     {file}')
            with open(os.path.join(base_dir, experiment, file)) as fp:
                data = json.load(fp)

                training_users = []
                training_ratings = []
                validation_positives = []
                validation_negatives = []
                validation_users = []
                test_positives = []
                test_negatives = []
                test_users = []

                for u, ratings in data['training']:
                    training_users.append(u)
                    training_ratings += ratings
                for u, (pos, negs) in data['validation']:
                    validation_users.append(u)
                    validation_positives.append(pos)
                    validation_negatives += negs
                for u, (pos, negs) in data['testing']:
                    test_users.append(u)
                    test_positives.append(pos)
                    test_negatives += negs

                movie_ratings = [r for r in training_ratings if r['is_movie_rating']]
                de_ratings = [r for r in training_ratings if not r['is_movie_rating']]

                logger.info('TRAINING:')
                logger.info(f'  # ratings: {len(training_ratings)}')
                logger.info(f'  # users: {len(training_users)}')
                logger.info(f'  # movie ratings: {len(movie_ratings)}')
                logger.info(f'  # DE ratings: {len(de_ratings)}')
                logger.info('VALIDATION:')
                logger.info(f'  # users: {len(validation_users)}')
                logger.info(f'  # positives in total: {len(set(validation_positives))}')
                logger.info(f'  # positives not in previous: {len(set(validation_positives) - last_val_pos)}')
                logger.info(f'  # positives not in train: {n_not_in_train(set(validation_positives), training_ratings)}')
                logger.info(f'  # negatives not in train: {n_not_in_train(set(validation_negatives), training_ratings)}')
                logger.info('TESTING:')
                logger.info(f'  # users: {len(test_users)}')
                logger.info(f'  # positives not in previous: {len(set(test_positives) - last_test_pos)}')
                logger.info(f'  # positives not in train: {n_not_in_train(set(test_positives), training_ratings)}')
                logger.info(f'  # negatives not in train: {n_not_in_train(set(test_negatives), training_ratings)}')
                last_val_pos = set(validation_positives)
                last_test_pos = set(test_positives)
