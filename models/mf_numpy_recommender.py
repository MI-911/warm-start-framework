from models.base_recommender import RecommenderBase
from models.mf_numpy import MatrixFactorisation
import numpy as np
import random
from loguru import logger
from utility.utility import get_combinations


class MatrixFactorisationRecommender(RecommenderBase):
    def __init__(self, split):
        super(MatrixFactorisationRecommender, self).__init__()
        self.split = split
        self.optimal_params = None

    def convert_rating(self, rating):
        if rating == 1:
            return 1
        elif rating == -1:
            return 0
        elif rating == 0:
            # We can make a choice here - either return 0 or 0.5
            return 0.5

    def batches(self, triples, n=64):
        for i in range(0, len(triples), n):
            yield triples[i:i + n]

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        hit_rates = []

        if self.optimal_params is None:
            parameters = {
                'k': [1, 2, 5, 10, 15, 25, 50]
            }
            for params in get_combinations(parameters):
                logger.debug(f'Fitting MF with params: {params}')
                self.model = MatrixFactorisation(self.split.n_users,
                                                 self.split.n_movies + self.split.n_descriptive_entities,
                                                 params['k'])
                hit_rates.append((self._fit(training, validation, max_iterations), params))

            hit_rates = sorted(hit_rates, key=lambda x: x[0], reverse=True)
            _, best_params = hit_rates[0]

            self.optimal_params = best_params

            self.model = MatrixFactorisation(self.split.n_users,
                                             self.split.n_movies + self.split.n_descriptive_entities,
                                             self.optimal_params['k'])
            logger.info(f'Found best parameters for MF: {self.optimal_params}')
            self._fit(training, validation)
        else:
            self.model = MatrixFactorisation(self.split.n_users,
                                             self.split.n_movies + self.split.n_descriptive_entities,
                                             self.optimal_params['k'])
            self._fit(training, validation)

    def _fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        # Preprocess training data
        training_triples = []
        for u, ratings in training:
            for r in ratings:
                rating = self.convert_rating(r.rating)
                training_triples.append((u, r.e_idx, rating))

        validation_history = []

        for epoch in range(max_iterations):
            random.shuffle(training_triples)
            self.model.train_als(training_triples)

            if epoch % 10 == 0:
                ranks = []
                for user, (pos, negs) in validation:
                    predictions = self.model.predict(user, negs + [pos])
                    predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                    predictions = {i: rank for rank, (i, s) in enumerate(predictions)}
                    ranks.append(predictions[pos])

                _hit = np.mean([1 if r < 10 else 0 for r in ranks])
                validation_history.append(_hit)

                if verbose:
                    logger.info(f'Hit@10 at epoch {epoch}: {np.mean([1 if r < 10 else 0 for r in ranks])}')

        return np.mean(validation_history[-10:])

    def predict(self, user, items):
        return self.model.predict(user, items)
