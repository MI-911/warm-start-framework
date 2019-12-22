import operator

from sklearn.utils.extmath import randomized_svd

from models.base_recommender import RecommenderBase
from utility.utility import csr, get_combinations
import scipy
import numpy as np
from loguru import logger


class SVDRecommender(RecommenderBase):
    def __init__(self):
        super().__init__()
        self.ratings = None
        self.U = None
        self.V = None

    def _recommend(self, users):
        return np.dot(self.U[users, :], self.V.T)

    def _fit(self, training, factors, only_positive):
        self.ratings = csr(training, only_positive)

        U, sigma, VT = randomized_svd(self.ratings, factors)
        sigma = scipy.sparse.diags(sigma, 0)
        self.U = U * sigma
        self.V = VT.T

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        parameters = {
            'factors': [1, 2, 3],
            'only_positive': [True, False]
        }

        combinations = get_combinations(parameters)
        logger.info(f'{len(combinations)} hyperparameter combinations')

        self.ratings = csr(training)

        results = list()
        for combination in combinations:
            logger.info(f'Trying {combination}')

            self._fit(training, **combination)

            hits, count = 0, 0

            for user, validation_tuple in validation:
                to_find, negative = validation_tuple

                scores = self.predict(user, [to_find] + negative)
                top_k = [item[0] for item in sorted(scores.items(), key=operator.itemgetter(1), reverse=True)][:10]

                if to_find in top_k:
                    hits += 1
                count += 1

            logger.info(f'Hit: {hits / count * 100:.2f}%')
            results.append((combination, hits / count))

        best = sorted(results, key=operator.itemgetter(1), reverse=True)[0]
        logger.info(f'Best: {best}')

        self._fit(training, **best[0])

    def predict(self, user, items):
        scores = self._recommend(user)
        item_scores = sorted(list(enumerate(scores)), key=operator.itemgetter(1), reverse=True)

        return {index: score for index, score in item_scores if index in items}
