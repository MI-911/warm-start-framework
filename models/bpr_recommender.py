import itertools as it
import operator

from loguru import logger

from models.base_recommender import RecommenderBase
from models.bpr import BPR, csr


def get_combinations(parameters):
    keys, values = zip(*parameters.items())
    return [dict(zip(keys, v)) for v in it.product(*values)]


class BPRRecommender(RecommenderBase):
    def __init__(self):
        super().__init__()
        self.ratings = None

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        parameters = {
            'reg': [0.001, 0.002],
            'learning_rate': [0.15, 0.175],
            'n_iters': [200, 250],
            'n_factors': [1],
            'batch_size': [16, 32]
        }

        results = list()
        combinations = get_combinations(parameters)
        logger.info(f'{len(combinations)} parameters to search')

        self.ratings = csr(training)

        for combination in combinations:
            logger.info(combination)

            self.model = BPR(**combination)
            self.model.fit(self.ratings)

            hits = 0
            count = 0

            for user, validation_tuple in validation:
                to_find, negative = validation_tuple

                scores = self.predict(user, [to_find] + negative)
                top_k = [item[0] for item in sorted(scores.items(), key=operator.itemgetter(1), reverse=True)][:10]

                if to_find in top_k:
                    hits += 1
                count += 1

            logger.info(f'Hit: {hits / count * 100}')
            results.append((combination, hits / count))

        best = sorted(results, key=operator.itemgetter(1), reverse=True)[0]
        logger.info(f'Best: {best}')

        self.model = BPR(**best[0])
        self.model.fit(self.ratings)

    def predict(self, user, items):
        scores = self.model.predict_user(user)

        return {item: scores[item] for item in items}
