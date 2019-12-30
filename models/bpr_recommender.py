import operator

from loguru import logger

from models.base_recommender import RecommenderBase
from models.bpr import BPR
from utility.utility import csr, get_combinations


class BPRRecommender(RecommenderBase):
    def __init__(self):
        super().__init__()
        self.ratings = None
        self.optimal_params = None

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        if not self.optimal_params:
            parameters = {
                'reg': [0.001],
                'learning_rate': [0.1],
                'n_iters': [250],
                'n_factors': [1, 2],
                'batch_size': [16, 32],
                'only_positive': [True, False]
            }

            results = list()
            combinations = get_combinations(parameters)
            logger.debug(f'{len(combinations)} hyperparameter combinations')

            for combination in combinations:
                logger.debug(combination)

                self.model = BPR(training, **combination)
                self.model.fit()

                hits = 0
                count = 0

                for user, validation_tuple in validation:
                    to_find, negative = validation_tuple

                    scores = self.predict(user, [to_find] + negative)
                    top_k = [item[0] for item in sorted(scores.items(), key=operator.itemgetter(1), reverse=True)][:10]

                    if to_find in top_k:
                        hits += 1
                    count += 1

                logger.debug(f'Hit: {hits / count * 100}')
                results.append((combination, hits / count))

            best = sorted(results, key=operator.itemgetter(1), reverse=True)[0][0]
            self.optimal_params = best

            logger.info(f'Found best: {best}')
        else:
            logger.debug(f'Using stored hyperparameters {self.optimal_params}')

        self.model = BPR(training, **self.optimal_params)
        self.model.fit()

    def predict(self, user, items):
        scores = self.model.predict_user(user)

        return {item: scores[item] for item in items}
