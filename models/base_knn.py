from loguru import logger

from models.base_recommender import RecommenderBase
import numpy as np
from loguru import logger


class BaseKNN(RecommenderBase):
    def __init__(self, split, n_xs, n_ys):
        super(BaseKNN).__init__()
        self.split = split
        self.n_xs = n_xs
        self.n_ys = n_ys
        self.entity_vectors = np.zeros((n_xs, n_ys))
        self.plain_entity_vectors = np.zeros((n_xs, n_ys))
        self.pearson_entity_vectors = np.zeros((n_xs, n_ys))
        self.user_ratings = {}
        self.k = 1

    def _cosine_similarity(self, user, user_k, eps=1e-8):
        raise NotImplementedError

    def _set_self(self, configuration):
        raise NotImplementedError

    def _fit_pred(self, cur_config, best_config, validation, verbose):
        cur_config = cur_config.copy()
        best_config = best_config.copy()

        self._set_self(cur_config)

        hits = 0
        for user, (pos_sample, neg_samples) in validation:
            samples = neg_samples + [pos_sample]
            score = self.predict(user, samples)

            score = sorted(score.items(), key=lambda x: x[1], reverse=True)[:10]
            score = [i for i, _ in score]
            if pos_sample in score:
                hits += 1

        hit_rate = hits / len(validation)
        cur_config['hit_rate'] = hit_rate

        if best_config['hit_rate'] < hit_rate:
            best_config = cur_config.copy()

        if verbose:
            logger.debug(cur_config)

        return best_config

    def optimize_k(self, cur_config, best_config, validation, k_range, verbose):
        cur_config = cur_config.copy()
        # Optimize k
        for k in k_range:
            cur_config['k'] = k
            best_config = self._fit_pred(cur_config, best_config, validation, verbose)

        return best_config
