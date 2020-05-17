from data_loading.loo_data_loader import LeaveOneOutDataLoader
from models.base_knn import BaseKNN
import numpy as np
from loguru import logger


class ItemKNNRecommender(BaseKNN):
    def __init__(self, split):
        super(ItemKNNRecommender, self).__init__(split, split.n_entities, split.n_users)
        self.entity_vectors = np.zeros(())
        self.user_adjusted_entity_vectors = np.zeros((self.n_xs, self.n_ys), dtype=np.float64)
        self.user_average = np.zeros(())
        self.use_shrunk_similarity = True
        self.shrink_factor = 100
        self.shrunk_similarity = np.zeros((self.n_xs, self.n_xs))

        self.optimal_params = None

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        """
        Fits the model to the training data.
        :param training: List<int, List<Rating>> - List of (user_index, ratings) pairs
        :param validation: List<int, (int, List<int>)> - List of (user_index, (pos_index, neg_indices)) pairs
        :param max_iterations: (Optional) int - To ensure that the fitting process will stop eventually.
        :param verbose: (Optional) boolean - Controls whether statistics are written to stdout during training.
        :param save_to: (Optional) string - The full path for the file in which to save during-training metrics.
        :return: None
        """

        # Insert user ratings in entity_vectors
        for user, ratings in training:
            self.user_ratings[user] = ratings
            for rating in ratings:
                self.plain_entity_vectors[rating.e_idx][user] = rating.rating

        # Calculate user average.
        user_average = {}
        for user, _ in training:
            user_average[user] = np.mean(self.plain_entity_vectors[:, user])

        # Set adjusted vectors
        for entity in range(self.n_xs):
            indices = np.where(self.plain_entity_vectors[entity] != 0)[0]
            for user in indices:
                self.user_adjusted_entity_vectors[entity][user] = self.plain_entity_vectors[entity][user] - user_average[user]

        # Calculate item average
        entity_average = {}
        for entity in range(self.n_xs):
            entity_average[entity] = np.mean(self.plain_entity_vectors[entity])

        # Set pearson vectors
        for entity in range(self.n_xs):
            indices = np.where(self.plain_entity_vectors[entity] != 0)[0]
            for user in indices:
                self.pearson_entity_vectors[entity][user] = self.plain_entity_vectors[entity][user] - entity_average[entity]

        # Calculate num co-rated
        zeros = np.zeros((self.n_xs, self.split.n_users))
        for i in range(self.n_xs):
            i_vector = self.plain_entity_vectors[i]
            i_matrix = np.zeros((self.n_xs, self.split.n_users))
            i_matrix[:] = i_vector
            zeros_filter = np.not_equal(zeros, i_matrix)
            equal_filter = np.equal(i_matrix, self.plain_entity_vectors)
            sim_i_j = np.sum(np.equal(zeros_filter, equal_filter), axis=1)

            self.shrunk_similarity[i] = sim_i_j

        if self.optimal_params is None:
            last_better = True
            best_outer_config = {'metric': 'cosine', 'k': 10, 'hit_rate': -1, 'use_shrunk': False, 'shrink_factor': 100}
            best_inner_config = {'metric': 'cosine', 'k': 10, 'hit_rate': 0, 'use_shrunk': False, 'shrink_factor': 100}
            iteration = 0
            while last_better and iteration < max_iterations:
                iteration += 1
                cur_configuration = best_inner_config.copy()
                # Optimize func
                for func in ['cosine', 'adjusted_cosine', 'pearson']:
                    cur_configuration['metric'] = func
                    best_inner_config = self._fit_pred(cur_configuration, best_inner_config, validation, verbose)

                cur_configuration = best_inner_config.copy()

                # Optimize k
                best_inner_config = self.optimize_k(cur_configuration, best_inner_config, validation,
                                                    [1, 2, 4, 6, 8, 10, 15, 20, 25, 35, 45, 55], verbose)

                # Find hit without shrunk sim
                cur_configuration['use_shrunk'] = False
                self._set_self(cur_configuration)
                no_shrink_hitrate = self._fit_pred(cur_configuration, best_inner_config, validation, verbose)['hit_rate']

                # Optimize shrunk sim
                cur_configuration['use_shrunk'] = True
                for s in [1, 10, 25, 50, 100, 150, 200, 250, 300]:
                    cur_configuration['shrink_factor'] = s
                    best_inner_config = self._fit_pred(cur_configuration, best_inner_config, validation, verbose)

                # Select best of with and without shrunk
                if best_inner_config['use_shrunk'] and best_inner_config['hit_rate'] < no_shrink_hitrate:
                    best_inner_config['use_shrunk'] = False
                    best_inner_config['hit_rate'] = no_shrink_hitrate

                if best_inner_config['hit_rate'] > best_outer_config['hit_rate']:
                    best_outer_config = best_inner_config.copy()
                    logger.info(f'New best: {best_outer_config}')
                else:
                    last_better = False

            self._set_self(best_outer_config)

            if verbose:
                logger.info(f'Found best configuration: {best_outer_config}')

            self.optimal_params = best_outer_config
        else:
            logger.info(f'Reusing params {self.optimal_params}')
            self._set_self(self.optimal_params)

    def _set_self(self, configuration):
        self.k = configuration['k']
        if configuration['metric'] == 'cosine':
            self.entity_vectors = self.plain_entity_vectors.copy()
        elif configuration['metric'] == 'adjusted_cosine':
            self.entity_vectors = self.user_adjusted_entity_vectors.copy()
        elif configuration['metric'] == 'pearson':
            self.entity_vectors = self.pearson_entity_vectors.copy()

        if configuration['use_shrunk']:
            self.use_shrunk_similarity = True
            self.shrink_factor = configuration['shrink_factor']
        else:
            self.use_shrunk_similarity = False

    def _cosine_similarity(self, samples, ratings, eps=1e-8):
        sample_vecs = self.entity_vectors[samples]
        rating_vecs = self.entity_vectors[ratings]
        top = np.einsum('ij,kj->ik', sample_vecs, rating_vecs)
        samples_norm = np.sqrt(np.sum(sample_vecs ** 2, axis=1))
        entity_norm = np.sqrt(np.sum(rating_vecs ** 2, axis=1))
        bottom = np.maximum(np.einsum('i,k->ik', samples_norm, entity_norm), eps)

        res = top / bottom

        if self.use_shrunk_similarity:
            ss_top = self.shrunk_similarity[samples][:, ratings]
            ss_bottom = ss_top + self.shrink_factor
            res = (ss_top / ss_bottom) * res

        return res

    def predict(self, user, items):
        """
        Predicts a score for all items given a user.
        :param user: int - user index
        :param items: List<int> - list of item indices
        :return: Dictionary<int, float> - A mapping from item indices to their score.
        """

        rating_idx = [rating.e_idx for rating in self.user_ratings[user]]
        ratings = [rating.rating for rating in self.user_ratings[user]]
        cs = self._cosine_similarity(items, rating_idx)
        score = {}
        for index, similarities in enumerate(cs):
            topk = sorted([(r, s) for r, s in zip(ratings, similarities)], key=lambda x: x[1], reverse=True)[:self.k]
            try:
                score[items[index]] = np.einsum('i,i->', *zip(*topk))
            except ValueError:
                score[items[index]] = 0

        # A high score means item knn is sure in a positive prediction.
        return score


if __name__ == '__main__':
    data_loader = LeaveOneOutDataLoader.load_from(
        path='../data_loading/mindreader',
        min_num_entity_ratings=5,
        movies_only=False,
        unify_user_indices=False
    )

    data_loader.random_seed = 1
    replace_movies_with_descriptive_entities = True

    tra, val, te = data_loader.make(
        movie_to_entity_ratio=4/4,
        replace_movies_with_descriptive_entities=replace_movies_with_descriptive_entities,
        n_negative_samples=100,
        keep_all_ratings=False
    )

    knn = ItemKNNRecommender(data_loader)

    knn.fit(tra, val)
