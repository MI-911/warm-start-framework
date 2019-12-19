from data_loading.loo_data_loader import DesignatedDataLoader
from models.base_knn import BaseKNN
from models.base_recommender import RecommenderBase
import numpy as np


class UserKNN(BaseKNN):
    def __init__(self, data_loader):
        super(UserKNN, self).__init__(data_loader, len(data_loader.e_idx_map), data_loader.n_users)
        self.mean_centered_ratings = np.zeros((self.data_loader.n_users, ))
        self.user_ratings = {}
        self.k = 1

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

        for user, ratings in training:
            self.user_ratings[user] = ratings
            for rating in ratings:
                self.plain_entity_vectors[user][rating.e_idx] = rating.rating

        # Calculate user average.
        for user, _ in training:
            self.mean_centered_ratings[user] = np.mean(self.plain_entity_vectors[user])

        # Set adjusted vectors
        for entity in range(self.n_xs):
            indices = np.where(self.plain_entity_vectors[:, entity] != 0)[0]
            for user in indices:
                self.pearson_entity_vectors[user][entity] = self.plain_entity_vectors[user][entity] - self.mean_centered_ratings[user]

        last_5 = np.array([True, True, True], dtype=np.bool)
        cur_index = 0
        best_outer_config = {'metric': 'cosine', 'k': 10, 'hitrate': -1}
        best_inner_config = {'metric': 'cosine', 'k': 10, 'hitrate': 0}
        iteration = 0
        while np.any(last_5) and iteration < max_iterations:
            iteration += 1
            cur_configuration = best_inner_config.copy()
            # Optimize func
            for func in ['cosine', 'pearson']:
                cur_configuration['metric'] = func
                best_inner_config = self._fit_pred(cur_configuration, best_inner_config, validation, verbose)

            cur_configuration = best_inner_config.copy()

            # Optimize k
            best_inner_config = self.optimize_k(cur_configuration, best_inner_config, validation,
                                                [1, 2, 4, 6, 8, 10, 15, 20, 25, 35, 45, 55], verbose)

            if best_inner_config['hitrate'] > best_outer_config['hitrate']:
                best_outer_config = best_inner_config.copy()
                last_5[cur_index] = True
                print(f'New best: {best_outer_config}')
            else:
                last_5[cur_index] = False

            cur_index = (cur_index + 1) % 3

        self._set_self(best_outer_config)

        if verbose:
            print(f'Found best configuration: {best_outer_config}')

    def _cosine_similarity(self, user, user_k, eps=1e-8):
        user_vecs = self.entity_vectors[user]
        user_sim_vec = self.entity_vectors[user_k]
        top = np.einsum('i,ji->j', user_vecs, user_sim_vec)
        samples_norm = np.sqrt(np.sum(user_vecs ** 2, axis=0))
        entity_norm = np.sqrt(np.sum(user_sim_vec ** 2, axis=1))
        bottom = np.maximum(samples_norm * entity_norm, eps)

        return top / bottom

    def _set_self(self, configuration):
        self.k = configuration['k']
        if configuration['metric'] == 'cosine':
            self.entity_vectors = self.plain_entity_vectors.copy()
        elif configuration['metric'] == 'pearson':
            self.entity_vectors = self.pearson_entity_vectors.copy()

    def predict(self, user, items):
        """
        Predicts a score for all items given a user.
        :param user: int - user index
        :param items: List<int> - list of item indices
        :return: Dictionary<int, float> - A mapping from item indices to their score.
        """

        score = {}

        for item in items:
            related = np.where(self.entity_vectors[:, item] != 0)[0]

            if related.size == 0:
                score[item] = 0
                continue

            cs = self._cosine_similarity(user, related)

            topk = sorted([(r, s) for r, s in zip(related, cs)], key=lambda x: x[1], reverse=True)[:self.k]
            ratings = [(self.entity_vectors[i][item], sim) for i, sim in topk]
            score[item] = np.einsum('i,i->', *zip(*ratings))

        # A high score means item knn is sure in a positive prediction.
        return score


if __name__ == '__main__':
    data_loader = DesignatedDataLoader.load_from(
        path='../data_loading/mindreader',
        min_num_entity_ratings=5,
        movies_only=False,
        unify_user_indices=False
    )

    data_loader.random_seed = 1
    replace_movies_with_descriptive_entities = True

    tra, val, te = data_loader.make(
        movie_to_entity_ratio=1/4,
        replace_movies_with_descriptive_entities=replace_movies_with_descriptive_entities,
        n_negative_samples=100,
        keep_all_ratings=False
    )

    knn = UserKNN(data_loader)

    knn.fit(tra, val)