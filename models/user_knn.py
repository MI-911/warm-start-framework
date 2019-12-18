from data_loading.loo_data_loader import DesignatedDataLoader
from models.base_recommender import RecommenderBase
import numpy as np


class UserKNN(RecommenderBase):
    def __init__(self, data_loader, k=10):
        super(UserKNN).__init__()
        self.n_entities = len(data_loader.e_idx_map)
        self.data_loader = data_loader
        self.entity_vectors = np.zeros((self.n_entities, data_loader.n_users)).transpose()
        self.user_ratings = {}
        self.k = k

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
                self.entity_vectors[user][rating.e_idx] = rating.rating

        hit = 0
        for user, (pos_sample, neg_samples) in validation:
            samples = neg_samples + [pos_sample]
            score = self.predict(user, samples)

            score = sorted(score.items(), key=lambda x: x[1], reverse=True)[:10]
            score = [i for i, _ in score]
            if pos_sample in score:
                hit += 1

        print(hit / len(validation))

    def _cosine_similarity(self, user, user_k, eps=1e-8):
        user_vecs = self.entity_vectors[user]
        user_sim_vec = self.entity_vectors[user_k]
        top = np.einsum('i,ji->j', user_vecs, user_sim_vec)
        samples_norm = np.sqrt(np.sum(user_vecs ** 2, axis=0))
        entity_norm = np.sqrt(np.sum(user_sim_vec ** 2, axis=1))
        bottom = np.maximum(samples_norm * entity_norm, eps)

        return top / bottom

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
        movie_to_entity_ratio=2/4,
        replace_movies_with_descriptive_entities=replace_movies_with_descriptive_entities,
        n_negative_samples=100,
        keep_all_ratings=False
    )

    knn = UserKNN(data_loader)

    knn.fit(tra, val)