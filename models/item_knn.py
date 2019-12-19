from tqdm import tqdm

from data_loading.loo_data_loader import DesignatedDataLoader
from models.base_recommender import RecommenderBase
import numpy as np


class ItemKNN(RecommenderBase):
    def __init__(self, data_loader):
        super(ItemKNN).__init__()
        self.n_entities = len(data_loader.e_idx_map)
        self.data_loader = data_loader
        self.entity_vectors = np.zeros((self.n_entities, data_loader.n_users))
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
                self.entity_vectors[rating.e_idx][user] = rating.rating

        hit_k = {}
        for k in range(1, max_iterations):
            self.k = k
            hits = 0
            for user, (pos_sample, neg_samples) in validation:
                samples = neg_samples + [pos_sample]
                score = self.predict(user, samples)

                score = sorted(score.items(), key=lambda x: x[1], reverse=True)[:10]
                score = [i for i, _ in score]
                if pos_sample in score:
                    hits += 1

            cur_hitrate = hits / len(validation)
            if verbose:
                print(f'Hitrate: {cur_hitrate} for k={k}')

            hit_k[k] = cur_hitrate

        optimal = sorted(hit_k.items(), key=lambda x: x[1])[-1]
        self.k = optimal[0]
        print(f'Found optimal number of neighbors to be {self.k} with hitrate {optimal[1]}')

    def _cosine_similarity(self, samples, ratings, eps=1e-8):
        sample_vecs = self.entity_vectors[samples]
        rating_vecs = self.entity_vectors[ratings]
        top = np.einsum('ij,kj->ik', sample_vecs, rating_vecs)
        samples_norm = np.sqrt(np.sum(sample_vecs ** 2, axis=1))
        entity_norm = np.sqrt(np.sum(rating_vecs ** 2, axis=1))
        bottom = np.maximum(np.einsum('i,k->ik', samples_norm, entity_norm), eps)

        return top / bottom

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
            score[items[index]] = np.einsum('i,i->', *zip(*topk))

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
        movie_to_entity_ratio=4/4,
        replace_movies_with_descriptive_entities=replace_movies_with_descriptive_entities,
        n_negative_samples=100,
        keep_all_ratings=False
    )

    knn = ItemKNN(data_loader)

    knn.fit(tra, val)
