import numpy as np 
import json 
import pandas as pd 
import os
from scipy.spatial.distance import cosine

from models.base_recommender import RecommenderBase


class CbfItemKnnRecommender(RecommenderBase): 
    def __init__(self, split):
        super(CbfItemKnnRecommender, self).__init__()

        self.split = split 
        self.n_entities = max(split.experiment.dataset.e_idx_map.values()) + 1
        self.n_users = split.n_users

        self.entity_vectors = np.zeros((self.n_entities, self.n_entities))
        self.user_ratings = np.zeros((self.n_users, self.n_entities))

        self.optimal_params = {'k': 100}

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

        # Load the KG. Entity vectors should be a one-hot encoding
        # of related 1-hop neighbours
        e_idx_map = self.split.experiment.dataset.e_idx_map
        for h, r, t in self._load_kg():
            if h not in e_idx_map or t not in e_idx_map:
                continue
            h = e_idx_map[h]
            t = e_idx_map[t]

            self.entity_vectors[h][t] = 1
            self.entity_vectors[t][h] = 1

        # Save users' ratings
        for user, ratings in training:
            for rating in ratings:
                self.user_ratings[user][rating.e_idx] = rating.rating

    def predict(self, user, items):
        """
        Predicts a score for all items given a user.
        :param user: int - user index
        :param items: List<int> - list of item indices
        :return: Dictionary<int, float> - A mapping from item indices to their score.
        """

        # For every item here, determine the item's similarity
        # to all other items that the user has rated.
        # Sort the items rated by the user by similarity.
        # Take the top-k.
        # Sum their similarity * rating.
        # This is the predicted rating.

        k = self.optimal_params['k']
        items_rated_by_user, = (self.user_ratings[user]).nonzero()

        predictions = []

        for item in items:
            sorted_neighbours = sorted([(other, 1 - cosine(self.entity_vectors[item], self.entity_vectors[other]))
                                        for other in items_rated_by_user], reverse=True)
            sorted_neighbours = list(sorted_neighbours)[:k]

            predicted_rating = sum([similarity * self.user_ratings[user][neighbour]
                                    for neighbour, similarity in sorted_neighbours])

            predictions.append((item, predicted_rating))

        return {item: prediction for item, prediction in predictions}

    def _load_kg(self):
        with open('data/triples.csv') as fp: 
            triples = pd.read_csv(fp)
            triples = [(h, r, t) for h, r, t in triples[['head_uri', 'relation', 'tail_uri']].values]
            return triples