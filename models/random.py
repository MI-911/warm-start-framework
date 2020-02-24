from models.base_recommender import RecommenderBase
import numpy as np


class RandomRecommender(RecommenderBase):
    def __init__(self):
        super(RandomRecommender, self).__init__(None)
        self.optimal_params = {'x': 1}

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        return

    def predict(self, user, items):
        # Assign a random score to every item
        scores = np.random.rand(len(items))
        return {i: s for i, s in zip(items, scores)}
