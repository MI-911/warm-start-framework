from models.base_recommender import RecommenderBase
import numpy as np


class UserKNNRecommender(RecommenderBase):
    def __init__(self):
        super().__init__()

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        # Try varying sizes of k
        # Try varying shrink values

        return

    def predict(self, user, items):
        return
