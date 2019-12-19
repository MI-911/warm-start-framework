import operator

from sklearn.utils.extmath import randomized_svd

from models.base_recommender import RecommenderBase
from utility.utility import csr
import scipy
import numpy as np


class SVDRecommender(RecommenderBase):
    def __init__(self):
        super().__init__()
        self.ratings = None
        self.k = 1
        self.U = None
        self.V = None

    def _recommend(self, users):
        return np.dot(self.U[users, :], self.V.T)

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        U, sigma, VT = randomized_svd(csr(training), self.k)
        sigma = scipy.sparse.diags(sigma, 0)
        self.U = U * sigma
        self.V = VT.T

    def predict(self, user, items):
        scores = self._recommend(user)
        item_scores = sorted(list(enumerate(scores)), key=operator.itemgetter(1), reverse=True)

        return {index: score for index, score in item_scores if index in items}
