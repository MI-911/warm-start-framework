from models.base_recommender import RecommenderBase
from models.entity2rec.entity2rec import Entity2Rec
from models.entity2rec.evaluator import Evaluator


class Entity2RecRecommender(RecommenderBase):
    def __init__(self, split):
        RecommenderBase.__init__(self)
        self.model = Entity2Rec(split, run_all=True)
        self.evaluater = Evaluator(implicit=True)

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        x_train, y_train, qids_train, items_train, x_test, y_test, qids_test, items_test, \
            x_val, y_val, qids_val, items_val = self.evaluater.features(self.model, training, None, n_jobs=1)
        pass

    def predict(self, user, items):
        pass
