from models.base_recommender import RecommenderBase
from models.entity2rec.entity2rec import Entity2Rec


class Entity2RecRecommender(RecommenderBase, Entity2Rec):
    def __init__(self, split):
        RecommenderBase.__init__(self)
        Entity2Rec.__init__(self, split, run_all=True)

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        pass

    def predict(self, user, items):
        pass
