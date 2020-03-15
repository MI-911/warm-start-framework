from models.base_recommender import RecommenderBase
from models.entity2rec.entity2rec import Entity2Rec
from models.entity2rec.evaluator import Evaluator


class Entity2RecRecommender(RecommenderBase):
    def __init__(self, split):
        RecommenderBase.__init__(self)

        self.model = None
        self.optimal_params = []
        self.split = split
        self.evaluater = Evaluator(implicit=True)

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        self.model = Entity2Rec(self.split, training, run_all=True)

        x_train, y_train, qids_train, items_train, \
            x_val, y_val, qids_val, items_val = self.evaluater.features(self.model, training, validation)

        self.model.fit(x_train, y_train, qids_train,
                  x_val=x_val, y_val=y_val, qids_val=qids_val, optimize='NDCG', n_estimators=1000)

    def predict(self, user, items):
        x, y, qids, items = self.evaluater._compute_features('test', self.model, [[user, items]])
        preds = self.model.predict(x, qids)
        return {i: p for i, p in zip(items, preds)}
