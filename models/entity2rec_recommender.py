import numpy as np
from pyltr.util.group import check_qids, get_groups
from pyltr.util.sort import get_sorted_y_positions

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
        self.model = Entity2Rec(self.split, training, run_all=True, p=1, q=4, walk_length=10, dimensions=500,
                                num_walks=500, window_size=10)

        # reads .dat format
        self.evaluater._parse_data(training, validation=None)

        users_list = self.evaluater._define_user_list(False, False, 8)

        # x_train, y_train, qids_train, items_train, \
        #     x_val, y_val, qids_val, items_val = self.evaluater.features(self.model, training, validation, n_jobs=1)

        # self.model.fit(x_train, y_train, qids_train,
        #                x_val=x_val, y_val=y_val, qids_val=qids_val,
        #                N=5, optimize='P', n_estimators=1000, lr=0.1)

    def predict(self, user, items):
        x, _, qids, _ = self.evaluater._compute_features('test', self.model, [[user, items]])
        preds = self.model.predict(x, qids)

        # x = np.mean(x, axis=1)
        return {i: p for i, p in zip(items, preds)}
