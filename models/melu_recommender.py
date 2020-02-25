from collections import OrderedDict
from copy import deepcopy

from models.base_recommender import RecommenderBase
import torch as tt

from models.melu import MeLU


class MeLURecommender(RecommenderBase):
    def __init__(self, split):
        super(MeLURecommender, self).__init__()
        self.split = split
        self.model = MeLU(1, 2, 3, 4, 5, 6, 7)
        self.optimal_params = None
        self.store_parameters()
        self.meta_optim = tt.optim.Adam(self.model.parameters(), lr=5e-5)
        self.local_update_target_weight_name = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'linear_out.weight',
                                                'linear_out.bias']

    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())
        self.weight_name = list(self.keep_weight.keys())
        self.weight_len = len(self.keep_weight)
        self.fast_weights = OrderedDict()

    def get_counts(self):
        # TODO: Do
        pass

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        # TODO: Do
        pass

    def predict(self, user, items):
        # TODO: Do
        pass
