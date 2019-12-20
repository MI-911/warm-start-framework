"""
Joint Matrix Factorization
"""

import torch as tt
from torch import nn


class JointMF(nn.Module):
    def __init__(self, user_count, item_count, latent_factors):
        super(JointMF, self).__init__()
        self.n_users = user_count
        self.n_items = item_count

        self.users = nn.Embedding(user_count, latent_factors)
        self.items = nn.Embedding(item_count, latent_factors)
        self.contexts = nn.Embedding(item_count, latent_factors)

        self.device = tt.device('cuda:0') if tt.cuda.is_available() else tt.device('cpu')

        self.to(self.device)

        self.loss_fn = nn.MSELoss()

    def params_to(self, a, b, c):
        return tt.tensor(a).to(self.device), tt.tensor(b).to(self.device), tt.tensor(c).to(self.device)

    def forward(self, user_id, item_id, rating, is_rating=False):
        if is_rating:
            user_id, item_id, rating = self.params_to(user_id, item_id, rating)
            predictions = (self.users(user_id) * self.items(item_id)).sum(dim=1)
            return self.loss_fn(rating, predictions)
        else:
            item_id, context_id, sppmi = self.params_to(user_id, item_id, rating)
            predictions = (self.items(item_id) * self.contexts(context_id)).sum(dim=1)
            return self.loss_fn(sppmi, predictions)

    def predict(self, user_id, item_indices=None):
        if item_indices is None:
            item_indices = range(self.n_items)
        item_indices = tt.tensor(item_indices).to(self.device)
        user_id = tt.tensor(user_id).to(self.device)
        return (self.users(user_id) * self.items(item_indices)).sum(dim=1)
