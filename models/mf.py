"""
Vanilla Matrix Factorization.
"""

import torch as tt
from torch import nn


class MF(nn.Module):
    def __init__(self, user_count, item_count, latent_factors):
        super(MF, self).__init__()
        self.n_users = user_count
        self.n_items = item_count

        self.users = nn.Embedding(user_count, latent_factors, sparse=True)
        self.items = nn.Embedding(item_count, latent_factors, sparse=True)

        self.device = tt.device('cuda') if tt.cuda.is_available() else tt.device('cpu')

        self.to(self.device)

    def forward(self, user_id, item_id):
        user_id = tt.tensor(user_id).to(self.device)
        item_id = tt.tensor(item_id).to(self.device)
        return (self.users(user_id) * self.items(item_id)).sum(1)

    def predict(self, user_id, item_indices=None):
        if item_indices is None:
            item_indices = tt.tensor(range(self.n_items)).to(self.device)

        user_id = tt.tensor(user_id).to(self.device)
        return self.users(user_id) * self.items(item_indices)
