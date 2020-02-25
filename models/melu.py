from collections import OrderedDict
from copy import deepcopy

import torch as tt
from torch import nn
from torch.nn import functional as F


class MeLU(nn.Module):
    def __init__(self, n_entities, n_decade, n_movies, n_categories, n_persons, n_companies, latent_factor):
        super(MeLU, self).__init__()
        self.entity_emb = nn.Linear(n_entities, latent_factor, bias=False)
        self.decade_emb = nn.Linear(n_decade, latent_factor, bias=False)
        self.movie_emb = nn.Linear(n_movies, latent_factor, bias=False)
        self.category_emb = nn.Linear(n_categories, latent_factor, bias=False)
        self.person_emb = nn.Linear(n_persons, latent_factor, bias=False)
        self.company_emb = nn.Linear(n_companies, latent_factor, bias=False)

        self.fc1 = nn.Linear(latent_factor*6, 64)
        self.fc2 = nn.Linear(64, 64)
        self.linear_out = nn.Linear(64, 1)

        self.activation = F.relu
        self.implicit_activation = F.sigmoid

    def _get_embeddings(self, entity_idxs, decade_idxs, movie_idxs, category_idxs, person_idxs, company_idxs):
        entities = self.entity_emb(entity_idxs).view(-1, 1)
        decades = self.entity_emb(decade_idxs).view(-1, 1)
        movies = self.entity_emb(movie_idxs).view(-1, 1)
        categories = self.entity_emb(category_idxs).view(-1, 1)
        persons = self.entity_emb(person_idxs).view(-1, 1)
        companies = self.entity_emb(company_idxs).view(-1, 1)

        return tt.cat((entities, decades, movies, categories, persons, companies), 1)

    def forward(self, entity_idxs, decade_idxs, movie_idxs, category_idxs, person_idxs, company_idxs, is_support=True):
        if is_support:
            with tt.no_grad():
                concatenated = self._get_embeddings(entity_idxs, decade_idxs, movie_idxs, category_idxs,
                                                    person_idxs, company_idxs)
        else:
            concatenated = self._get_embeddings(entity_idxs, decade_idxs, movie_idxs, category_idxs,
                                                person_idxs, company_idxs)

        x = self.fc1(concatenated)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        return self.implicit_activation(self.linear_out(x))
