from collections import OrderedDict
from copy import deepcopy

import torch as tt
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np


class Item(nn.Module):
    def __init__(self, n_decade, n_movies, n_categories, n_persons, n_companies, latent_factor):
        super(Item, self).__init__()
        self.decade_emb = nn.Linear(n_decade, latent_factor, bias=False)
        self.movie_emb = nn.Linear(n_movies, latent_factor, bias=False)
        self.category_emb = nn.Linear(n_categories, latent_factor, bias=False)
        self.person_emb = nn.Linear(n_persons, latent_factor, bias=False)
        self.company_emb = nn.Linear(n_companies, latent_factor, bias=False)

    def forward(self, decade_idxs, movie_idxs, category_idxs, person_idxs, company_idxs):
        decades, s = self.decade_emb(decade_idxs.float()), tt.sum(decade_idxs.float(), 1).view(-1, 1)
        decades[np.flatnonzero(s)] /= s[np.flatnonzero(s)].view(-1, 1)

        movies, s = self.movie_emb(movie_idxs.float()), tt.sum(movie_idxs.float(), 1)
        decades[np.flatnonzero(s)] /= s[np.flatnonzero(s)].view(-1, 1)

        categories, s = self.category_emb(category_idxs.float()), tt.sum(category_idxs.float(), 1)
        categories[np.flatnonzero(s)] /= s[np.flatnonzero(s)].view(-1, 1)

        persons, s = self.person_emb(person_idxs.float()), tt.sum(person_idxs.float(), 1)
        persons[np.flatnonzero(s)] /= s[np.flatnonzero(s)].view(-1, 1)

        companies, s = self.company_emb(company_idxs.float()), tt.sum(company_idxs.float(), 1)
        companies[np.flatnonzero(s)] /= s[np.flatnonzero(s)].view(-1, 1)

        return tt.cat((decades, movies, categories, persons, companies), 1)


class User(nn.Module):
    def __init__(self, n_entities, latent_factor):
        super(User, self).__init__()
        self.entity_emb = nn.Linear(n_entities, latent_factor, bias=False)

    def forward(self, entity_idxs):
        entities, s = self.entity_emb(entity_idxs.float()), tt.sum(entity_idxs.float(), 1)
        entities[np.flatnonzero(s)] /= s[np.flatnonzero(s)].view(-1, 1)

        return entities


class MeLU(nn.Module):
    def __init__(self, n_entities, n_decade, n_movies, n_categories, n_persons, n_companies, latent_factor):
        super(MeLU, self).__init__()
        self.n_entities, self.n_decade, self.n_movies, self.n_categories, self.n_persons, self.n_companies = \
            n_entities, n_decade, n_movies, n_categories, n_persons, n_companies

        self.item_emb = Item(n_decade, n_movies, n_categories, n_persons, n_companies, latent_factor)
        self.user_emb = User(n_entities, latent_factor)

        self.fc1 = nn.Linear(latent_factor*6, 64)
        self.fc2 = nn.Linear(64, 64)
        self.linear_out = nn.Linear(64, 1)

        self.activation = F.relu
        self.implicit_activation = tt.tanh

    def forward(self, x):
        s = self._split(x)
        item_emb = self.item_emb(*s[1:])
        user_emb = self.user_emb(s[0])
        concatenated = tt.cat((item_emb, user_emb), 1)

        x = self.fc1(concatenated)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        return self.linear_out(x)

    def _split(self, x):
        start, stop = 0, self.n_entities
        entity = Variable(x[:, start:stop], requires_grad=False)

        start = stop
        stop += self.n_decade
        decade = Variable(x[:, start:stop], requires_grad=False)

        start = stop
        stop += self.n_movies
        movie = Variable(x[:, start:stop], requires_grad=False)

        start = stop
        stop += self.n_categories
        category = Variable(x[:, start:stop], requires_grad=False)

        start = stop
        stop += self.n_persons
        person = Variable(x[:, start:stop], requires_grad=False)

        start = stop
        stop += self.n_companies
        company = Variable(x[:, start:stop], requires_grad=False)
        return entity, decade, movie, category, person, company
