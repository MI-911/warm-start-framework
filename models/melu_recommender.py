from collections import OrderedDict
from copy import deepcopy
from random import shuffle

from loguru import logger
from torch.autograd import Variable

from models.base_recommender import RecommenderBase
import torch as tt
from torch.nn import functional as F


from models.melu import MeLU
import pandas as pd
import numpy as np


class MeLURecommender(RecommenderBase):
    def __init__(self, split):
        super(MeLURecommender, self).__init__()
        self.split = split

        self.decade_index, self.movie_index, self.category_index, self.person_index, \
            self.company_index = self._get_indices()

        self.entity_metadata = self._create_metadata()

        self.n_decade, self.n_movies, self.n_categories, self.n_persons, self.n_companies = len(self.decade_index), \
                       len(self.movie_index), len(self.category_index), len(self.person_index), len(self.company_index)

        self.model = MeLU(split.n_entities, self.n_decade, self.n_movies, self.n_categories, self.n_persons,
                          self.n_companies, 32)

        # Initialise variables
        self.optimal_params = None
        self.keep_weight = None
        self.weight_name = None
        self.weight_len = None
        self.fast_weights = None
        self.use_cuda = False
        self.local_lr = 5e-6

        self.store_parameters()
        self.meta_optim = tt.optim.Adam(self.model.parameters(), lr=5e-5)
        self.local_update_target_weight_name = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'linear_out.weight',
                                                'linear_out.bias']

    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())
        self.weight_name = list(self.keep_weight.keys())
        self.weight_len = len(self.keep_weight)
        self.fast_weights = OrderedDict()

    def _get_indices(self):
        df = pd.read_csv(self.split.experiment.dataset.triples_path)
        triples = [(h, r, t) for h, r, t in df[['head_uri', 'relation', 'tail_uri']].values]
        e_idx_map = self.split.experiment.dataset.e_idx_map

        decades = set()
        movies = set()
        categories = set()
        persons = set()
        companies = set()

        for h, r, t in triples:
            if h not in e_idx_map or t not in e_idx_map:
                continue

            if r != 'SUBCLASS_OF':
                movies.add(e_idx_map[h])
            else:
                categories.add(e_idx_map[t])

            if r == 'FROM_DECADE':
                decades.add(e_idx_map[t])
            elif r == 'DIRECTED_BY' or r == 'STARRING':
                persons.add(e_idx_map[t])
            elif r == 'HAS_SUBJECT' or r == 'HAS_GENRE':
                categories.add(e_idx_map[t])
            elif r == 'PRODUCED_BY':
                companies.add(e_idx_map[t])

        decade_index = {e_idx: i for i, e_idx in enumerate(decades)}
        movie_index = {e_idx: i for i, e_idx in enumerate(movies)}
        category_index = {e_idx: i for i, e_idx in enumerate(categories)}
        person_index = {e_idx: i for i, e_idx in enumerate(persons)}
        company_index = {e_idx: i for i, e_idx in enumerate(companies)}

        return decade_index, movie_index, category_index, person_index, company_index

    def _create_metadata(self):
        df = pd.read_csv(self.split.experiment.dataset.triples_path)
        triples = [(h, r, t) for h, r, t in df[['head_uri', 'relation', 'tail_uri']].values]
        e_idx_map = self.split.experiment.dataset.e_idx_map

        entities = {}

        def append_entities(e, h, t):
            if h not in e:
                e[h] = {'d': set(), 'm': set(), 'cat': set(), 'p': set(), 'com': set()}

            if t in self.decade_index:
                e[h]['d'].add(self.decade_index[t])
            if t in self.movie_index:
                e[h]['m'].add(self.movie_index[t])
            if t in self.category_index:
                e[h]['cat'].add(self.category_index[t])
            if t in self.person_index:
                e[h]['p'].add(self.person_index[t])
            if t in self.company_index:
                e[h]['com'].add(self.company_index[t])

        for h, _, t in triples:
            if h not in e_idx_map or t not in e_idx_map:
                continue

            h = e_idx_map[h]
            t = e_idx_map[t]

            append_entities(entities, h, t)
            append_entities(entities, t, h)

        entity_metadata = {e: [tt.tensor(list(metadata)) for _, metadata in m.items()] for e, m in entities.items()}

        return entity_metadata

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        user_ratings = {}
        for user, ratings in training:
            num_support = min(len(ratings) // 2, 10)
            shuffle(ratings)
            support = ratings[:num_support]
            query = ratings[num_support:]
            user_ratings[user] = [support, query]

        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []

        logger.debug(f'Creating train data')
        for user, (support, query) in user_ratings.items():
            # Create support data
            support_x = None
            for item in support:
                meta = self.entity_metadata[item.e_idx]
                meta = [tt.tensor([[0, x] for x in type]).t() for type in meta]
                onehots = self.to_onehot([], *meta)

                if support_x is None:
                    support_x = onehots
                else:
                    support_x = tt.cat((support_x, onehots), 0)

            support_y = tt.FloatTensor([r.rating for r in support])

            # Create query data
            query_x = None
            for item in query:
                meta = self.entity_metadata[item.e_idx]
                meta = [tt.tensor([[0, x] for x in type]).t() for type in meta]
                e = tt.tensor([[0, r.e_idx] for r in support if r.rating == 1]).t()
                onehots = self.to_onehot(e, *meta)

                if query_x is None:
                    query_x = onehots
                else:
                    query_x = tt.cat((query_x, onehots), 0)

            query_y = tt.FloatTensor([r.rating for r in query])

            support_xs.append(support_x)
            support_ys.append(support_y)

            user_ratings[user].append([support_x, support_y])

            query_xs.append(query_x)
            query_ys.append(query_y)

        train_data = list(zip(support_xs, support_ys, query_xs, query_ys))
        del support_xs, support_ys, query_xs, query_ys

        val = []
        logger.debug(f'Creating validation set')
        for user, (pos_sample, neg_samples) in validation:
            u_val = None
            support, _, support_train = user_ratings[user]
            samples = np.array([pos_sample] + neg_samples)
            shuffle(samples)
            rank = np.argwhere(samples == pos_sample)[0]

            for item in samples:
                meta = self.entity_metadata[item]
                meta = [tt.tensor([[0, x] for x in type]).t() for type in meta]
                e = tt.tensor([[0, r.e_idx] for r in support if r.rating == 1]).t()
                onehots = self.to_onehot(e, *meta)

                if u_val is None:
                    u_val = onehots
                else:
                    u_val = tt.cat((u_val, onehots), 0)

            val.append((rank, u_val, support_train))

        batch_size = 16
        n_batches = (len(train_data) // batch_size) + 1

        logger.debug('Starting training')

        # Go through all epochs
        for i in range(max_iterations):
            logger.debug(f'Starting epoch {i+1}')
            # Ensure random order
            shuffle(train_data)

            self.model.train()
            # go through all batches
            for batch_n in range(n_batches):
                batch = train_data[batch_size*batch_n:batch_size*(batch_n+1)]
                self.global_update(*zip(*batch))

            logger.debug('Starting validation')
            hit = 0.
            for rank, val_data, (support_x, support_y) in val:
                lst = np.arange(len(val_data))
                preds = self.forward(support_x, support_y, val_data, 1)
                ordered = sorted(zip(preds, lst), reverse=True)
                hit += 1. if rank in [r for _, r in ordered][:10] else 0.

            hitrate = hit / len(val)
            logger.debug(f'Hit at 10: {hitrate}')

    def forward(self, support_set_x, support_set_y, query_set_x, num_local_update):
        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            support_set_y_pred = self.model(*self._split(support_set_x))
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
            self.model.zero_grad()
            grad = tt.autograd.grad(loss, self.model.parameters(), create_graph=True)
            # local update
            for i in range(self.weight_len):
                if self.weight_name[i] in self.local_update_target_weight_name:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]
                else:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
        self.model.load_state_dict(self.fast_weights)
        query_set_y_pred = self.model(*self._split(query_set_x))
        self.model.load_state_dict(self.keep_weight)
        return query_set_y_pred

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys, num_local_update=1):
        batch_sz = len(support_set_xs)
        losses_q = []
        if self.use_cuda:
            for i in range(batch_sz):
                support_set_xs[i] = support_set_xs[i].cuda()
                support_set_ys[i] = support_set_ys[i].cuda()
                query_set_xs[i] = query_set_xs[i].cuda()
                query_set_ys[i] = query_set_ys[i].cuda()
        for i in range(batch_sz):
            query_set_y_pred = self.forward(support_set_xs[i], support_set_ys[i], query_set_xs[i], num_local_update)
            loss_q = F.mse_loss(query_set_y_pred, query_set_ys[i].view(-1, 1))
            losses_q.append(loss_q)
        losses_q = tt.stack(losses_q).mean(0)
        self.meta_optim.zero_grad()
        losses_q.backward()
        self.meta_optim.step()
        self.store_parameters()
        return

    def _split(self, x):
        start, stop = 0, self.split.n_entities
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

    def to_onehot(self, entities, decade, movie, category, person, company):
        entity = self.create_onehot(entities, (1, self.split.n_entities))
        decade = self.create_onehot(decade, (1, self.n_decade))
        movie = self.create_onehot(movie, (1, self.n_movies))
        category = self.create_onehot(category, (1, self.n_categories))
        person = self.create_onehot(person, (1, self.n_persons))
        company = self.create_onehot(company, (1, self.n_companies))
        return tt.cat((entity, decade, movie, category, person, company), 1)

    def create_onehot(self, indices, shape):
        t = tt.zeros(shape)
        if len(indices) > 0:
            t[indices.tolist()] = 1
        return t

    def predict(self, user, items):
        # TODO: Do
        pass
