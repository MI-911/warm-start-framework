from models.base_recommender import RecommenderBase
from models.trans_h import TransH
from data_loading.generic_data_loader import Rating
import numpy as np
import torch as tt
import pandas as pd
import random
from loguru import logger
import json
import pickle


def unify_user_indices(ratings, u_idx_start):
    unified = []
    for user_index, rest in ratings:
        unified.append((user_index + u_idx_start, rest))
        u_idx_start += 1

    return unified


def convert_ratings(ratings):
    converted = []
    for user_index, rs in ratings:
        converted.append((user_index, [
            Rating(r.u_idx, r.e_idx, 1 if r.rating == 1 else 0 if r.rating == -1 else 2, r.is_movie_rating)
            for r in rs
        ]))

    return converted


def flatten_ratings(user_ratings, movies_only=False):
    rating_triples = []
    for u, rs in user_ratings:
        rating_triples += (
            [(r.u_idx, 1 if r.rating == 1 else 0, r.e_idx) for r in rs if r.is_movie_rating] if movies_only else
            [(r.u_idx, 1 if r.rating == 1 else 0, r.e_idx) for r in rs]
        )

    return rating_triples


def corrupt_std(flat_ratings, all_entities):
    corrupted = []
    for h, r, t in flat_ratings:
        if random.random() > 0.5:
            corrupted.append((random.choice(all_entities), r, t))
        else:
            corrupted.append((h, r, random.choice(all_entities)))

    return corrupted


def batchify(pos, neg, batch_size=64):
    for i in range(0, len(pos), batch_size):
        yield zip(*pos[i:i + batch_size]), zip(*neg[i:i + batch_size])


def evaluate_loss(model, user_ratings):
    with tt.no_grad():
        model.eval()

        all_ratings = flatten_ratings(user_ratings, movies_only=True)
        heads, relations, tails = zip(*all_ratings)

        loss = model(heads, relations, tails)

        return float(loss.mean().cpu().numpy().sum())


def dcg(rank, n=10):
    r = np.zeros(n)
    if rank < n:
        r[rank] = 1

    return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))


def evaluate_hit(model, user_samples, n=10):
    with tt.no_grad():
        model.eval()
        # Each entry is (user, (pos_sample, neg_samples))
        ranks = []
        dcgs = []

        for user, (pos_sample, neg_samples) in user_samples:
            fast_rank = model.fast_rank(user, 1, pos_sample, neg_samples)
            fast_dcg = dcg(fast_rank, n=n)
            ranks.append(fast_rank)
            dcgs.append(fast_dcg)

        _dcg = np.mean(dcgs)
        _hit = len(np.where(np.array(ranks) < n)[0]) / len(user_samples)

        return float(_hit), float(_dcg)


def load_kg_triples(split):
    e_idx_map = split.experiment.dataset.e_idx_map

    with open(split.experiment.dataset.triples_path) as fp:
        df = pd.read_csv(fp)
        triples = [(h, r, t) for h, r, t in df[['head_uri', 'relation', 'tail_uri']].values]
        triples = [(e_idx_map[h], r, e_idx_map[t]) for h, r, t in triples if h in e_idx_map and t in e_idx_map]

    indexed_triples = []
    r_idx_map = {}
    rc = 3
    for h, r, t in triples:
        if r not in r_idx_map:
            r_idx_map[r] = rc
            rc += 1

        # Reverse
        not_r = f'not-{r}'
        if not_r not in r_idx_map: 
            r_idx_map[not_r] = rc
            rc += 1
        
        indexed_triples.append((h, r_idx_map[r], t))
        indexed_triples.append((t, r_idx_map[not_r], h))

    return indexed_triples, r_idx_map


def get_like_matrix(ratings, n_users, n_entities):
    u_idx_to_matrix_map, uc = {}, 0
    e_idx_to_matrix_map, ec = {}, 0

    R = np.zeros((n_users, n_entities))
    for u, r, e in ratings:
        if u not in u_idx_to_matrix_map:
            u_idx_to_matrix_map[u] = uc
            uc += 1
        if e not in e_idx_to_matrix_map:
            e_idx_to_matrix_map[e] = ec
            ec += 1

        R[u_idx_to_matrix_map[u]][e_idx_to_matrix_map[e]] = 1 if r == 1 else -1

    return R, u_idx_to_matrix_map, e_idx_to_matrix_map


def corrupt_rating_triples(triples, ratings_matrix, u_idx_to_matrix_map, e_idx_to_matrix_map):
    corrupted = []
    for h, r, t in triples:
        h_mat = u_idx_to_matrix_map[h]
        t_mat = e_idx_to_matrix_map[t]
        if random.random() > 0.5:
            if r == 1:
                # Find a user that dislikes t
                users_disliking_t = np.argwhere(ratings_matrix[:, t_mat] == -1).flatten()
                if len(users_disliking_t) == 0:
                    users_disliking_t = list(e_idx_to_matrix_map.keys())
                corrupted.append((random.choice(users_disliking_t), r, t))
            else:
                # Find a user that likes t
                users_liking_t = np.argwhere(ratings_matrix[:, t_mat] == 1).flatten()
                if len(users_liking_t) == 0:
                    users_liking_t = list(e_idx_to_matrix_map.keys())
                corrupted.append((random.choice(users_liking_t), r, t))
        else:
            if r == 1:
                # Find an item that h dislikes
                items_disliked_by_h = np.argwhere(ratings_matrix[h_mat] == -1).flatten()
                if len(items_disliked_by_h) == 0:
                    items_disliked_by_h = list(e_idx_to_matrix_map.keys())
                corrupted.append((h, r, random.choice(items_disliked_by_h)))
            else:
                # Find an item that h likes
                items_liked_by_h = np.argwhere(ratings_matrix[h_mat] == 1).flatten()
                if len(items_liked_by_h) == 0:
                    items_liked_by_h = list(e_idx_to_matrix_map.keys())
                corrupted.append((h, r, random.choice(items_liked_by_h)))

    return corrupted


class CollabTransHRecommender(RecommenderBase):
    def __init__(self, split):
        super(CollabTransHRecommender, self).__init__()

        self.n_entities = split.n_users + split.n_movies + split.n_descriptive_entities
        self.n_relations = 3  # Like, dislike, unknown

        self.with_kg_triples = False
        self.with_standard_corruption = True
        self.split = split

        self.learning_rate = 0.003
        self.margin = 1.0

        self.optimal_params = None

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        self.best_hit = 0
        self.best_k = None
        self.best_model = None
        
        if self.optimal_params is None:
            for n_latent_factors in [50, 100, 250]:
                logger.debug(f'Fitting TransH with {n_latent_factors} latent factors')
                self.model = TransH(self.n_entities, self.n_relations, self.margin, n_latent_factors)
                self._fit(training, validation)

            self.optimal_params = {'k': self.best_k}

        self.model = TransH(self.n_entities, self.n_relations, self.margin, self.optimal_params['k'])

        logger.info(f'Fitting TransH with {self.optimal_params["k"]} latent factors')
        self._fit(training, validation, max_iterations, verbose)

    def _fit(self, training, validation, max_iterations=100, verbose=True):
        val_hit_history = []
        val_dcg_history = []

        # Convert likes/dislikes to relation 1 and 0
        train = convert_ratings(training)

        # Load KG triples if needed
        kg_triples, r_idx_map = load_kg_triples(self.split) if self.with_kg_triples else ([], {})
        self.n_relations += len(r_idx_map)

        optimizer = tt.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Rating triples
        all_train_ratings = flatten_ratings(train)
        ratings_matrix, u_idx_to_matrix_map, e_idx_to_matrix_map = get_like_matrix(
            all_train_ratings,
            self.split.n_users,
            self.split.n_entities)

        for epoch in range(max_iterations):
            if epoch % 1 == 0:
                _hit, _dcg = evaluate_hit(self.model, validation, n=10)
                if _hit > self.best_hit: 
                    logger.info(f'Found new best TransH with hit {_hit}')
                    self.best_hit = _hit
                    self.best_k = self.model.k
                    self.best_model = pickle.loads(pickle.dumps(self.model))

                val_hit_history.append(_hit)
                val_dcg_history.append(_dcg)

                if verbose:
                    logger.info(f'Hit@10 at epoch {epoch}: {_hit}')

            corrupted_train_ratings = (
                corrupt_std(all_train_ratings, range(self.n_entities))
                if self.with_standard_corruption else
                corrupt_rating_triples(all_train_ratings, ratings_matrix, u_idx_to_matrix_map, e_idx_to_matrix_map))
            corrupted_train_kg_triples = corrupt_std(kg_triples, range(self.n_entities))

            all_pairs = list(zip(all_train_ratings, corrupted_train_ratings))
            all_pairs += list(zip(kg_triples, corrupted_train_kg_triples))[:len(all_pairs)]

            random.shuffle(all_pairs)
            positive_samples, negative_samples = zip(*all_pairs)

            self.model.train()
            for (p_h, p_r, p_t), (n_h, n_r, n_t) in batchify(positive_samples, negative_samples):
                p_distance = self.model(p_h, p_r, p_t)
                n_distance = self.model(n_h, n_r, n_t)

                loss = tt.relu(self.model.margin + p_distance - n_distance).sum()

                p_h, p_r, p_t = self.model.params_to(p_h, p_r, p_t)
                n_h, n_r, n_t = self.model.params_to(n_h, n_r, n_t)

                e_embeddings = self.model.entity_embeddings(tt.cat([p_h, p_t, n_h, n_t]))
                r_embeddings = self.model.relation_embeddings(tt.cat([p_r, n_r]))
                norm_embeddings = self.model.norm_embeddings(tt.cat([p_r, n_r]))

                # Calculate the orthogonal loss
                loss += tt.sum(tt.sum(norm_embeddings * r_embeddings, dim=1, keepdim=True) ** 2 / tt.sum(r_embeddings ** 2, dim=1, keepdim=True))

                # Calculate the projection loss
                e_norm = tt.sum(e_embeddings ** 2, dim=1, keepdim=True)
                r_norm = tt.sum(r_embeddings ** 2, dim=1, keepdim=True)

                loss += tt.sum(tt.max(e_norm - tt.tensor(1.0).to(self.model.device), tt.tensor(0.0).to(self.model.device)))
                loss += tt.sum(tt.max(r_norm - tt.tensor(1.0).to(self.model.device), tt.tensor(0.0).to(self.model.device)))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Normalise hyperplanes
                self.model.normalize_hyperplanes(p_r)

        # Return the latest average Hit@k, use for grid search
        self.model = self.best_model
        return np.mean(val_hit_history[-10:])

    def predict(self, user, items):
        # Do the prediction
        with tt.no_grad():
            self.model.eval()
            return self.model.predict_movies_for_user(user, relation_idx=1, movie_indices=items)


class KGTransHRecommender(RecommenderBase):
    def __init__(self, split):
        super(KGTransHRecommender, self).__init__()

        self.n_entities = split.n_users + split.n_movies + split.n_descriptive_entities
        self.n_relations = 3 + 7 + 7  # Like, dislike, unknown + reverse

        self.with_kg_triples = True
        self.with_standard_corruption = True
        self.split = split

        self.learning_rate = 0.003
        self.margin = 1.0
        self.optimal_params = None

    def fit(self, training, validation, max_iterations=5, verbose=True, save_to='./'):
        self.best_hit = 0
        self.best_model = None

        if self.optimal_params is None:
            for n_latent_factors in [50]:
                logger.debug(f'Fitting TransH-KG with {n_latent_factors} latent factors')
                self.model = TransH(self.n_entities, self.n_relations, self.margin, n_latent_factors)
                self._fit(training, validation)

            self.optimal_params = {'k': self.best_k}

        self.model = TransH(self.n_entities, self.n_relations, self.margin, self.optimal_params['k'])

        logger.info(f'Fitting TransH-KG with { self.optimal_params["k"]} latent factors')
        self._fit(training, validation, max_iterations, verbose)

    def _fit(self, training, validation, max_iterations=5, verbose=True):
        val_hit_history = []
        val_dcg_history = []

        # Convert likes/dislikes to relation 1 and 0
        train = convert_ratings(training)

        # Load KG triples if needed
        kg_triples, r_idx_map = load_kg_triples(self.split) if self.with_kg_triples else ([], {})
        self.n_relations += len(r_idx_map)

        optimizer = tt.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Rating triples
        all_train_ratings = flatten_ratings(train)
        ratings_matrix, u_idx_to_matrix_map, e_idx_to_matrix_map = get_like_matrix(
            all_train_ratings,
            self.split.n_users,
            self.split.n_entities)

        for epoch in range(max_iterations):
            if epoch % 1 == 0:
                _hit, _dcg = evaluate_hit(self.model, validation, n=10)

                if _hit > self.best_hit: 
                    logger.info(f'Found new best TransH-KG with hit {_hit}')
                    self.best_hit = _hit
                    self.best_k = self.model.k
                    self.best_model = pickle.loads(pickle.dumps(self.model))
                val_hit_history.append(_hit)
                val_dcg_history.append(_dcg)

                if verbose:
                    logger.info(f'Hit@10 at epoch {epoch}: {_hit}')

            corrupted_train_ratings = (
                corrupt_std(all_train_ratings, range(self.n_entities))
                if self.with_standard_corruption else
                corrupt_rating_triples(all_train_ratings, ratings_matrix, u_idx_to_matrix_map, e_idx_to_matrix_map))
            corrupted_train_kg_triples = corrupt_std(kg_triples, range(self.n_entities))

            all_pairs = list(zip(all_train_ratings, corrupted_train_ratings))
            all_pairs += list(zip(kg_triples, corrupted_train_kg_triples))[:len(all_pairs)]

            random.shuffle(all_pairs)
            positive_samples, negative_samples = zip(*all_pairs)

            self.model.train()
            for (p_h, p_r, p_t), (n_h, n_r, n_t) in batchify(positive_samples, negative_samples):
                p_distance = self.model(p_h, p_r, p_t)
                n_distance = self.model(n_h, n_r, n_t)

                loss = tt.relu(self.model.margin + p_distance - n_distance).sum()

                p_h, p_r, p_t = self.model.params_to(p_h, p_r, p_t)
                n_h, n_r, n_t = self.model.params_to(n_h, n_r, n_t)

                e_embeddings = self.model.entity_embeddings(tt.cat([p_h, p_t, n_h, n_t]))
                r_embeddings = self.model.relation_embeddings(tt.cat([p_r, n_r]))
                norm_embeddings = self.model.norm_embeddings(tt.cat([p_r, n_r]))

                # Calculate the orthogonal loss
                loss += tt.sum(tt.sum(norm_embeddings * r_embeddings, dim=1, keepdim=True) ** 2 / tt.sum(r_embeddings ** 2, dim=1, keepdim=True))

                # Calculate the projection loss
                e_norm = tt.sum(e_embeddings ** 2, dim=1, keepdim=True)
                r_norm = tt.sum(r_embeddings ** 2, dim=1, keepdim=True)

                loss += tt.sum(tt.max(e_norm - tt.tensor(1.0).to(self.model.device), tt.tensor(0.0).to(self.model.device)))
                loss += tt.sum(tt.max(r_norm - tt.tensor(1.0).to(self.model.device), tt.tensor(0.0).to(self.model.device)))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Normalise hyperplanes
                self.model.normalize_hyperplanes(p_r)

        # Return the latest average Hit@k, use for grid search
        self.model = self.best_model
        return np.mean(val_hit_history[-10:])

    def predict(self, user, items):
        # Do the prediction
        with tt.no_grad():
            self.model.eval()
            return self.model.predict_movies_for_user(user, relation_idx=1, movie_indices=items)

