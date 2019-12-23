from models.base_recommender import RecommenderBase
from models.mf import MF
import numpy as np
import random
import torch as tt
from loguru import logger


class JointMatrixFactorisationRecommender(RecommenderBase):
    def __init__(self, data_loader):
        super(JointMatrixFactorisationRecommender, self).__init__()
        self.data_loader = data_loader

    def convert_rating(self, rating):
        if rating == 1:
            return 1
        elif rating == -1:
            return 0
        elif rating == 0:
            # We can make a choice here - either return 0 or 0.5
            return 0.5

    def get_sppmi(self, rating_triples, n_items):
        co_occurrence_matrix = np.zeros((n_items, n_items))

        u_r_map = {}
        for u, e, r in rating_triples:
            if u not in u_r_map:
                u_r_map[u] = []
            if r == 1:
                u_r_map[u].append(e)

        for u, ratings in u_r_map.items():
            for i, r in enumerate(ratings):
                for j, o in enumerate(ratings[i + 1:]):
                    co_occurrence_matrix[r][o] += 1

        M = np.zeros((n_items, n_items))
        D = co_occurrence_matrix.sum()
        for i in range(n_items):
            for j in range(n_items):
                if i == j:
                    continue
                M[i][j] = (
                    (co_occurrence_matrix[i][j] * D) /
                    (co_occurrence_matrix[i].sum() * co_occurrence_matrix[:, j].sum())
                )

        k = 5
        return np.max(0, M - np.log(k))

    def batches(self, triples, n=64):
        for i in range(0, len(triples), n):
            yield triples[i:i + n]

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        n_users = self.data_loader.n_users
        n_movies = self.data_loader.n_movies
        n_descriptive_entities = self.data_loader.n_descriptive_entities
        n_latent_factors = 25

        self.model = MF(n_users, n_movies + n_descriptive_entities, latent_factors=n_latent_factors)

        optimizer = tt.optim.Adam(self.model.parameters(), lr=0.003)

        # Preprocess training data
        training_triples = []

        for u, ratings in training:
            for r in ratings:
                rating = self.convert_rating(r.rating)
                training_triples.append((u, r.e_idx, rating))

        sppmi_triples = self.get_sppmi(training_triples, self.data_loader.n_movies + self.data_loader.n_descriptive_entities)

        # Set a boolean so the model knows if its a rating or an SPPMI entry
        training_triples = [(u, e, r, True) for u, e, r in training_triples]
        sppmi_triples = [(i ,j, s, False) for i, j, s in sppmi_triples]

        for epoch in range(max_iterations):
            training_triples += sppmi_triples
            random.shuffle(training_triples)
            for batch_triples in self.batches(training_triples):
                self.model.train()
                users, items, ratings, is_ratings = zip(*batch_triples)
                loss = self.model(users, items, ratings, is_ratings)

                loss.backward()
                optimizer.step()

                self.model.zero_grad()

            if epoch % 5 == 0:
                with tt.no_grad():
                    self.model.eval()

                    ranks = []
                    for user, (pos, negs) in validation:
                        predictions = self.model.predict(user, negs + [pos])
                        pred_map = {i: s for i, s in zip(negs + [pos], predictions)}
                        pred_map = sorted(pred_map.items(), key=lambda x: x[1], reverse=True)
                        pred_map = {i: rank for rank, (i, s) in enumerate(pred_map)}
                        ranks.append(pred_map[pos])

                if verbose:
                    logger.info(f'Hit@10 at epoch {epoch}: {np.mean([1 if r < 10 else 0 for r in ranks])}')

    def predict(self, user, items):
        predictions = self.model.predict(user, items)
        return {i: s for i, s in zip(items, predictions)}
