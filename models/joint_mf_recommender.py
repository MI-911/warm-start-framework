from models.base_recommender import RecommenderBase
from models.joint_mf import JointMF
import numpy as np
import random
import torch as tt
from loguru import logger


class JointMatrixFactorisationRecommender(RecommenderBase):
    def __init__(self, split):
        super(JointMatrixFactorisationRecommender, self).__init__()
        self.split = split

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
        hit_rates = {}

        for n_latent_factors in [1, 5, 10, 15, 25, 50, 100]:
            logger.debug(f'Fitting JointMF with {n_latent_factors} latent factors')
            self.model = JointMF(self.split.n_users, self.split.n_movies + self.split.n_descriptive_entities, n_latent_factors)
            hit_rates[n_latent_factors] = self._fit(training, validation, max_iterations)

        hit_rates = sorted(hit_rates.items(), key=lambda x: x[1], reverse=True)
        best_n_latent_factors = [n for n, hit in hit_rates][0]

        self.model = JointMF(self.split.n_users, self.split.n_movies + self.split.n_descriptive_entities, best_n_latent_factors)
        logger.info(f'Fitting JointMF with {best_n_latent_factors} latent factors')
        self._fit(training, validation)

    def _fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):

        optimizer = tt.optim.Adam(self.model.parameters(), lr=0.003)

        # Preprocess training data
        training_triples = []

        for u, ratings in training:
            for r in ratings:
                rating = self.convert_rating(r.rating)
                training_triples.append((u, r.e_idx, rating))

        sppmi_triples = self.get_sppmi(training_triples, self.split.n_movies + self.split.n_descriptive_entities)

        # Set a boolean so the model knows if its a rating or an SPPMI entry
        training_triples = [(u, e, r, True) for u, e, r in training_triples]
        sppmi_triples = [(i ,j, s, False) for i, j, s in sppmi_triples]

        validation_history = []

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

                    _hit = np.mean([1 if r < 10 else 0 for r in ranks])
                    validation_history.append(_hit)

                if verbose:
                    logger.debug(f'Hit@10 at epoch {epoch}: {_hit}')

        return np.mean(validation_history[-10:])

    def predict(self, user, items):
        predictions = self.model.predict(user, items)
        return {i: s for i, s in zip(items, predictions)}
