from models.base_recommender import RecommenderBase
from models.mf import MF
import numpy as np
import random
import torch as tt
from loguru import logger


class MatrixFactorisationRecommender(RecommenderBase):
    def __init__(self, split):
        super(MatrixFactorisationRecommender, self).__init__()
        self.split = split

    def convert_rating(self, rating):
        if rating == 1:
            return 1
        elif rating == -1:
            return 0
        elif rating == 0:
            # We can make a choice here - either return 0 or 0.5
            return 0.5

    def batches(self, triples, n=64):
        for i in range(0, len(triples), n):
            yield triples[i:i + n]

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        hit_rates = {}

        for n_latent_factors in [2, 5, 10, 15, 25, 50, 100]:
            logger.info(f'Fitting MF with {n_latent_factors} latent factors.')
            self.model = MF(self.split.n_users, self.split.n_movies + self.split.n_descriptive_entities,
                                 n_latent_factors)
            hit_rates[n_latent_factors] = self._fit(training, validation, max_iterations)

        hit_rates = sorted(hit_rates.items(), key=lambda x: x[1], reverse=True)
        best_n_latent_factors = [n for n, hit in hit_rates][0]

        logger.info(f'Found best n_latent_factors at {best_n_latent_factors}.')

        self.model = MF(self.split.n_users, self.split.n_movies + self.split.n_descriptive_entities,
                             best_n_latent_factors)
        logger.info(f'Fitting MF with {best_n_latent_factors} latent factors.')
        self._fit(training, validation)

    def _fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        optimizer = tt.optim.Adam(self.model.parameters(), lr=0.003)

        # Preprocess training data
        training_triples = []
        for u, ratings in training:
            for r in ratings:
                rating = self.convert_rating(r.rating)
                training_triples.append((u, r.e_idx, rating))

        validation_history = []

        for epoch in range(max_iterations):
            batch_loss = tt.tensor(0.0)
            random.shuffle(training_triples)
            for batch_triples in self.batches(training_triples):
                self.model.train()
                users, items, ratings = zip(*batch_triples)
                loss = self.model(users, items, ratings)

                batch_loss += loss

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
                    logger.info(f'Hit@10 at epoch {epoch}: {np.mean([1 if r < 10 else 0 for r in ranks])}')

        return np.mean(validation_history[-10:])

    def predict(self, user, items):
            predictions = self.model.predict(user, items)
            return {i: s for i, s in zip(items, predictions)}
