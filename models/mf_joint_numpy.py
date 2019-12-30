import numpy as np
from numpy.linalg import solve
from loguru import logger


class JointMatrixFactorization:
    def __init__(self, n_users, n_items, n_latent_factors):
        self.k = n_latent_factors
        self.n_users = n_users
        self.n_items = n_items

        self.regularization = 0.001
        self.learning_rate = 0.001

        self.U = np.random.rand(self.n_users, self.k)
        self.M = np.random.rand(self.n_items, self.k)
        self.C = np.random.rand(self.n_items, self.k)

    def train_als(self, triples, sppmi):
        R = np.zeros((self.n_users, self.n_items))
        for u, e, r in triples:
            R[u][e] = r

        self.U = self.solve_vectors(self.U, self.M, R)
        self.C = self.solve_vectors(self.C, self.M, sppmi)
        self.M = self.solve_vectors(self.M, self.U, R.T)

    def solve_vectors(self, latent_vectors, fixed_vectors, ratings):
        YTY = fixed_vectors.T.dot(fixed_vectors)
        lambda_i = np.eye(self.k) * self.regularization

        for i in range(latent_vectors.shape[0]):
            latent_vectors[i] = solve((YTY + lambda_i),
                                      ratings[i].dot(fixed_vectors))

        return latent_vectors

    def solve_joint_vectors(self, latent_vectors, user_vectors, context_vectors, ratings, sppmi):
        UTU = user_vectors.T.dot(user_vectors)
        CTC = context_vectors.T.dot(context_vectors)
        lambda_i = np.eye(self.k) * self.regularization

        for i in range(latent_vectors.shape[0]):
            latent_vectors[i] = solve(
                (UTU + CTC + lambda_i),
                (ratings[i].dot(user_vectors) + sppmi[i].dot(context_vectors))
            )

    def predict(self, user, movies):
        predictions = (self.U[user] * self.M[movies]).sum(axis=1)
        return {m: s for m, s in zip(movies, predictions)}

