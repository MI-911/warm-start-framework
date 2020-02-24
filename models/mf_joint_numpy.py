import numpy as np
from numpy.linalg import solve


class JointMatrixFactorization:
    def __init__(self, n_users, n_items, n_latent_factors, relative_rating_influence):
        self.k = n_latent_factors
        self.n_users = n_users
        self.n_items = n_items

        self.regularisation = 0.0015
        self.relative_rating_influence = relative_rating_influence
        self.relative_sppmi_influence = 1 - self.relative_rating_influence

        self.U = np.random.rand(self.n_users, self.k)
        self.M = np.random.rand(self.n_items, self.k)
        self.C = np.random.rand(self.n_items, self.k)

    def train_als(self, triples, sppmi):
        R = np.zeros((self.n_users, self.n_items))
        for u, e, r in triples:
            R[u][e] = r

        self.U = self.solve_vectors(self.U, self.M, R, self.regularisation, self.relative_rating_influence)
        self.C = self.solve_vectors(self.C, self.M, sppmi, self.regularisation, self.relative_sppmi_influence)
        self.M = self.solve_joint_vectors(self.M, self.U, self.C, R.T, sppmi, self.regularisation)

    def solve_vectors(self, latent_vectors, fixed_vectors, ratings, regularization, influence):
        YTY = fixed_vectors.T.dot(fixed_vectors) * influence
        lambda_i = np.eye(self.k) * regularization

        for i in range(latent_vectors.shape[0]):
            latent_vectors[i] = solve((YTY + lambda_i),
                                      (ratings[i] * influence).dot(fixed_vectors))

        return latent_vectors

    def solve_joint_vectors(self, latent_vectors, user_vectors, context_vectors, ratings, sppmi, regularization):
        UTU = user_vectors.T.dot(user_vectors) * self.relative_rating_influence
        CTC = context_vectors.T.dot(context_vectors) * self.relative_sppmi_influence
        lambda_i = np.eye(self.k) * regularization

        for i in range(latent_vectors.shape[0]):
            # nz_s = sppmi[i].nonzero()
            latent_vectors[i] = solve(
                (UTU + CTC + lambda_i),
                ((ratings[i] * self.relative_rating_influence).dot(user_vectors) +
                 # (sppmi[i][nz_s] * self.relative_sppmi_influence).dot(context_vectors[nz_s]))
                 (sppmi[i] * self.relative_sppmi_influence).dot(context_vectors))
            )

        return latent_vectors

    def predict(self, user, movies):
        predictions = (self.U[user] * self.M[movies]).sum(axis=1)
        return {m: s for m, s in zip(movies, predictions)}

