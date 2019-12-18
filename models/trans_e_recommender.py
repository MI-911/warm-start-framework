from models.base_recommender import RecommenderBase
from models.other_trans_e import TransE


class TransERecommender(RecommenderBase):
    def __init__(self, n_entities, n_relations, margin, n_latent_factors):
        super(TransERecommender, self).__init__(TransE(n_entities, n_relations, margin, n_latent_factors))

    def fit(self, training, validation, max_iterations=100):
        # Do the training
        pass

    def predict(self, user, items):
        # Do the prediction
        return self.model.predict_movies_for_user(user, relation_idx=1, movie_indices=items)
