from models.base_recommender import RecommenderBase
from models.pagerank.pagerank_recommender import construct_knowledge_graph


class ContentRecommender(RecommenderBase):
    def __init__(self, split):
        super().__init__()
        self.entity_idx = split.experiment.dataset.e_idx_map
        self.optimal_params = None
        self.training = dict()
        self.graph = construct_knowledge_graph(split.experiment.dataset.triples_path, self.entity_idx)

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        for user, ratings in training:
            self.training[user] = ratings

    def predict(self, user, items):
        # Get user ratings
        user_ratings = self.training[user]
