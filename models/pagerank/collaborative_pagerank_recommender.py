from models.pagerank.pagerank_recommender import PageRankRecommender, construct_collaborative_graph
from networkx import Graph


class CollaborativePageRankRecommender(PageRankRecommender):
    def __init__(self, only_positive=False):
        super().__init__(only_positive)

    def construct_graph(self, training):
        return construct_collaborative_graph(Graph(), training, self.only_positive)
