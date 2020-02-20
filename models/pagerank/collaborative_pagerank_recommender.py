from models.pagerank.pagerank_recommender import PageRankRecommender, construct_collaborative_graph
from networkx import Graph


class CollaborativePageRankRecommender(PageRankRecommender):
    def __init__(self):
        super().__init__()

    def construct_graph(self, training):
        return construct_collaborative_graph(Graph(), training)
