from models.pagerank.pagerank_recommender import *


class KnowledgeGraphPageRankRecommender(PageRankRecommender):
    def __init__(self, data_loader, triples_path='./data_loading/mindreader/triples.csv'):
        super().__init__()
        self.triples_path = triples_path
        self.entity_idx = data_loader.e_idx_map

    def construct_graph(self, training=None):
        return construct_knowledge_graph(self.triples_path, self.entity_idx)
