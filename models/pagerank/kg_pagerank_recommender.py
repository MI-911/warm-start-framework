from models.pagerank.pagerank_recommender import *


class KnowledgeGraphPageRankRecommender(PageRankRecommender):
    def __init__(self, split):
        super().__init__()
        self.triples_path = split.experiment.dataset.triples_path
        self.entity_idx = split.experiment.dataset.e_idx_map

    def construct_graph(self, training=None):
        return construct_knowledge_graph(self.triples_path, self.entity_idx)
