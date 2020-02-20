from models.pagerank.pagerank_recommender import PageRankRecommender, construct_knowledge_graph, \
    construct_collaborative_graph


class JointPageRankRecommender(PageRankRecommender):
    def __init__(self, split):
        super().__init__()
        self.triples_path = split.experiment.dataset.triples_path
        self.entity_idx = split.experiment.dataset.e_idx_map

    def construct_graph(self, training):
        base_graph = construct_knowledge_graph(self.triples_path, self.entity_idx)

        return construct_collaborative_graph(base_graph, training)
