from models.pagerank.pagerank_recommender import PageRankRecommender, construct_knowledge_graph, \
    construct_collaborative_graph


class JointPageRankRecommender(PageRankRecommender):
    def __init__(self, split, only_positive=False):
        super().__init__(only_positive)
        self.triples_path = split.experiment.dataset.triples_path
        self.entity_idx = split.experiment.dataset.e_idx_map

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        self._fit(training)

    def construct_graph(self, training):
        base_graph = construct_knowledge_graph(self.triples_path, self.entity_idx)

        return construct_collaborative_graph(base_graph, training, self.only_positive)
