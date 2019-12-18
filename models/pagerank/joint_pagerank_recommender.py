from models.pagerank.pagerank_recommender import PageRankRecommender, construct_knowledge_graph, \
    construct_collaborative_graph


class JointPageRankRecommender(PageRankRecommender):
    def __init__(self, data_loader, only_positive=False, triples_path='../data_loading/mindreader/triples.csv'):
        super().__init__(only_positive)
        self.triples_path = triples_path
        self.entity_idx = data_loader.e_idx_map

    def construct_graph(self, training):
        base_graph = construct_knowledge_graph(self.triples_path, self.entity_idx)

        return construct_collaborative_graph(base_graph, training, self.only_positive)
