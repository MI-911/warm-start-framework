import operator
from csv import DictReader

import tqdm
from networkx import pagerank_scipy, Graph

from models.base_recommender import RecommenderBase
import numpy as np
from loguru import logger


def construct_collaborative_graph(graph, training, only_positive=False):
    for user, ratings in training:
        user_id = f'user_{user}'
        graph.add_node(user_id)

        for rating in ratings:
            if only_positive and rating.rating != 1:
                continue

            graph.add_node(rating.e_idx)
            graph.add_edge(user_id, rating.e_idx)

    return graph


def construct_knowledge_graph(triples_path, entity_idx):
    graph = Graph()

    with open(triples_path, 'r') as graph_fp:
        graph_reader = DictReader(graph_fp)

        for row in graph_reader:
            head = row['head_uri']
            tail = row['tail_uri']

            head = entity_idx[head] if head in entity_idx else head
            tail = entity_idx[tail] if tail in entity_idx else tail
            relation = row['relation']

            graph.add_node(head)
            graph.add_node(tail)
            graph.add_edge(head, tail, type=relation)

    return graph


class PageRankRecommender(RecommenderBase):
    def __init__(self, only_positive=False):
        super().__init__()
        self.graph = None
        self.alpha = 0.85
        self.only_positive = only_positive
        self.user_ratings = dict()

    def predict(self, user, items):
        return self._scores(self.alpha, self.get_source_nodes(user), items)

    def construct_graph(self, training):
        raise NotImplementedError

    def _scores(self, alpha, source_nodes, items):
        items = set(items)
        if not source_nodes:
            return {item: 0 for item in items}

        scores = pagerank_scipy(self.graph, alpha=alpha, personalization={entity: 1 for entity in source_nodes}).items()

        return {item: score for item, score in scores if item in items}

    def get_source_nodes(self, user):
        source_nodes = []
        for rating in self.user_ratings[user]:
            if self.only_positive and rating.rating != 1:
                continue

            source_nodes.append(rating.e_idx)

        return source_nodes

    def _validate(self, alpha, source_nodes, validation_item, negatives, k=10):
        scores = self._scores(alpha, source_nodes, [validation_item] + negatives)
        scores = sorted(list(scores.items()), key=operator.itemgetter(1), reverse=True)[:k]

        return validation_item in [item[0] for item in scores]

    def _fit(self, training):
        for user, ratings in training:
            self.user_ratings[user] = ratings

        self.graph = self.construct_graph(training)

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        self._fit(training)

        alpha_ranges = [0.35, 0.6, 0.85]
        alpha_hit = dict()

        for alpha in alpha_ranges:
            logger.debug(f'Trying alpha value {alpha}')

            hits = 0
            count = 0

            for user, validation_tuple in validation:
                source_nodes = self.get_source_nodes(user)
                if not source_nodes:
                    continue

                hits += self._validate(alpha, source_nodes, *validation_tuple)
                count += 1

            hit_ratio = hits / count
            alpha_hit[alpha] = hit_ratio

            logger.debug(f'Hit ratio of {alpha}: {hit_ratio}')

        best = max(alpha_hit.items(), key=operator.itemgetter(1))
        self.alpha = best[0]
        logger.info(f'Best: {best}')

        return alpha_hit
