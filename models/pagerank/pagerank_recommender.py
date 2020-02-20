import operator
from csv import DictReader
from functools import reduce

from loguru import logger
from networkx import pagerank_scipy, Graph

from models.base_recommender import RecommenderBase


RATING_CATEGORIES = {1, 0, -1}


def construct_collaborative_graph(graph, training, only_positive=False):
    for user, ratings in training:
        user_id = f'user_{user}'
        graph.add_node(user_id, entity=False)

        for rating in ratings:
            if only_positive and rating.rating != 1:
                continue

            graph.add_node(rating.e_idx, entity=True)
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

            graph.add_node(head, entity=True)
            graph.add_node(tail, entity=True)
            graph.add_edge(head, tail, type=relation)

    return graph


class PageRankRecommender(RecommenderBase):
    def __init__(self, only_positive=False):
        super().__init__()
        self.graph = None
        self.only_positive = only_positive
        self.user_ratings = dict()
        self.optimal_params = None
        self.rating_importance = {1: 0.9, 0: 0.1, -1: 0}
        self.entity_indices = set()

    def get_entity_indices(self):
        if self.entity_indices:
            return self.entity_indices

        indices = set()

        for idx, data in self.graph.nodes(data=True):
            if data['entity']:
                indices.add(idx)

        self.entity_indices = indices

        return indices

    def predict(self, user, items):
        return self._scores(self.optimal_params['alpha'], self.get_node_weights(user), items)

    def construct_graph(self, training):
        raise NotImplementedError

    def _scores(self, alpha, node_weights, items):
        scores = pagerank_scipy(self.graph, alpha=alpha, personalization=node_weights).items()

        return {item: score for item, score in scores if item in items}

    def _weight(self, category, ratings):
        if not ratings[category] or not self.rating_importance[category]:
            return 0

        return self.rating_importance[category] / len(ratings[category])

    def get_node_weights(self, user):
        if not self.user_ratings[user]:
            return []

        ratings = {category: set() for category in RATING_CATEGORIES}

        for rating in self.user_ratings[user]:
            if self.only_positive and rating.rating != 1:
                continue

            ratings[rating.rating].add(rating.e_idx)

        # Find rated and unrated entities
        rated_entities = reduce(lambda a, b: a.union(b), ratings.values())
        unrated_entities = self.get_entity_indices().difference(rated_entities)

        # Treat unrated entities as unknown ratings
        ratings[0] = ratings[0].union(unrated_entities)

        # Compute the weight of each rating category
        rating_weight = {category: self._weight(category, ratings) for category in RATING_CATEGORIES}

        # Assign weight to each node depending on their rating
        return {
            idx: rating_weight[category] for category in RATING_CATEGORIES for idx in ratings[category]
        }

    def _validate(self, alpha, source_nodes, validation_item, negatives, k=10):
        scores = self._scores(alpha, source_nodes, [validation_item] + negatives)
        scores = sorted(list(scores.items()), key=operator.itemgetter(1), reverse=True)[:k]

        return validation_item in [item[0] for item in scores]

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        for user, ratings in training:
            self.user_ratings[user] = ratings

        self.graph = self.construct_graph(training)

        if not self.optimal_params:
            alpha_ranges = [0.85]
            alpha_hit = dict()

            for alpha in alpha_ranges:
                logger.debug(f'Trying alpha value {alpha}')

                hits = 0
                count = 0

                for user, validation_tuple in validation:
                    node_weights = self.get_node_weights(user)
                    if not node_weights:
                        continue

                    hits += self._validate(alpha, node_weights, *validation_tuple)
                    count += 1

                hit_ratio = hits / count
                alpha_hit[alpha] = hit_ratio

                logger.debug(f'Hit ratio of {alpha}: {hit_ratio}')

            best = max(alpha_hit.items(), key=operator.itemgetter(1))
            logger.info(f'Best: {best}')

            self.optimal_params = {'alpha': best[0]}
