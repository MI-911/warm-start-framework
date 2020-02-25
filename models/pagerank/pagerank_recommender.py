import operator
from csv import DictReader
from functools import reduce

from loguru import logger
from networkx import pagerank_scipy, Graph

from models.base_recommender import RecommenderBase
from utility.utility import get_combinations

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
    def __init__(self):
        super().__init__()
        self.graph = None
        self.user_ratings = dict()
        self.optimal_params = None
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
        return self._scores(self.optimal_params['alpha'],
                            self.get_node_weights(user, self.optimal_params['importance']), items)

    def construct_graph(self, training):
        raise NotImplementedError

    def _scores(self, alpha, node_weights, items):
        scores = pagerank_scipy(self.graph, alpha=alpha, personalization=node_weights).items()

        return {item: score for item, score in scores if item in items}

    @staticmethod
    def _weight(category, ratings, importance):
        if not ratings[category] or not importance[category]:
            return 0

        return importance[category] / len(ratings[category])

    def get_node_weights(self, user, importance):
        if not self.user_ratings[user]:
            return []

        ratings = {category: set() for category in RATING_CATEGORIES}

        for rating in self.user_ratings[user]:
            ratings[rating.rating].add(rating.e_idx)

        # Find rated and unrated entities
        rated_entities = reduce(lambda a, b: a.union(b), ratings.values())
        unrated_entities = self.get_entity_indices().difference(rated_entities)

        # Treat unrated entities as unknown ratings
        ratings[0] = ratings[0].union(unrated_entities)

        # Compute the weight of each rating category
        rating_weight = {category: self._weight(category, ratings, importance) for category in RATING_CATEGORIES}

        # Assign weight to each node depending on their rating
        return {idx: rating_weight[category] for category in RATING_CATEGORIES for idx in ratings[category]}

    def _validate(self, alpha, source_nodes, validation_item, negatives, k=10):
        scores = self._scores(alpha, source_nodes, [validation_item] + negatives)
        scores = sorted(list(scores.items()), key=operator.itemgetter(1), reverse=True)[:k]

        return validation_item in [item[0] for item in scores]

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        for user, ratings in training:
            self.user_ratings[user] = ratings

        self.graph = self.construct_graph(training)

        if not self.optimal_params:
            parameters = {
                'alpha': [0.85],
                'importance': [
                    {1: 1, 0: 0, -1: 0},
                    {1: 0.5, 0: 0, -1: 0.5},
                    {1: 0.9, 0: 0.1, -1: 0.0},
                    {1: 0.0, 0: 0.1, -1: 0.9},
                    {1: 0.05, 0: 0.9, -1: 0.05},
                ]
            }

            combinations = get_combinations(parameters)
            logger.debug(f'{len(combinations)} hyperparameter combinations')

            results = list()

            for combination in combinations:
                logger.debug(f'Trying {combination}')

                hits = 0
                count = 0

                for user, validation_tuple in validation:
                    node_weights = self.get_node_weights(user, combination['importance'])
                    if not node_weights:
                        continue

                    hits += self._validate(combination['alpha'], node_weights, *validation_tuple)
                    count += 1

                logger.debug(f'Hit: {hits / count * 100:.2f}%')
                results.append((combination, hits / count))

            best = sorted(results, key=operator.itemgetter(1), reverse=True)[0][0]
            logger.info(f'Found best: {best}')

            self.optimal_params = best
