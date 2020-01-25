import json
import os
import random

from loguru import logger


def get_label_map(entities):
    m = {}
    for uri, name, labels in entities:
        if uri not in m:
            m[uri] = []

        for label in labels.split('|'):
            if label not in m[uri]:
                m[uri].append(label)

    return m


def remove_unrated_entities(ratings, min_num_ratings=1):
    entity_rating_counts = {}
    for u, e, r in ratings:
        if e not in entity_rating_counts:
            entity_rating_counts[e] = 0
        entity_rating_counts[e] += 1

    ratings = [(u, e, r) for u, e, r in ratings if entity_rating_counts[e] >= min_num_ratings]
    return ratings


def remove_k_percent_most_popular_movies(ratings, label_map, k_percent):
    entity_rating_counts = {}
    for u, e, r in ratings:
        if 'Movie' in label_map[e]:
            if e not in entity_rating_counts:
                entity_rating_counts[e] = 0
            entity_rating_counts[e] += 1

    sorted_entity_counts = sorted(entity_rating_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_entities = [e for e, count in sorted_entity_counts]

    k = int(len(sorted_entities) * k_percent)
    entities_to_discard = sorted_entities[:k]  # k most popular entities
    return [(u, e, r) for u, e, r in ratings if e not in entities_to_discard]


class User:
    def __init__(self, idx):
        self.idx = idx
        self.movie_ratings = []
        self.descriptive_entity_ratings = []


class Rating:
    def __init__(self, u_idx, e_idx, rating, is_movie_rating):
        self.u_idx = u_idx  # User
        self.e_idx = e_idx  # Entity
        self.rating = rating  # Rating
        self.is_movie_rating = is_movie_rating  # Is this a movie or DE rating?


class DataLoader:
    def __init__(self, ratings, n_users, movie_indices, descriptive_entity_indices, e_idx_map, backwards_u_map,
                 backwards_e_map):
        logger.info(f'Initialized data loader with {len(ratings)} ratings')
        self.ratings = ratings
        self.n_users = n_users
        self.n_movies = len(movie_indices)
        self.n_descriptive_entities = len(descriptive_entity_indices)
        self.descriptive_entity_indices = descriptive_entity_indices
        self.movie_indices = movie_indices
        self.e_idx_map = e_idx_map
        self.random_seed = 51  # The run seed - change this at every run
        self.random = random.Random(self.random_seed)

        # Backwards maps
        self.backwards_u_map = backwards_u_map
        self.backwards_e_map = backwards_e_map

    @staticmethod
    def load_from(path, filter_unknowns=True, min_num_entity_ratings=1, movies_only=False, unify_user_indices=False,
                  remove_top_k_percent=None):
        """
        Load rating triples from the provided path.
        :param path: The path to load ratings from. Must include a ratings_clean.json and entities_clean.json.
        :param filter_unknowns: (Boolean) Should unknown ratings be ignored?
        :param min_num_entity_ratings: (Integer) How many ratings should each entity have, at minimum?
        :param movies_only: (Boolean) Should descriptive entity ratings be ignored?
        :param unify_user_indices: (Boolean) Should users be indexed in the same space as entities?
        :param remove_top_k_percent: (Float) Ignores the top k% popular movies (not entities). Does nothing is None.
        :return: A DataLoader instance.
        """
        return DataLoader(*DataLoader._load_from(path, filter_unknowns, movies_only, unify_user_indices,
                                                 remove_top_k_percent))

    @staticmethod
    def _load_from(path, filter_unknowns=True, min_num_entity_ratings=1, movies_only=False, unify_user_indices=False,
                   remove_top_k_percent=None, random_seed=42):
        with open(os.path.join(path, 'ml_ratings.json')) as ratings_p:
            ratings = json.load(ratings_p)
        with open(os.path.join(path, 'entities_clean.json')) as entities_p:
            entities = json.load(entities_p)

        label_map = get_label_map(entities)

        # Remove unknown ratings?
        if filter_unknowns:
            ratings = [(u, e, r) for u, e, r in ratings if not (r == 0)]

        # Remove entity ratings?
        if movies_only:
            ratings = [(u, e, r) for u, e, r in ratings if 'Movie' in label_map[e]]

        # Remove entities with < 5 or so ratings (so we can put at least one in each bucket for 5-fold)
        ratings = remove_unrated_entities(ratings, min_num_ratings=min_num_entity_ratings)

        # Remove most popular movies?
        if remove_top_k_percent is not None:
            assert isinstance(remove_top_k_percent, float)
            ratings = remove_k_percent_most_popular_movies(ratings, label_map, remove_top_k_percent)

        # Create index mappings
        u_idx_map, uc = {}, 0
        e_idx_map, ec = {}, 0
        movie_indices, descriptive_entity_indices = [], []

        # Create backwards mappings (to get URIs from indices)
        backwards_u_map = {}
        backwards_e_map = {}

        for user, entity, rating in ratings:
            if user not in u_idx_map:
                u_idx_map[user] = uc
                if uc not in backwards_u_map:
                    backwards_u_map[uc] = user
                uc += 1
            if entity not in e_idx_map:
                e_idx_map[entity] = ec
                if 'Movie' in label_map[entity]:
                    movie_indices.append(ec)
                else:
                    descriptive_entity_indices.append(ec)
                if ec not in backwards_e_map:
                    backwards_e_map[ec] = entity

                ec += 1

        ratings = ([Rating(u_idx_map[u] + ec, e_idx_map[e], r, e_idx_map[e] in movie_indices) for u, e, r in ratings]
                   if unify_user_indices else
                   [Rating(u_idx_map[u], e_idx_map[e], r, e_idx_map[e] in movie_indices) for u, e, r in ratings])

        random.Random(random_seed).shuffle(ratings)

        return ratings, uc, movie_indices, descriptive_entity_indices, e_idx_map, backwards_u_map, backwards_e_map

    def info(self):
        return f''' 
            DataLoader Information
            -----------------------------------
            n_users:                      {self.n_users}
            n_movies:                     {self.n_movies}
            n_descriptive_entities:       {self.n_descriptive_entities}

            n_ratings:                    {len(self.ratings)}
            n_movie_ratings:              {len([rating for rating in self.ratings if rating.is_movie_rating])}
            n_descriptive_entity_ratings: {len([rating for rating in self.ratings if not rating.is_movie_rating])}
        '''


if __name__ == '__main__':
    data_loader = DataLoader.load_from('./mindreader', 
        filter_unknowns=True,
        min_num_entity_ratings=1,
        unify_user_indices=False
    )
    with open('meta.json', 'w') as fp: 
        json.dump({
            'e_idx_map': data_loader.e_idx_map,
            'n_users': data_loader.n_users
        }, fp)
    