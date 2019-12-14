import numpy as np
import pandas as pd
import os
import json
import random


def get_label_map(entities):
    m = {}
    for uri, name, labels in entities:
        if uri not in m:
            m[uri] = []

        for label in labels.split('|'):
            if label not in m[uri]:
                m[uri].append(label)

    return m


def remove_unrated_entities(ratings, min_num_ratings=5):
    entity_rating_counts = {}
    for u, e, r in ratings:
        if e not in entity_rating_counts:
            entity_rating_counts[e] = 0
        entity_rating_counts[e] += 1

    ratings = [(u, e, r) for u, e, r in ratings if entity_rating_counts[e] >= min_num_ratings]
    return ratings


class User:
    def __init__(self, idx):
        self.idx = idx
        self.movie_ratings = {}
        self.descriptive_entity_ratings = {}


class Rating:
    def __init__(self, u_idx, e_idx, rating, is_movie_rating):
        self.u_idx = u_idx  # User
        self.e_idx = e_idx  # Entity
        self.rating = rating  # Rating
        self.is_movie_rating = is_movie_rating  # Is this a movie or DE rating?


class DataLoader:
    def __init__(self, ratings, user_ratings, movie_indices, descriptive_entity_indices):
        print(f'Init dataloader with {len(ratings)} ratings')
        self.ratings = ratings
        self.user_ratings = user_ratings
        self.n_users = len(user_ratings)
        self.n_movies = len(movie_indices)
        self.n_descriptive_entities = len(descriptive_entity_indices)
        self.descriptive_entity_indices = descriptive_entity_indices
        self.movie_indices = movie_indices

    @staticmethod
    def load_from(path, filter_unknowns=True):
        return DataLoader(*DataLoader._load_from(path, filter_unknowns))

    @staticmethod
    def _load_from(path, filter_unknowns=True, min_num_ratings_per_entity=5):
        with open(os.path.join(path, 'ratings_clean.json')) as ratings_p:
            ratings = json.load(ratings_p)
        with open(os.path.join(path, 'entities_clean.json')) as entities_p:
            entities = json.load(entities_p)

        label_map = get_label_map(entities)

        # Remove unknown ratings?
        if filter_unknowns:
            ratings = [(u, e, r) for u, e, r in ratings if not r == 0]

        # Remove entities with < 5 or so ratings (so we can put at least one in each bucket for 5-fold)
        ratings = remove_unrated_entities(ratings, min_num_ratings=min_num_ratings_per_entity)

        # Create index mappings
        u_idx_map, uc = {}, 0
        e_idx_map, ec = {}, 0
        movie_indices, descriptive_entity_indices = [], []

        for user, entity, rating in ratings:
            if user not in u_idx_map:
                u_idx_map[user] = uc
                uc += 1
            if entity not in e_idx_map:
                e_idx_map[entity] = ec
                if 'Movie' in label_map[entity]:
                    movie_indices.append(ec)
                else:
                    descriptive_entity_indices.append(ec)
                ec += 1

        ratings = [Rating(u_idx_map[u], e_idx_map[e], r, e_idx_map[e] in movie_indices) for u, e, r in ratings]

        random.Random(42).shuffle(ratings)

        # Build user-major ratings
        user_ratings = {}
        for rating in ratings:
            if rating.u_idx not in user_ratings:
                user_ratings[rating.u_idx] = User(rating.u_idx)
            if rating.is_movie_rating:
                user_ratings[rating.u_idx].movie_ratings[rating.e_idx] = rating.rating
            else:
                user_ratings[rating.u_idx].descriptive_entity_ratings[rating.e_idx] = rating.rating

        return ratings, user_ratings, movie_indices, descriptive_entity_indices

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
    data_loader = DataLoader.load_from('./mindreader', filter_unknowns=True)
    print(data_loader.info())
