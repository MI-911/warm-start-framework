import json
import os
from typing import List

from data_loading.generic_data_loader import Rating


class Dataset:
    def __init__(self, path: str, experiments: List[str]):
        self.triples_path = os.path.join(path, 'triples.csv')

        if not os.path.exists(path):
            raise IOError(f'Dataset path {path} does not exist')

        if not os.path.exists(self.triples_path):
            raise IOError(f'Dataset path {path} does not contain triples')

        self.name = os.path.basename(path)
        self.experiment_paths = []

        for item in os.listdir(path):
            full_path = os.path.join(path, item)

            if not os.path.isdir(full_path) or experiments and item not in experiments:
                continue

            self.experiment_paths.append(full_path)

        if not self.experiment_paths:
            raise RuntimeError(f'Dataset path {path} contains no experiments')

        meta_file = os.path.join(path, 'meta.json')
        if not os.path.exists(meta_file):
            raise IOError(f'Dataset path {path} has no meta file')

        with open(meta_file, 'r') as fp:
            data = json.load(fp)

            for required in ['e_idx_map']:
                if required not in data:
                    raise RuntimeError(f'Dataset path {path} does not contain {required} in meta data')

            self.e_idx_map = data['e_idx_map']
            self.n_users = data['n_users']

    def __str__(self):
        return self.name

    def experiments(self):
        for path in self.experiment_paths:
            yield Experiment(self, path)


class Experiment:
    def __init__(self, parent, path):
        if not os.path.exists(path):
            raise IOError(f'Experiment path {path} does not exist')

        self.dataset = parent
        self.name = os.path.basename(path)
        self.split_paths = []

        for file in os.listdir(path):
            if not file.endswith('.json'):
                continue

            self.split_paths.append(os.path.join(path, file))

        if not self.split_paths:
            raise RuntimeError(f'Experiment path {path} contains no splits')

        self.split_paths = sorted(self.split_paths)

    def __str__(self):
        return f'{self.dataset}/{self.name}'

    def splits(self):
        for path in self.split_paths:
            yield Split(self, path)


class Split:
    def __init__(self, parent, path):
        if not os.path.exists(path):
            raise IOError(f'Split path {path} does not exist')

        self.experiment = parent
        self.name = os.path.basename(path)

        with open(path, 'r') as fp:
            data = json.load(fp)

            for required in ['training', 'testing', 'validation']:
                if required not in data:
                    raise RuntimeError(f'Split path {path} does not contain {required}')

            self.testing = data['testing']
            self.validation = data['validation']
            self.training = []

            movies = set()
            descriptive_entities = set()
            users = set()

            for user, ratings in data['training']:
                user_ratings = list()
                users.add(user)

                for rating in ratings:
                    is_movie_rating = rating['is_movie_rating']
                    e_idx = rating['e_idx']

                    user_ratings.append(Rating(user, e_idx, rating['rating'], is_movie_rating))
                    movies.add(e_idx) if is_movie_rating else descriptive_entities.add(e_idx)

                self.training.append((user, user_ratings))

            self.n_users = max(users) + 1 if users else 0
            self.n_descriptive_entities = max(descriptive_entities) + 1 if descriptive_entities else 0
            self.n_movies = max(movies) + 1 if movies else 0
            self.n_entities = max(self.n_movies, self.n_descriptive_entities)

    def __str__(self):
        return f'{self.experiment}/{self.name}'
