import json
import os

from data_loading.generic_data_loader import Rating


class Dataset:
    def __init__(self, path):
        if not os.path.exists(path):
            raise IOError(f'Dataset path {path} does not exist')

        self.name = os.path.dirname(path)
        self.experiment_paths = []

        for item in os.listdir(path):
            item = os.path.join(path, item)

            if not os.path.isdir(item):
                continue

            self.experiment_paths.append(item)

        if not self.experiment_paths:
            raise RuntimeError(f'Dataset path {path} contains no experiments')

        meta_file = os.path.join(path, 'meta.json')
        if not os.path.exists(meta_file):
            raise IOError(f'Dataset path {path} has no meta file')

        with open(meta_file, 'r') as fp:
            data = json.load(fp)

            for required in ['e_idx_map', 'n_users']:
                if required not in data:
                    raise RuntimeError(f'Dataset path {path} does not contain {required} in meta data')

            self.e_idx_map = data['e_idx_map']
            self.n_users = data['n_users']

    def experiments(self):
        for path in self.experiment_paths:
            yield Experiment(path)


class Experiment:
    def __init__(self, path):
        if not os.path.exists(path):
            raise IOError(f'Experiment path {path} does not exist')

        self.name = os.path.dirname(path)
        self.split_paths = []
        for file in os.listdir(path):
            if not file.endswith('.json'):
                continue

            self.split_paths.append(os.path.join(path, file))

        if not self.split_paths:
            raise RuntimeError(f'Experiment path {path} contains no splits')

    def splits(self):
        for path in self.split_paths:
            yield Split(path)


class Split:
    def __init__(self, path):
        if not os.path.exists(path):
            raise IOError(f'Split path {path} does not exist')

        self.name = os.path.basename(path)

        with open(path, 'r') as fp:
            data = json.load(fp)

            for required in ['training', 'testing', 'validation']:
                if required not in data:
                    raise RuntimeError(f'Split path {path} does not contain {required}')

            self.testing = data['testing']
            self.validation = data['validation']
            self.training = []

            for user, ratings in data['training']:
                self.training.append((user, [Rating(rating['u_idx'], rating['e_idx'], rating['rating'],
                                                    rating['is_movie_rating']) for rating in ratings]))


if __name__ == '__main__':
    Experiment('../data/top')
