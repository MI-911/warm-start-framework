import numpy as np
import pandas as pd

"""
Loads data almost unmanipulated.
"""


def load_csv_data_ratings(path, select_columns=None):
    """
    Loads data from a csv file and returns selected columns as a list of lists.
    :param path: Path to datafile.
    :param select_columns: A list of selected column names.
    :return: A list, with the selected columns as lists.
    """
    if select_columns is None:
        select_columns = ['userId', 'uri', 'isItem', 'sentiment']

    df = pd.read_csv(path)
    df = df[select_columns]

    return np.array([val.to_numpy() for _, val in df.iterrows()])


def load_csv_mindreader_ratings_with_indices(path):
    """
    Loads mindreader data ratings and creates index for entities and users
    :param path: Path to datafile
    :return: Data, user to index and entity to index
    """

    data = load_csv_data_ratings(path)

    user_index = {user: i for i, user in enumerate(set(data[:, 0]))}
    entity_index = {entity: i for i, entity in enumerate(set(data[:, 1]))}

    return data, user_index, entity_index


if __name__ == '__main__':
    a = load_csv_mindreader_ratings_with_indices('mindreader/ratings.csv')
    print(a[:10])
