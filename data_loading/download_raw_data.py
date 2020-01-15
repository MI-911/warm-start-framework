import io
import json
import os
from os.path import join

import pandas as pd
import requests


def _ensure_directory_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def download_data(save_to='./mindreader', only_completed=True):
    """
    Downloads the mindreader dataset.
    :param save_to: Directory to save ratings.csv and entities.csv.
    :param only_completed: If True, only downloads ratings for users who reached the final screen.
    """
    _ensure_directory_exists(save_to)

    ratings_url = 'https://mindreader.tech/api/ratings?versions=100k,100k-newer,100k-fix'
    entities_url = 'https://mindreader.tech/api/entities'
    triples_url = 'https://mindreader.tech/api/triples'

    if only_completed:
        ratings_url += '&final=yes'

    ratings_response = requests.get(ratings_url)
    entities_response = requests.get(entities_url)
    triples_response = requests.get(triples_url)

    ratings = pd.read_csv(io.BytesIO(ratings_response.content))
    entities = pd.read_csv(io.BytesIO(entities_response.content))
    triples = pd.read_csv(io.BytesIO(triples_response.content))

    with open(join(save_to, 'ratings.csv'), 'w') as rfp:
        pd.DataFrame.to_csv(ratings, rfp)
    with open(join(save_to, 'entities.csv'), 'w') as efp:
        pd.DataFrame.to_csv(entities, efp)
    with open(join(save_to, 'triples.csv'), 'w') as efp:
        pd.DataFrame.to_csv(triples, efp)

    ratings = [(uid, uri, rating) for uid, uri, rating in ratings[['userId', 'uri', 'sentiment']].values]
    entities = [(uri, name, labels) for uri, name, labels in entities[['uri', 'name', 'labels']].values]

    # Filter out rating entities that aren't present in the entity set
    e_uris = [uri for uri, name, labels in entities]
    ratings = [(uid, uri, rating) for uid, uri, rating in ratings if uri in e_uris]

    with open(join(save_to, 'ratings_clean.json'), 'w') as rfp:
        json.dump(ratings, rfp)
    with open(join(save_to, 'entities_clean.json'), 'w') as efp:
        json.dump(entities, efp)


if __name__ == '__main__':
    download_data()
