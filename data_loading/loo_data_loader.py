import random
from collections import Counter
from random import shuffle

from data_loading.generic_data_loader import load_csv_data_ratings, load_csv_mindreader_ratings_with_indices


def load_loo_data(path, movie_percentage=1):
    data, user_idx, entity_idx = load_csv_mindreader_ratings_with_indices(path)
    idx_entity = {idx: entity for entity, idx in entity_idx.items()}

    # Get movie indices
    idx_movie = {entity_idx[uri]: uri for uri in set(data[data[:, 2].astype(bool)][:, 1])}

    # Get number of ratings per entity
    entity_num_ratings = Counter(data[:, 1])

    # Get rating per user
    user_ratings = {}
    for user, uri, is_item, rating in data:
        # Skip entities with to few ratings
        if entity_num_ratings[uri] < 2:
            continue
        user_id = user_idx[user]
        if user_id not in user_ratings:
            user_ratings[user_id] = []

        user_ratings[user_id].append((entity_idx[uri], int(is_item)))

    user_ratings = list(user_ratings.items())
    shuffle(user_ratings)

    train = []
    test = []
    for user, ratings in user_ratings:
        do_sample = True
        sample, rating = None, None

        while do_sample:
            sample, rating = random.sample(ratings, 1)[0]
            sample_uri = idx_entity[sample]

            if sample in idx_movie and \
                    sample_uri in entity_num_ratings and \
                    entity_num_ratings[sample_uri] - 1 > 0:
                do_sample = False

        entity_num_ratings[sample_uri] -= 1
        ratings.remove((sample, rating))
        train.append((user, ratings))
        test.append((user, (sample, rating)))

    return train, test


if __name__ == '__main__':
    load_loo_data('mindreader/ratings.csv')




