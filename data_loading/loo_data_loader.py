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

        # Ensure user is in dictionary
        user_id = user_idx[user]
        if user_id not in user_ratings:
            user_ratings[user_id] = []

        user_ratings[user_id].append((entity_idx[uri], int(is_item)))

    user_ratings = list(user_ratings.items())
    shuffle(user_ratings)

    train = []
    validation = []
    test = []

    # Iterate over all users and sample a rating for validation and testing.
    for user, ratings in user_ratings:
        val_sample = __sample(ratings, idx_movie, idx_entity, entity_num_ratings)
        test_sample = __sample(ratings, idx_movie, idx_entity, entity_num_ratings)

        train.append((user, ratings))
        validation.append((user, val_sample))
        test.append((user, test_sample))

    return train, test


def __sample(ratings, idx_movie, idx_entity, entity_count):
    do_sample = True
    sample, rating = None, None

    while do_sample:
        sample, rating = random.sample(ratings, 1)[0]
        sample_uri = idx_entity[sample]

        if sample in idx_movie and \
                sample_uri in entity_count and \
                entity_count[sample_uri] - 1 > 0:
            do_sample = False
            entity_count[sample_uri] -= 1

    ratings.remove((sample, rating))

    return sample, rating


if __name__ == '__main__':
    load_loo_data('mindreader/ratings.csv')




