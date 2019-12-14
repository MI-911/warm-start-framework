import random
from collections import Counter
from random import shuffle

from data_loading.generic_data_loader import DataLoader


def load_loo_data(path, movie_percentage=1, num_negative_samples=100, seed=1):
    random.seed(seed)
    data_loader = DataLoader.load_from(path)

    # Get number of ratings per entity
    entity_num_ratings = Counter([rating.e_idx for rating in data_loader.ratings])

    # Get rating per user
    user_ratings = {}
    for rating in data_loader.ratings:
        # Skip entities with to few ratings
        if entity_num_ratings[rating.e_idx] < 3:
            continue

        # Ensure user is in dictionary
        if rating.u_idx not in user_ratings:
            user_ratings[rating.u_idx] = []

        user_ratings[rating.u_idx].append(rating)

    user_ratings = list(user_ratings.items())
    shuffle(user_ratings)

    train = []
    validation = []
    test = []

    # Iterate over all users and sample a rating for validation and testing.
    for user, ratings in user_ratings:
        all_movie_ratings = [rating.e_idx for rating in ratings if rating.is_movie_rating]

        # Skip users with too few positive ratings.
        if len([1 for rating in ratings if rating.is_movie_rating and rating.rating == 1]) < 2:
            continue

        test_sample = __sample(data_loader, ratings, all_movie_ratings, entity_num_ratings, num_negative_samples)
        val_sample = __sample(data_loader, ratings, all_movie_ratings, entity_num_ratings, num_negative_samples)

        train.append((user, ratings))
        validation.append((user, val_sample))
        test.append((user, test_sample))

    return train, validation, test


def __sample(data_loader, modified_ratings, all_ratings, entity_count, num_negative):
    do_sample = True
    sample = None

    while do_sample:
        sample = random.sample(modified_ratings, 1)[0]

        # Ensure sample is movie, rated positively and is in train set.
        if sample.is_movie_rating and \
                sample.rating == 1 and \
                sample.e_idx in entity_count and \
                entity_count[sample.e_idx] - 1 > 0:
            do_sample = False
            entity_count[sample.e_idx] -= 1

    modified_ratings.remove(sample)

    do_sample = True
    negative_samples = []
    while do_sample:
        negative_samples = random.sample(data_loader.movie_indices, num_negative)

        # Continue if negative samples intersect with rating
        if not set(negative_samples).intersection(set(all_ratings)):
            do_sample = False

    return sample.e_idx, negative_samples


if __name__ == '__main__':
    load_loo_data('mindreader/')




