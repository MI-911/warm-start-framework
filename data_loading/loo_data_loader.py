import random
from collections import Counter
from random import shuffle

from data_loading.generic_data_loader import DataLoader


def load_loo_data(path, movie_percentage=1., num_negative_samples=100, seed=1):
    random.seed(seed)
    data_loader = DataLoader.load_from(path)

    # Get number of ratings per entity
    entity_num_ratings = Counter([rating.e_idx for rating in data_loader.ratings])

    # Get rating per user
    user_ratings = {}
    for rating in data_loader.ratings:
        # Skip entities with to few ratings
        if entity_num_ratings[rating.e_idx] < 2:
            continue

        # Ensure user is in dictionary
        if rating.u_idx not in user_ratings:
            user_ratings[rating.u_idx] = []

        user_ratings[rating.u_idx].append(rating)

    # Filter ratings to have part movies and part entities.
    for user, ratings in user_ratings.items():
        user_ratings[user] = __filter_ratings(ratings, movie_percentage)
    entity_num_ratings = Counter([rating.e_idx for rating in data_loader.ratings])

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
    """
    Samples a liked movie, and find n negative samples (unrated by the user). Note changes modified_ratings and
    entity_count by reference.
    :param data_loader: Used for movie indexes :type data_loader: DataLoader
    :param modified_ratings: Rating that allows for side effects. :type modified_ratings: [Ratings]
    :param all_ratings: All movie ratings for the user. :type all_ratings: [int]
    :param entity_count: Number of ratings left in dataset for each entity. :type entity_count: dict
    :param num_negative: Number of negative samples to sample, :type num_negative: int
    :return: Tuple containing positive sample, and a list of negative samples.
    """
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


def __filter_ratings(ratings, movie_percentage):
    movies = [rating for rating in ratings if rating.is_movie_rating]
    d_entities = [rating for rating in ratings if not rating.is_movie_rating]

    movie_length = int(len(movies) * movie_percentage)
    d_entity_length = len(movies) - movie_length

    if len(d_entities) < d_entity_length:
        d_entity_length = int(len(d_entities) * (1 - movie_percentage))
        movie_length = len(d_entities) - d_entity_length

    # Randomly sample
    movies = random.sample(movies, movie_length)
    d_entities = random.sample(d_entities, d_entity_length)

    # Return movies and descriptive entities shuffled
    ratings = movies + d_entities
    random.shuffle(ratings)
    return ratings


if __name__ == '__main__':
    load_loo_data('mindreader/', movie_percentage=0.5)




