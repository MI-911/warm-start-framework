import random
from collections import Counter
from random import shuffle

from data_loading.generic_data_loader import DataLoader


class DesignatedDataLoader(DataLoader):
    def __init__(self, args):
        super(DesignatedDataLoader, self).__init__(*args)
        self.train = []
        self.validation = []
        self.test = []

    def make(self, movie_to_entity_ratio=0.5, replace_movies_with_descriptive_entities=True, n_negative_samples=100,
             keep_all_ratings=False):
        """
        Samples new positive and negative items for every user.
        """
        u_r_map = {}
        for r in self.ratings:
            if r.u_idx not in u_r_map:
                u_r_map[r.u_idx] = []
            u_r_map[r.u_idx].append(r)

        if not keep_all_ratings:
            for u, ratings in list(u_r_map.items()):
                # Mix entity and movie ratings by the provided ratio
                # Comment out this part to simply include all loaded ratings for each user
                u_r_map[u] = self.mix_ratings(ratings, movie_to_entity_ratio, replace_movies_with_descriptive_entities)

        train, validation, test = [], [], []

        # We need to make sure that all negative samples also occur in the training set.
        movie_counts = {}
        for u, ratings in u_r_map.items():
            for m in [r.e_idx for r in ratings if r.is_movie_rating]:
                if m not in movie_counts:
                    movie_counts[m] = 0
                movie_counts[m] += 1

        for u, ratings in u_r_map.items():
            # Set the random generator for this user
            self.random = random.Random(self.random_seed + u + int(100 * movie_to_entity_ratio))

            # All positive samples must have at least twice appearance in the training set
            available_movies = [m for m, count in movie_counts.items() if count > 1]
            liked_movie_ratings = [
                r for r in self.ratings  # All ratings in the dataset (not necessarily in this training set)
                if r.is_movie_rating     # It's a movie
                and r.rating == 1        # It's a liked movie
                and r.e_idx in available_movies  # It appears at least twice in the training set
                and r.u_idx == u]        # It's rated by this user

            if len(liked_movie_ratings) < 2:
                # Just add their ratings to the training set, we cannot create a val/test set for this user.
                train.append((u, ratings))
                continue

            # Randomly sample the positive samples and remove them from the training ratings
            val_pos_sample = self.random.choice(liked_movie_ratings)
            liked_movie_ratings.remove(val_pos_sample)
            test_pos_sample = self.random.choice(liked_movie_ratings)
            liked_movie_ratings.remove(test_pos_sample)

            # If these movies appear in this user's training set, remove them
            if val_pos_sample in ratings:
                ratings.remove(val_pos_sample)
            if test_pos_sample in ratings:
                ratings.remove(test_pos_sample)

            # These samples now occur one less time in this training set
            val_pos_sample, test_pos_sample = val_pos_sample.e_idx, test_pos_sample.e_idx
            movie_counts[val_pos_sample] -= 1
            movie_counts[test_pos_sample] -= 1

            # Randomly sample 99 negative samples that all appear in the training set at least once
            val_neg_samples = self.sample_negative(u, movie_counts, n_negative_samples, val_pos_sample)
            test_neg_samples = self.sample_negative(u, movie_counts, n_negative_samples, test_pos_sample)

            assert len(val_neg_samples) == n_negative_samples
            assert len(test_neg_samples) == n_negative_samples

            train.append((u, ratings))
            validation.append((u, (val_pos_sample, val_neg_samples)))
            test.append((u, (test_pos_sample, test_neg_samples)))

        # Verify that all positive samples are not in a user's train ratings
        print(f'Asserting positive samples not in training set for each user...')
        for u, (pos_sample, neg_samples) in validation:
            train_movies = [r.e_idx for r in u_r_map[u]]
            assert pos_sample not in train_movies
            assert pos_sample in self.movie_indices

        for u, (pos_sample, neg_samples) in test:
            train_movies = [r.e_idx for r in u_r_map[u]]
            assert pos_sample not in train_movies
            assert pos_sample in self.movie_indices

        # Verify that all negative samples occur at least once in the training set
        print(f'Asserting negative samples occurrence in training set, but not rated for each user...')
        for u, (pos_sample, neg_samples) in validation:
            user_rated_movies = [r.e_idx for r in u_r_map[u]]
            for neg_sample in neg_samples:
                assert neg_sample not in user_rated_movies
                assert neg_sample in movie_counts and movie_counts[neg_sample] > 0
                assert neg_sample in self.movie_indices

        for u, (pos_sample, neg_samples) in test:
            user_rated_movies = [r.e_idx for r in u_r_map[u]]
            for neg_sample in neg_samples:
                assert neg_sample not in user_rated_movies
                assert neg_sample in movie_counts and movie_counts[neg_sample] > 0
                assert neg_sample in self.movie_indices

        # Same shit, different algo
        tra_ratings = []
        val_ratings = []
        tes_ratings = []

        for u, ratings in train:
            for r in ratings:
                tra_ratings.append(r.e_idx)

        for u, (pos, negs) in validation:
            for r in [pos] + negs:
                val_ratings.append(r)

        for u, (pos, negs) in test:
            for r in [pos] + negs:
                tes_ratings.append(r)

        for r in val_ratings:
            assert r in tra_ratings
        for r in tes_ratings:
            assert r in tra_ratings

        # Verify that no positive samples occur in the negative samples
        print(f'Asserting positive samples do not occur in negative samples...')
        for u, (pos_sample, neg_samples) in validation:
            assert pos_sample not in neg_samples

        for u, (pos_sample, neg_samples) in test:
            assert pos_sample not in neg_samples

        assert len(validation) == len(test)

        # Verify that all negative samples appear in the training set

        print(f'Returning a dataset over {len(train)} users.')

        self.train = train
        self.validation = validation
        self.test = test

        return train, validation, test

    def sample_negative(self, user, movie_counts, n, pos_sample):
        seen_movies = set([r.e_idx for r in self.ratings if r.is_movie_rating and r.u_idx == user] + [pos_sample])
        all_movies = set([m for m, count in movie_counts.items() if count > 0])
        unseen_movies = list(all_movies - seen_movies)
        self.random.shuffle(unseen_movies)

        return unseen_movies[:n]

    def sample(self, user, ratings, liked_movie_ratings, n_negative_samples, r, movie_counts):
        """
        Samples one liked movie and removes it from the user's ratings.
        Samples n_negative_samples unseen movies.
        """
        # Positive sample
        positive_sample = r.choice(liked_movie_ratings)
        liked_movie_ratings.remove(positive_sample)
        ratings.remove(positive_sample)
        movie_counts[positive_sample.e_idx] -= 1

        # Negative samples
        seen_movies = set([r.e_idx for r in self.ratings if r.is_movie_rating and r.u_idx == user])
        all_movies = set([m for m, count in movie_counts.items() if count > 0])
        unseen_movies = list(all_movies - seen_movies)
        r.shuffle(unseen_movies)
        negative_samples = unseen_movies[:n_negative_samples]

        return positive_sample.e_idx, negative_samples

    def mix_ratings(self, ratings, movie_to_entity_ratio, replace_movies_with_entities=True):
        movies = [rating for rating in ratings if rating.is_movie_rating]
        d_entities = [rating for rating in ratings if not rating.is_movie_rating]

        movie_length = int(len(movies) * movie_to_entity_ratio)
        d_entity_length = len(movies) - movie_length

        if len(d_entities) < d_entity_length and replace_movies_with_entities:
            d_entity_length = int(len(d_entities) * (1 - movie_to_entity_ratio))
            movie_length = len(d_entities) - d_entity_length

        # Randomly sample
        movies = self.random.sample(movies, movie_length)
        d_entities = self.random.sample(d_entities, d_entity_length) if replace_movies_with_entities else []

        # Return movies and descriptive entities shuffled
        ratings = movies + d_entities
        self.random.shuffle(ratings)
        return ratings

    @staticmethod
    def load_from(path, filter_unknowns=True, min_num_entity_ratings=5, movies_only=False, unify_user_indices=False,
                  remove_top_k_percent=None):
        return DesignatedDataLoader(DataLoader._load_from(
            path, filter_unknowns, min_num_entity_ratings, movies_only, unify_user_indices, remove_top_k_percent)
        )


def load_loo_data(path, movie_percentage=1., num_negative_samples=100, seed=42):
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
