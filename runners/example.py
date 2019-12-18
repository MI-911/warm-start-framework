from data_loading.loo_data_loader import DesignatedDataLoader
from data_loading.generic_data_loader import Rating
from models.base_recommender import RecommenderBase


if __name__ == '__main__':

    # Load the data
    data_loader = DesignatedDataLoader.load_from(
        path='../data_loading/mindreader',  # Where to load MindReader ratings from
        min_num_entity_ratings=5,           # Entities with < this num ratings will not be loaded
        movies_only=False,                  # If True, no descriptive entities are loaded
        unify_user_indices=False,           # If True, users are indexed in the same space as entities
        remove_top_k_percent=.1             # If not None, removes the top-k% popular MOVIES from the rating set
    )

    # Fields
    _ = data_loader.info()  # Statistics string
    _ = data_loader.n_movies  # Number of movies
    _ = data_loader.n_users  # Number of users
    _ = data_loader.n_descriptive_entities  # Number of DEs

    _ = data_loader.movie_indices  # The movie indices
    _ = data_loader.descriptive_entity_indices  # The DE indices

    _ = data_loader.backwards_e_map  # Dictionary mapping entity indices --> URIs
    _ = data_loader.backwards_u_map  # Dictionary mapping user indices --> UIDs

    # Set the run seed
    data_loader.random_seed = 42

    # Generate training, validation, and test sets
    train, validation, test = data_loader.make(
        movie_to_entity_ratio=4/4,                      # How many of each user's movie ratings should we keep?
        replace_movies_with_descriptive_entities=True,  # Should the removed movie ratings be replaced by DEs?
        n_negative_samples=99,                          # How many negative samples should accompany the positive?
        keep_all_ratings=False                          # If True, does nothing to the users' ratings
    )

    # -------- STRUCTURE --------- #
    # The training set is structured as follows:
    # [
    #   (user_index_1, [Rating() objects]),
    #   (user_index_2, [Rating() objects])
    # ]

    # The validation and test sets are structured as follows:
    # [
    #   (user_index_1, (positive_movie_index, [negative_movie_indices])),
    #   (user_index_2, (positive_movie_index, [negative_movie_indices]))
    # ]

    # -------- SCENARIOS -------- #
    # -------- SCENARIO 1: Run 10 runs, replacing removed movie ratings with DE ratings
    for run in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        data_loader = DesignatedDataLoader.load_from(
            path='../data_loading/mindreader',
            min_num_entity_ratings=5,
            movies_only=False,
            unify_user_indices=False
        )

        print(data_loader.info())

        data_loader.random_seed = run

        for n in [4, 3, 2, 1]:
            tra, val, te = data_loader.make(
                movie_to_entity_ratio=n/4,
                replace_movies_with_descriptive_entities=True,  # Removed movie ratings ((4-n)/4) replaced with DEs
                n_negative_samples=99,
                keep_all_ratings=False
            )

            recommender = RecommenderBase(model='Some model')
            recommender.fit(training=tra, validation=val)

            for u, (pos_sample, neg_samples) in te:
                scores = recommender.predict(u, neg_samples + [pos_sample])
                predicted_rank = scores[pos_sample]

                for k in range(50):
                    # Calculate Hit@k for predicted_rank
                    # Calculate DCG@k for predicted_rank
                    # Store in save file
                    pass

    # -------- SCENARIO 2: Run 10 runs, just remove movie ratings, don't replace with DEs
    for run in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        data_loader = DesignatedDataLoader.load_from(
            path='../data_loading/mindreader',
            min_num_entity_ratings=5,
            movies_only=False,
            unify_user_indices=False
        )

        print(data_loader.info())

        data_loader.random_seed = run

        for n in [4, 3, 2, 1]:
            tra, val, te = data_loader.make(
                movie_to_entity_ratio=n / 4,
                replace_movies_with_descriptive_entities=False,  # Movie ratings are removed, but aren't replaced
                n_negative_samples=99,
                keep_all_ratings=False
            )

    # -------- SCENARIO 3: Run 10 runs, include all users' movie and entity ratings
    for run in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        data_loader = DesignatedDataLoader.load_from(
            path='../data_loading/mindreader',
            min_num_entity_ratings=5,
            movies_only=False,  # Load both movie and entity ratings
            unify_user_indices=False
        )

        print(data_loader.info())

        data_loader.random_seed = run

        tra, val, te = data_loader.make(
            movie_to_entity_ratio=4/4,
            replace_movies_with_descriptive_entities=False,
            n_negative_samples=99,
            keep_all_ratings=True  # Don't touch the users' ratings before making the training set
        )

    # -------- SCENARIO 4: Run 10 runs, include all users' movie ratings only
    for run in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        data_loader = DesignatedDataLoader.load_from(
            path='../data_loading/mindreader',
            min_num_entity_ratings=5,
            movies_only=True,  # Load only movie ratings
            unify_user_indices=False
        )

        print(data_loader.info())

        data_loader.random_seed = run

        tra, val, te = data_loader.make(
            movie_to_entity_ratio=4 / 4,
            replace_movies_with_descriptive_entities=False,
            n_negative_samples=99,
            keep_all_ratings=True  # Don't touch the users' ratings before making the training set
        )

    # --------- SCENARIO 4: Run 10 runs, replace removed movie ratings with DEs, don't include top-10 popular entities
    for run in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        data_loader = DesignatedDataLoader.load_from(
            path='../data_loading/mindreader',
            min_num_entity_ratings=5,
            movies_only=False,
            unify_user_indices=False,
            remove_top_k_percent=.1  # Remove the top-10 popular movies from the ratings set
        )

        print(data_loader.info())

        data_loader.random_seed = run

        for n in [4, 3, 2, 1]:
            tra, val, te = data_loader.make(
                movie_to_entity_ratio=n / 4,
                replace_movies_with_descriptive_entities=True,  # Removed movie ratings ((4-n)/4) replaced with DEs
                n_negative_samples=99,
                keep_all_ratings=False
            )