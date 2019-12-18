from models.trans_e_recommender import TransERecommender
from data_loading.loo_data_loader import DesignatedDataLoader

if __name__ == '__main__':
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

            recommender = TransERecommender(n_entities=100, n_relations=200, margin=1, n_latent_factors=50)
            recommender.fit(training=tra, validation=val)

            for u, (pos_sample, neg_samples) in te:
                scores = recommender.predict(u, neg_samples + [pos_sample])

                for k in range(50):
                    # Calculate Hit@k for predicted_rank
                    # Calculate DCG@k for predicted_rank
                    # Store in save file
                    pass
