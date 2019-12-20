from models.pagerank.kg_pagerank_recommender import KnowledgeGraphPageRankRecommender
from data_loading.loo_data_loader import DesignatedDataLoader
from metrics.metrics import dcg
import numpy as np
import json
import os


def get_rank_of(item, score_dict):
    sorted_score_dict = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    for rank, (i, s) in enumerate(sorted_score_dict):
        if i == item:
            return rank


def run(save_dir, model_name):
    for run in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        data_loader = DesignatedDataLoader.load_from(
            path='../data_loading/mindreader',
            min_num_entity_ratings=5,
            movies_only=False,
            unify_user_indices=False
        )

        # Result files are stored with the following naming convention:
        #  - ../results/<MODEL_NAME>/<RUN>/<FILENAME>  for test results
        #  - ../results/<MODEL_NAME>/<RUN>/training/<FILENAME>  for (under training) validation/training results
        #
        # Make sure that <FILENAME> is different for every configuration of the dataset and model.

        print(data_loader.info())

        data_loader.random_seed = run

        SAVE_DIR = os.path.join(f'../{save_dir}/{model_name}', str(run))
        TRAINING_SAVE_DIR = os.path.join(f'../{save_dir}/{model_name}/training', str(run))

        replace_movies_with_descriptive_entities = False
        n_negative_samples = 100
        keep_all_ratings = False
        with_kg_triples = False
        with_standard_corruption = True

        # Generate unique file name from the configuration
        file_name = model_name
        file_name += '.json'

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        if not os.path.exists(TRAINING_SAVE_DIR):
            os.makedirs(TRAINING_SAVE_DIR)

        tra, val, te = data_loader.make(
            movie_to_entity_ratio=4 / 4,
            replace_movies_with_descriptive_entities=replace_movies_with_descriptive_entities,
            n_negative_samples=n_negative_samples,
            keep_all_ratings=keep_all_ratings
        )

        recommender = KnowledgeGraphPageRankRecommender(data_loader=data_loader)

        print(f'Fitting {file_name} at run {run}...')

        recommender.fit(
            training=tra,
            validation=val,
            max_iterations=100,
            verbose=True,
            save_to=os.path.join(TRAINING_SAVE_DIR, file_name)
        )

        hits_at_k = [[] for i in range(50)]
        dcgs_at_k = [[] for i in range(50)]

        for u, (pos_sample, neg_samples) in te:
            scores = recommender.predict(u, neg_samples + [pos_sample])
            rank = get_rank_of(pos_sample, scores)

            for k in range(50):
                hits_at_k[k].append(1 if rank < k + 1 else 0)
                dcgs_at_k[k].append(dcg(rank, k + 1))

        hits_at_k = list(map(float, [np.mean(hits) for hits in hits_at_k]))
        dcgs_at_k = list(map(float, [np.mean(dcgs) for dcgs in dcgs_at_k]))

        with open(os.path.join(SAVE_DIR, file_name), 'w') as fp:
            json.dump({
                'hits_at_k': hits_at_k,
                'dcgs_at_k': dcgs_at_k
            }, fp, indent=True)


