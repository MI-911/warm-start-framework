from data_loading.loo_data_loader import DesignatedDataLoader
from data_loading.generic_data_loader import Rating
from models.other_trans_e import TransE
import numpy as np
import torch as tt
import pandas as pd
import random
import json
import os


def unify_user_indices(ratings, u_idx_start):
    unified = []
    for user_index, rest in ratings:
        unified.append((user_index + u_idx_start, rest))
        u_idx_start += 1

    return unified


def convert_ratings(ratings):
    converted = []
    for user_index, rs in ratings:
        converted.append((user_index, [
            Rating(r.u_idx, r.e_idx, 1 if r.rating == 1 else 0, r.is_movie_rating)
            for r in rs
        ]))

    return converted


def flatten_ratings(user_ratings, movies_only=False):
    rating_triples = []
    for u, rs in user_ratings:
        rating_triples += (
            [(r.u_idx, 1 if r.rating == 1 else 0, r.e_idx) for r in rs if r.is_movie_rating] if movies_only else
            [(r.u_idx, 1 if r.rating == 1 else 0, r.e_idx) for r in rs]
        )

    return rating_triples


def corrupt_std(flat_ratings, all_entities):
    corrupted = []
    for h, r, t in flat_ratings:
        if random.random() > 0.5:
            corrupted.append((random.choice(all_entities), r, t))
        else:
            corrupted.append((h, r, random.choice(all_entities)))

    return corrupted


def batchify(pos, neg, batch_size=64):
    for i in range(0, len(pos), batch_size):
        yield zip(*pos[i:i + batch_size]), zip(*neg[i:i + batch_size])


def evaluate_loss(model, user_ratings, pre_str='Training'):
    with tt.no_grad():
        model.eval()

        all_ratings = flatten_ratings(user_ratings, movies_only=True)
        heads, relations, tails = zip(*all_ratings)

        loss = model(heads, relations, tails)
        print(f'[{pre_str}] Mean distance between e(h) + e(r) and e(t): {loss.mean()}')

        return loss


def dcg(rank, n=10):
    r = np.zeros(n)
    if rank < n:
        r[rank] = 1

    return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))


def evaluate_hit(model, user_samples, pre_str, n=10):
    with tt.no_grad():
        model.eval()
        # Each entry is (user, (pos_sample, neg_samples))
        ranks = []
        dcgs = []
        for user, (pos_sample, neg_samples) in user_samples:
            fast_rank = model.fast_rank(user, 1, pos_sample, neg_samples)
            fast_dcg = dcg(fast_rank, n=n)
            ranks.append(fast_rank)
            dcgs.append(fast_dcg)

        print(f'[{pre_str}] Mean rank: {np.mean(ranks)}')
        print(f'[{pre_str}] Hit@10:    {len(np.where(np.array(ranks) < n)[0]) / len(user_samples)}')
        print(f'[{pre_str}] DCG@10:    {np.mean(dcgs)}')

        return np.mean(ranks), len(np.where(np.array(ranks) < n)[0]) / len(user_samples), np.mean(dcgs)


def load_kg_triples(e_idx_map):
    with open('../data_loading/mindreader/triples.csv') as fp:
        df = pd.read_csv(fp)
        triples = [(h, r, t) for h, r, t in df[['head_uri', 'relation', 'tail_uri']].values]
        triples = [(e_idx_map[h], r, e_idx_map[t]) for h, r, t in triples if h in e_idx_map and t in e_idx_map]

    indexed_triples = []
    r_idx_map = {}
    rc = 2
    for h, r, t in triples:
        if r not in r_idx_map:
            r_idx_map[r] = rc
            rc += 1
        indexed_triples.append((h, r_idx_map[r], t))

    return indexed_triples, r_idx_map


def get_like_matrix(ratings, n_users, n_entities):
    u_idx_to_matrix_map, uc = {}, 0
    e_idx_to_matrix_map, ec = {}, 0

    R = np.zeros((n_users, n_entities))
    for u, r, e in ratings:
        if u not in u_idx_to_matrix_map:
            u_idx_to_matrix_map[u] = uc
            uc += 1
        if e not in e_idx_to_matrix_map:
            e_idx_to_matrix_map[e] = ec
            ec += 1

        R[u_idx_to_matrix_map[u]][e_idx_to_matrix_map[e]] = 1 if r == 1 else -1

    return R, u_idx_to_matrix_map, e_idx_to_matrix_map


def corrupt_rating_triples(triples, ratings_matrix, u_idx_to_matrix_map, e_idx_to_matrix_map):
    corrupted = []
    for h, r, t in triples:
        h_mat = u_idx_to_matrix_map[h]
        t_mat = e_idx_to_matrix_map[t]
        if random.random() > 0.5:
            if r == 1:
                # Find a user that dislikes t
                users_disliking_t = np.argwhere(ratings_matrix[:, t_mat] == -1).flatten()
                if len(users_disliking_t) == 0:
                    users_disliking_t = list(e_idx_to_matrix_map.keys())
                corrupted.append((random.choice(users_disliking_t), r, t))
            else:
                # Find a user that likes t
                users_liking_t = np.argwhere(ratings_matrix[:, t_mat] == 1).flatten()
                if len(users_liking_t) == 0:
                    users_liking_t = list(e_idx_to_matrix_map.keys())
                corrupted.append((random.choice(users_liking_t), r, t))
        else:
            if r == 1:
                # Find an item that h dislikes
                items_disliked_by_h = np.argwhere(ratings_matrix[h_mat] == -1).flatten()
                if len(items_disliked_by_h) == 0:
                    items_disliked_by_h = list(e_idx_to_matrix_map.keys())
                corrupted.append((h, r, random.choice(items_disliked_by_h)))
            else:
                # Find an item that h likes
                items_liked_by_h = np.argwhere(ratings_matrix[h_mat] == 1).flatten()
                if len(items_liked_by_h) == 0:
                    items_liked_by_h = list(e_idx_to_matrix_map.keys())
                corrupted.append((h, r, random.choice(items_liked_by_h)))

    return corrupted


if __name__ == '__main__':
    for random_seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        configs = [
            # Standard corruption, no KG

            # [N_EPOCHS, MOVIE_RATIO, ALL_RATIO, STANDARD_CORRUPTION, WITH_KG_TRIPLES]
            # [100, 4, 4, True, False],
            # [100, 3, 4, True, False],
            # [100, 2, 4, True, False],
            # [100, 1, 4, True, False],
            #
            # # Custom corruption, no KG
            # [100, 4, 4, False, False],
            # [100, 3, 4, False, False],
            # [100, 2, 4, False, False],
            # [100, 1, 4, False, False],

            # Standard corruption, with KG
            # [100, 4, 4, True, True],
            # [100, 3, 4, True, True],
            # [100, 2, 4, True, True],
            # [100, 1, 4, True, True],

            # Custom corruption, with KG
            [100, 4, 4, False, True],
            [100, 3, 4, False, True],
            [100, 2, 4, False, True],
            [100, 1, 4, False, True],
        ]
        for n_epochs, movie_ratio, all_ratio, standard_corruption, with_kg_triples in configs:
            # Training configuration
            # n_epochs = 100
            # movie_ratio = 4
            # all_ratio = 4
            # standard_corruption = True
            # with_kg_triples = False

            training_loss_history = []

            validation_hitrate_history = []
            validation_mean_rank_history = []
            validation_dcg_history = []

            testing_hitrate_history = []
            testing_mean_rank_history = []
            testing_dcg_history = []

            data_loader = DesignatedDataLoader.load_from(
                path='../data_loading/mindreader',
                movies_only=False,
                min_num_entity_ratings=5,
                filter_unknowns=True,
                unify_user_indices=True
            )

            data_loader.random_seed = random_seed

            print(data_loader.info())

            train, validation, test = data_loader.make(
                movie_to_entity_ratio=movie_ratio / all_ratio,
                n_negative_samples=99,
                replace_movies_with_descriptive_entities=True
            )

            # Hyper parameters
            n_latent_factors = 50
            learning_rate = 0.003
            n_total_entities = data_loader.n_users + data_loader.n_descriptive_entities + data_loader.n_movies
            n_total_entities_no_users = n_total_entities - data_loader.n_users
            n_relations = 2

            # Data pre-processing
            # Convert user indices to same index space as entities
            # train = unify_user_indices(train, n_total_entities_no_users)
            # validation = unify_user_indices(validation, n_total_entities_no_users)
            # test = unify_user_indices(test, n_total_entities_no_users)
            # Convert likes/dislikes to their relation indices
            train = convert_ratings(train)

            # What indices are for users, movies and entities, respectively?
            user_indices = list(range(n_total_entities_no_users, n_total_entities))
            movie_indices = data_loader.movie_indices
            descriptive_entity_indices = data_loader.descriptive_entity_indices

            # KG triples
            # kg_triples, r_idx_map = load_kg_triples(data_loader.e_idx_map)
            # n_relations += len(r_idx_map)

            kg_triples, r_idx_map = load_kg_triples(data_loader.e_idx_map) if with_kg_triples else ([], {})
            n_relations += len(r_idx_map)

            # Rating triples
            all_train_ratings = flatten_ratings(train)
            ratings_matrix, u_idx_to_matrix_map, e_idx_to_matrix_map = get_like_matrix(
                all_train_ratings,
                data_loader.n_users,
                data_loader.n_movies + data_loader.n_descriptive_entities)

            # Model building
            model = TransE(n_total_entities, n_relations, margin=1.0, k=n_latent_factors)
            optimizer = tt.optim.Adam(model.parameters(), lr=learning_rate)


            def save():
                file_name = f'trans_e_{n_epochs}_epochs_{movie_ratio}-{all_ratio}_{"KG" if with_kg_triples else "NOKG"}_{"SC" if standard_corruption else "CC"}.json'
                to_save = {
                    'file_name': file_name,
                    'n_epochs': n_epochs,
                    'movie_to_entity_ratio': f"{movie_ratio}/{all_ratio}",
                    'with_kg_triples': with_kg_triples,
                    'standard_corruption': standard_corruption,
                    'train_loss': training_loss_history,
                    'validation': {
                        'mean_rank': validation_mean_rank_history,
                        'hit_ratio': validation_hitrate_history,
                        'dcg': validation_dcg_history
                    },
                    'testing': {
                        'mean_rank': testing_mean_rank_history,
                        'hit_ratio': testing_hitrate_history,
                        'dcg': testing_dcg_history
                    }
                }

                if not os.path.exists(f'../results/trans_e/user_seed/{random_seed}'):
                    os.makedirs(f'../results/trans_e/user_seed/{random_seed}')

                with open(f'../results/trans_e/user_seed/{random_seed}/{file_name}', 'w') as fp:
                    json.dump(to_save, fp, indent=True)


            for epoch in range(n_epochs):
                # 1. Evaluate
                #    a) The loss (mean distance from e(h) + e(r) to e(t))
                #    b) The validation hitrate ()

                if epoch % 5 == 0:
                    print(
                        f'Epoch {epoch} [TransE@{n_epochs} epochs, {movie_ratio}/{all_ratio} movies, {"Standard Corruption" if standard_corruption else "Custom Corruption"}, {"With KG" if with_kg_triples else "No KG"}]')
                    train_loss = evaluate_loss(model, train, pre_str='TRAIN')
                    training_loss_history.append(float(train_loss.mean().cpu().numpy().sum()))

                    val_mean_rank, val_hit_rate, val_dcg = evaluate_hit(model, validation, pre_str='VALIDATION')
                    test_mean_rank, test_hit_rate, test_dcg = evaluate_hit(model, test, pre_str='TEST')

                    validation_hitrate_history.append(float(val_hit_rate))
                    validation_mean_rank_history.append(float(val_mean_rank))
                    validation_dcg_history.append(float(val_dcg))
                    testing_hitrate_history.append(float(test_hit_rate))
                    testing_mean_rank_history.append(float(test_mean_rank))
                    testing_dcg_history.append(float(test_dcg))

                    save()

                # 2. Corrupt triples
                #    a) FIRST: Standard corruption
                #    b) TRY: Different corruptions for ratings and KG triples

                corrupted_train_ratings = (
                    corrupt_std(all_train_ratings, user_indices + movie_indices + descriptive_entity_indices)
                    if standard_corruption else
                    corrupt_rating_triples(all_train_ratings, ratings_matrix, u_idx_to_matrix_map, e_idx_to_matrix_map))
                corrupted_train_kg_triples = corrupt_std(kg_triples, movie_indices + descriptive_entity_indices)

                all_pairs = list(zip(all_train_ratings, corrupted_train_ratings))
                all_pairs += list(zip(kg_triples, corrupted_train_kg_triples))

                random.shuffle(all_pairs)
                positive_samples, negative_samples = zip(*all_pairs)

                # 3. Train
                model.train()
                for (p_h, p_r, p_t), (n_h, n_r, n_t) in batchify(positive_samples, negative_samples):
                    p_distance = model(p_h, p_r, p_t)
                    n_distance = model(n_h, n_r, n_t)

                    loss = tt.relu(model.margin + p_distance - n_distance).sum()

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            print(
                f'Epoch {epoch} [TransE@{n_epochs} epochs, {movie_ratio}/{all_ratio} movies, {"Standard Corruption" if standard_corruption else "Custom Corruption"}, {"With KG" if with_kg_triples else "No KG"}]')
            train_loss = evaluate_loss(model, train, pre_str='TRAIN')
            training_loss_history.append(float(train_loss.mean().cpu().numpy().sum()))

            val_mean_rank, val_hit_rate, val_dcg = evaluate_hit(model, validation, pre_str='VALIDATION')
            test_mean_rank, test_hit_rate, test_dcg = evaluate_hit(model, test, pre_str='TEST')

            validation_hitrate_history.append(float(val_hit_rate))
            validation_mean_rank_history.append(float(val_mean_rank))
            validation_dcg_history.append(float(val_dcg))
            testing_hitrate_history.append(float(test_hit_rate))
            testing_mean_rank_history.append(float(test_mean_rank))
            testing_dcg_history.append(float(test_dcg))

            save()

            # train = [
            #   (user_index, [Rating() objects]),
            #   (user_index, [Rating() objects]),
            #       ...
            #   ]

            # validation = [
            #   (user_index, (positive_entity_id, [negative_entity_ids])),
            #   (user_index, (positive_entity_id, [negative_entity_ids])),
            #       ...
            # ]

            # test = [
            #   (user_index, (positive_entity_id, [negative_entity_ids])),
            #   (user_index, (positive_entity_id, [negative_entity_ids])),
            #       ...
            # ]


