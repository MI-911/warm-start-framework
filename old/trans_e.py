import torch as tt
import torch.optim as optim
from models.trans_e import TransE
from data_loading.k_fold_data_loader import KFoldLoader, Rating
import random
import numpy as np
import pandas as pd


# Values as they appear in MindReader ratings
DISLIKE = -1
LIKE = 1


def unify_user_entity_indices(ratings, starting_index=0):
    unified = []
    for r in ratings:
        unified.append(Rating(r.u_idx + starting_index, r.e_idx, r.rating, r.is_movie_rating))

    return unified


def convert_hrt_triples(ratings, r_idx_map):
    return [(r.u_idx, r_idx_map[r.rating], r.e_idx) for r in ratings]


def corrupt_hrt_triples(triples, all_entities):
    corrupted = []
    for h, r, t in triples:
        if random.random() > 0.5:
            corrupted.append((random.choice(all_entities), r, t))
        else:
            corrupted.append((h, r, random.choice(all_entities)))

    return corrupted


def batchify(triples, batch_size=64):
    for i in range(0, len(triples), batch_size):
        yield zip(*triples[i:i + batch_size])


def evaluate_loss(model, triples, pre_string=None):
    with tt.no_grad():
        model.eval()

        heads, relations, tails = zip(*triples)
        loss = model(heads, relations, tails)

        print(f'[{pre_string}] Mean distance between e(h) + e(r) and e(t): {loss.sum() / len(triples)}')


def evaluate_rank(model, triples, pre_string=None):
    with tt.no_grad():
        model.eval()

        mean_ranks = []

        for h, r, t in triples:
            if r == 1:
                mean_ranks.append(model.fast_validate(h, r, t, data_loader.movie_indices))

        print(f'[{pre_string}] Mean ranks: {np.mean(mean_ranks)}')


def load_kg_triples(e_idx_map, r_idx_map):
    with open('../data_loading/mindreader/triples.csv') as fp:
        df = pd.read_csv(fp)

    entity_starting_index = len(e_idx_map)
    relation_starting_index = len(r_idx_map)

    kg_triples = []
    for h, r, t, in df[['head_uri', 'relation', 'tail_uri']].values:
        if h not in e_idx_map:
            e_idx_map[h] = entity_starting_index
            entity_starting_index += 1
        if r not in r_idx_map:
            r_idx_map[r] = relation_starting_index
            relation_starting_index += 1
        if t not in e_idx_map:
            e_idx_map[t] = entity_starting_index
            entity_starting_index += 1

        kg_triples.append((e_idx_map[h], r_idx_map[r], e_idx_map[t]))

    return kg_triples


if __name__ == '__main__':
    data_loader = KFoldLoader.load_from('../data_loading/mindreader',
                                        filter_unknowns=True,
                                        movies_only=False,
                                        min_num_entity_ratings=5)

    data_loader.fill_buckets(5)
    n_epochs = 100
    batch_size = 64

    # Load the knowledge graph triples
    e_idx_map = data_loader.e_idx_map
    r_idx_map = {DISLIKE: 0, LIKE: 1}
    kg_triples = load_kg_triples(e_idx_map, r_idx_map)

    n_users = data_loader.n_users
    n_entities = len(e_idx_map)
    n_relations = len(r_idx_map)

    for fold_index, (train_ratings, test_ratings) in enumerate(data_loader.generate_folds()):
        total_n_entities = n_users + n_entities
        total_n_entities_no_users = total_n_entities - n_users

        user_indices = list(range(total_n_entities_no_users, total_n_entities))
        all_indices = list(range(total_n_entities))

        train_ratings = unify_user_entity_indices(train_ratings, starting_index=total_n_entities_no_users)
        test_ratings = unify_user_entity_indices(test_ratings, starting_index=total_n_entities_no_users)

        model = TransE(total_n_entities, n_relations=n_relations, margin=1.0, k=50)
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        for epoch in range(n_epochs):
            pos_train_triples = convert_hrt_triples(train_ratings, r_idx_map)
            pos_train_triples += kg_triples  # Comment out to not include KG triples in training

            random.shuffle(pos_train_triples)
            neg_train_triples = corrupt_hrt_triples(pos_train_triples, all_indices)

            pos_test_triples = convert_hrt_triples(test_ratings, r_idx_map)
            neg_test_triples = corrupt_hrt_triples(pos_test_triples, all_indices)

            print(f'Epoch {epoch}')
            if epoch % 10 == 0:
                evaluate_loss(model, pos_train_triples, pre_string='Train')
                evaluate_rank(model, pos_train_triples, pre_string='Train')
                evaluate_loss(model, pos_test_triples, pre_string='Test ')
                evaluate_rank(model, pos_test_triples, pre_string='Test ')

            for (p_h, p_r, p_t), (n_h, n_r, n_t) in zip(batchify(pos_train_triples), batchify(neg_train_triples)):
                model.train()

                p_ht_distance = model(p_h, p_r, p_t)
                n_ht_distance = model(n_h, n_r, n_t)

                loss = tt.relu(model.margin + p_ht_distance - n_ht_distance).sum()
                loss.backward()
                optimizer.step()

                model.zero_grad()

