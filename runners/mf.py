import numpy as np
import torch as tt
import torch.optim as optim
from models.mf import MF
from metrics.metrics import average_precision
from data_loading.k_fold_data_loader import KFoldLoader, User, Rating


def unpack(rating):
    return rating.u_idx, rating.e_idx, float(rating.rating), rating.is_movie_rating


def column_batch(triple_batch):
    return list(zip(*triple_batch))


def batchify(lst, n):
    for i in range(0, len(lst), n):
        yield column_batch(lst[i:i + n])


def get_user_ratings_map(ratings):
    u_r_map = {}
    for u, e, r, is_movie in ratings:
        if u not in u_r_map:
            u_r_map[u] = {}
        user = u_r_map[u]

        if is_movie:
            user[e] = r

    return u_r_map


def eval():
    data_loader = KFoldLoader.load_from('../data_loading/mindreader',
                                        filter_unknowns=True,
                                        movies_only=False,
                                        min_num_entity_ratings=5)
    data_loader.fill_buckets(5)
    n_epochs = 10
    batch_size = 64

    result_strings = []

    for fold_index, (train_ratings, test_ratings) in enumerate(data_loader.generate_folds()):
        train_ratings = list(map(unpack, train_ratings))
        test_ratings = list(map(unpack, test_ratings))

        k = 25
        model = MF(data_loader.n_users, data_loader.n_movies + data_loader.n_descriptive_entities, k)
        optimizer = optim.SGD(model.parameters(), lr=0.003)
        loss_fn = tt.nn.MSELoss()

        # Evaluation
        at_n = 20
        average_precisions = []

        for epoch in range(n_epochs):
            print(f'Evaluating at epoch {epoch}...')

            with tt.no_grad():
                model.eval()
                train_batch = column_batch(train_ratings)
                test_batch = column_batch(test_ratings)

                train_loss = loss_fn(model(*train_batch[:-2]), tt.tensor(train_batch[-2]).to(model.device))
                test_loss = loss_fn(model(*test_batch[:-2]), tt.tensor(test_batch[-2]).to(model.device))

                print(f'Epoch {epoch}:')
                print(f'    Train MSE: {train_loss}')
                print(f'    Test MSE:  {test_loss}')

            for user_batch, item_batch, rating_batch, _ in batchify(train_ratings, n=batch_size):
                model.train()  # Enable gradients

                pred_batch = model(user_batch, item_batch)
                rating_batch = tt.tensor(rating_batch).to(model.device)
                loss = loss_fn(pred_batch, rating_batch)

                loss.backward()
                optimizer.step()

                model.zero_grad()  # Zero the gradients for the next iteration

        with tt.no_grad():
            model.eval()  # Disable gradients and dropout
            u_r_map = get_user_ratings_map(test_ratings)
            n_users = len(u_r_map)
            u_count = 0
            for u, ratings in u_r_map.items():
                predictions = model.predict(u, item_indices=data_loader.movie_indices)
                sorted_predictions = list(sorted(enumerate(predictions), key=lambda x: x[1], reverse=True))[:at_n]
                ranked_relevancy_list = np.zeros(at_n)
                for i, (p, _) in enumerate(sorted_predictions):
                    if p in ratings:
                        if ratings[p] == 1:
                            ranked_relevancy_list[i] = 1

                average_precisions.append(average_precision(ranked_relevancy_list))

                if u_count % 50 == 0:
                    print(f'Evaluating ({(u_count / n_users) * 100 : 2.2f}%)')

                u_count += 1

            mean_average_precision = np.mean(average_precisions)

            result_strings.append(f'Fold {fold_index}: MAP@{at_n} = {mean_average_precision}\n')

        with open(f'results.txt', 'w') as fp:
            fp.writelines(result_strings)


if __name__ == '__main__':
    eval()