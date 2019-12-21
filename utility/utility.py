from scipy.sparse import csr_matrix
import itertools as it


def csr(train, only_positive=False):
    all_ratings = []
    users = []
    items = []

    for user, ratings in train:
        for rating in ratings:
            if only_positive:
                all_ratings.append(1)
            else:
                all_ratings.append(2 if rating.rating == 1 else 1)

            users.append(user)
            items.append(rating.e_idx)

    return csr_matrix((all_ratings, (users, items)))


def get_combinations(parameters):
    keys, values = zip(*parameters.items())
    return [dict(zip(keys, v)) for v in it.product(*values)]
