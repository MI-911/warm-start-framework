from scipy.sparse import csr_matrix


def csr(train, only_positive=False):
    all_ratings = []
    users = []
    items = []

    for user, ratings in train:
        for rating in ratings:
            if only_positive and rating.rating != 1:
                continue

            all_ratings.append(1 if rating.rating == 1 else 0)
            users.append(user)
            items.append(rating.e_idx)

    return csr_matrix((all_ratings, (users, items)))
