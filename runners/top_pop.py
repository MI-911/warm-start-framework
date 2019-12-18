from collections import defaultdict

from data_loading.loo_data_loader import DesignatedDataLoader


def run():
    pass


def get_top_pop(train_list):
    movie_count = defaultdict(int)

    for user, ratings in train_list:
        for rating in ratings:
            movie_count[rating.e_idx] += 1

    return sorted(list(movie_count.items()), key=lambda pair: pair[1])[::1]


def evaluate():
    pass


if __name__ == '__main__':
    data_loader = DesignatedDataLoader.load_from(
        path='../data_loading/mindreader',
        movies_only=False,
        min_num_entity_ratings=1,
        filter_unknowns=True
    )

    data_loader.random_seed = 42

    print(data_loader.info())

    train, validation, test = data_loader.make(
        movie_to_entity_ratio=1,
        n_negative_samples=99
    )


    run()
