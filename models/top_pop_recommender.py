from collections import defaultdict

from models.base_recommender import RecommenderBase


def get_item_count(train_list):
    movie_count = defaultdict(int)

    for user, ratings in train_list:
        for rating in ratings:
            movie_count[rating.e_idx] += 1

    return movie_count


class TopPopRecommender(RecommenderBase):
    def __init__(self):
        super().__init__()
        self.item_count = None

    def predict(self, user, items):
        # Just give items their number of ratings, we do not care about the user
        return {item: self.item_count[item] for item in items}

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        self.item_count = get_item_count(training)
