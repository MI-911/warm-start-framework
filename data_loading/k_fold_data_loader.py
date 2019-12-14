from data_loading.generic_data_loader import DataLoader, User, Rating
import random


class Bucket:
    def __init__(self, idx):
        self.idx = idx
        self.ratings = []
        self.entities_in_bucket = set()
        self.users_in_bucket = set()

    def add(self, rating):
        self.ratings.append(rating)
        self.users_in_bucket.add(rating.u_idx)
        self.entities_in_bucket.add(rating.e_idx)

    def size(self):
        return len(self.ratings)


class KFoldLoader(DataLoader):
    def __init__(self, args):
        super(KFoldLoader, self).__init__(*args)
        self.buckets = []
        self.n = None

    def fill_buckets(self, n):
        self.n = n
        self.buckets = [Bucket(i) for i in range(self.n)]

        for user in self._get_user_major_ratings().values():
            for r in user.movie_ratings:
                self._get_best_bucket(r).add(r)
            for r in user.descriptive_entity_ratings:
                self._get_best_bucket(r).add(r)

    def generate_folds(self):
        bucket_set = set(self.buckets)
        bucket_dict = {i: b for i, b in enumerate(self.buckets)}
        for i in range(self.n):
            test_bucket = bucket_dict[i]
            train_buckets = bucket_set - set([test_bucket])

            train_ratings = []
            test_ratings = test_bucket.ratings

            for bucket in train_buckets:
                train_ratings += bucket.ratings

            yield train_ratings, test_ratings

    @staticmethod
    def load_from(path, filter_unknowns=True):
        return KFoldLoader(DataLoader._load_from(path, filter_unknowns))

    def info(self):
        return f'''
            KFoldLoader info
            -----------------------------------------------------
            n: {self.n}
            n_ratings_in_buckets:  {[b.size() for b in self.buckets]}
            n_users_in_buckets:    {[len(b.users_in_bucket) for b in self.buckets]}
            n_entities_in_buckets: {[len(b.entities_in_bucket) for b in self.buckets]}
        '''

    def _get_best_bucket(self, rating):
        # First, check if any bucket needs this entity
        for bucket in self.buckets:
            if rating.e_idx not in bucket.entities_in_bucket:
                return bucket

        # Then, if any bucket needs this user
        for bucket in self.buckets:
            if rating.u_idx not in bucket.users_in_bucket:
                return bucket

        # Otherwise, fill the smallest bucket
        smallest_bucket = self.buckets[0]
        for bucket in self.buckets:
            if bucket.size() < smallest_bucket.size():
                smallest_bucket = bucket

        return smallest_bucket

    def _get_user_major_ratings(self):
        user_ratings_map = {}
        for r in self.ratings:
            if r.u_idx not in user_ratings_map:
                user_ratings_map[r.u_idx] = User(r.u_idx)
            u = user_ratings_map[r.u_idx]
            if r.is_movie_rating:
                u.movie_ratings.append(r)
            else:
                u.descriptive_entity_ratings.append(r)

        return user_ratings_map


if __name__ == '__main__':
    data_loader = KFoldLoader.load_from('./mindreader')
    data_loader.fill_buckets(n=5)
    print(data_loader.info())

    for train_ratings, test_ratings in data_loader.generate_folds():
        print(len(train_ratings), len(test_ratings))
