from data_loading.generic_data_loader import DataLoader
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
    def __init__(self):
        super(DataLoader, self).__init__()
        self.buckets = []
        self.n = None

    def fill_buckets(self, n):
        self.n = n
        self.buckets = [Bucket(i) for i in range(self.n)]

        for user in self.user_ratings.values:
            # First priority is to make sure that all buckets
            # contain the same entities at least once.
            # After that, the second priority is to make sure
            # that all buckets have the same number of ratings.
            pass

    def _get_best_bucket(self, rating):
        # First, check if any bucket needs this entity
        for bucket in self.buckets:
            if rating.e_idx not in bucket.entities_in_bucket:
                return bucket

        # Otherwise, fill the smallest bucket
        smallest_bucket = self.buckets[0]
        for bucket in self.buckets:
            if bucket.size() < smallest_bucket.size():
                smallest_bucket = bucket

        return smallest_bucket


if __name__ == '__main__':
    data_loader = KFoldLoader.load_from('./mindreader')
    print(data_loader.info())
