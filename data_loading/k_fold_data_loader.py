from data_loading.generic_data_loader import DataLoader
import random


class KFoldLoader(DataLoader):
    def __init__(self):
        super(DataLoader, self).__init__()
        self.buckets = []
        self.n = None

    def fill_buckets(self, n):
        self.n = n
        self.buckets = [[] for _ in range(self.n)]

        train_buckets = self.buckets[:-1]
        test_bucket = self.buckets[-1]

        entities_in_train_buckets = set()
        entities_in_test_bucket = set()

        for user in self


if __name__ == '__main__':
    data_loader = KFoldLoader.load_from('./mindreader')
    print(data_loader.info())
