from data_loading.generic_data_loader import DataLoader
import random


class KFoldLoader(DataLoader):
    def __init__(self):
        super(DataLoader, self).__init__()
        self.buckets = []

    def fill_buckets(self, n):
        pass


if __name__ == '__main__':
    data_loader = KFoldLoader.load_from('./mindreader')
    print(data_loader.info())
