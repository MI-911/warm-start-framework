import pandas as pd

"""
Loads data almost unmanipulated.
"""


def load_csv_data_ratings(path, select_columns=None):
    """
    Loads data from a csv file and returns selected columns as a list of lists.
    :param path: Path to file.
    :param select_columns: A list of selected column names.
    :return: A list, with the selected columns as lists.
    """
    if select_columns is None:
        select_columns = ['userId', 'uri', 'isItem', 'sentiment']

    df = pd.read_csv(path)
    df = df[select_columns]

    return [val.to_list() for _, val in df.iterrows()]


if __name__ == '__main__':
    a = load_csv_data_ratings('mindreader/ratings.csv')
    print(a[:10])
