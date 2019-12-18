

class RecommenderBase:
    def __init__(self, model):
        self.model = model

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        """
        Fits the model to the training data.
        :param training: List<int, List<Rating>> - List of (user_index, ratings) pairs
        :param validation: List<int, (int, List<int>)> - List of (user_index, (pos_index, neg_indices)) pairs
        :param max_iterations: (Optional) int - To ensure that the fitting process will stop eventually.
        :param verbose: (Optional) boolean - Controls whether statistics are written to stdout during training.
        :param save_to: (Optional) string - The full path for the file in which to save during-training metrics.
        :return: None
        """
        raise NotImplementedError

    def predict(self, user, items):
        """
        Predicts a score for all items given a user.
        :param user: int - user index
        :param items: List<int> - list of item indices
        :return: Dictionary<int, float> - A mapping from item indices to their score.
        """
        raise NotImplementedError

