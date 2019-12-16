import numpy as np


def average_precision(ranked_relevancy_list):
    """
    Calculates the average precision AP@k. In this setting, k is the length of
    ranked_relevancy_list.
    :param ranked_relevancy_list: A one-hot numpy list mapping the recommendations to
    1 if the recommendation was relevant, otherwise 0.
    :return: AP@k
    """

    if len(ranked_relevancy_list) == 0:
        a_p = 0.0
    else:
        p_at_k = ranked_relevancy_list * np.cumsum(ranked_relevancy_list, dtype=np.float32) / (1 + np.arange(ranked_relevancy_list.shape[0]))
        a_p = np.sum(p_at_k) / ranked_relevancy_list.shape[0]

    assert 0 <= a_p <= 1, a_p
    return a_p


if __name__ == '__main__':
    is_relevant_1 = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    is_relevant_2 = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    ap_1 = average_precision(is_relevant_1)
    ap_2 = average_precision(is_relevant_2)

    assert(ap_1 > ap_2)