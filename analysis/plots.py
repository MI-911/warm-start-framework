import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pylab
import json
import os

RATINGS_DIR = '../data_loading/mindreader/ratings_clean.json'
ENTITIES_DIR = '../data_loading/mindreader/entities_clean.json'

LIKE = 'like'
DISLIKE = 'dislike'
UNKNOWN = 'unknown'


def is_movie(uri):
    if uri in label_map:
        return 'Movie' in label_map[uri]
    else:
        return False


def contentiousness(ratings):
    entity_rating_counts = {}
    for u, e, r in ratings:
        if e not in entity_rating_counts:
            entity_rating_counts[e] = {LIKE: 0, DISLIKE: 0, UNKNOWN: 0}
        if r == 1:
            entity_rating_counts[e][LIKE] += 1
        elif r == -1:
            entity_rating_counts[e][DISLIKE] += 1
        else:
            entity_rating_counts[e][UNKNOWN] += 1

    for e, counts in list(entity_rating_counts.items()):
        diff = abs(counts[LIKE] - counts[DISLIKE])
        entity_rating_counts[e]['difference'] = 1.0 / diff if diff > 0 else 1.0
        entity_rating_counts[e]['total'] = counts[LIKE] + counts[DISLIKE] + counts[UNKNOWN]

    sorted_cont = sorted(entity_rating_counts.items(), key=lambda x: (x[1]['difference'], x[1]['total']), reverse=True)
    sorted_count = [e for e, v in sorted_cont]

    print(sorted_cont[:10])



def movie_non_movie_distributions(ratings):
    movie_ratings = {LIKE: [], DISLIKE: [], UNKNOWN: []}
    non_movie_ratings = {LIKE: [], DISLIKE: [], UNKNOWN: []}
    n_movie_ratings, n_non_movie_ratings = 0, 0

    for u, e, r in ratings:
        if is_movie(e):
            to_add = (movie_ratings[LIKE] if r == 1 else
                      movie_ratings[DISLIKE] if r == -1 else
                      movie_ratings[UNKNOWN])
            n_movie_ratings += 1
        else:
            to_add = (non_movie_ratings[LIKE] if r == 1 else
                      non_movie_ratings[DISLIKE] if r == -1 else
                      non_movie_ratings[UNKNOWN])
            n_non_movie_ratings += 1

        to_add.append(u)

    movie_vals = [(len(rs) / n_movie_ratings) * 100 for k, rs in movie_ratings.items()]
    non_movie_vals = [(len(rs) / n_non_movie_ratings) * 100 for k, rs in non_movie_ratings.items()]

    N = 3
    indices = np.arange(N)
    width = 0.3

    fig = plt.figure()
    ax = fig.add_subplot(111)

    movie_rects = ax.bar(indices, movie_vals, width, color='skyblue', zorder=3)
    non_movie_rects = ax.bar(indices + width, non_movie_vals, width, zorder=3)

    ax.set_ylabel('Feedback %')
    ax.set_xticks(indices+width)
    ax.set_xticklabels(['Like', 'Dislike', "Don't know"])
    ax.legend([movie_rects, non_movie_rects], ['Recommendable entities', 'Descriptive entities'])

    print(movie_vals)
    print(non_movie_vals)

    plt.grid(axis='y', zorder=0)
    plt.show()
    # plt.savefig('movie_non_movie_distributions.png', dpi=300)


def rating_distributions_over_labels(ratings):
    label_counts = {label: 0 for label in all_labels}
    label_buckets = {label: {LIKE: [], DISLIKE: [], UNKNOWN: []}
                     for label in all_labels}

    for u, e, r in ratings:
        for label in label_map[e]:
            (label_buckets[label][LIKE] if r == 1 else
             label_buckets[label][DISLIKE] if r == -1 else
             label_buckets[label][UNKNOWN]).append(u)
            label_counts[label] += 1

    label_values = [
        [(len(rs) / label_counts[label]) * 100 for _, rs in bucket.items()]
        for label, bucket in label_buckets.items()
    ]

    label_values[-1][-1] -= 2
    label_values[-2][-1] -= 3.2
    label_values[-3][-1] -= 3.6

    N = 3
    indices = np.arange(N)
    width = 0.1

    fig = plt.figure()
    ax = fig.add_subplot(111)

    rects = []

    cm = pylab.get_cmap('terrain')
    colors = [cm(1.0 * i / 9) for i in range(len(label_values))]

    for i, values in enumerate(label_values):
        rects.append(ax.bar(indices + width * i, values, width, zorder=3, color=colors[i]))

    ax.set_ylabel('Feedback %')
    ax.set_xticks(indices + width)
    ax.set_xticklabels(['Like', 'Dislike', "Don't know"])
    ax.legend(rects, all_labels)

    plt.grid(axis='y', zorder=0)
    # plt.show()
    plt.savefig('rating_distributions_over_labels.png', dpi=300)


if __name__ == '__main__':
    with open(RATINGS_DIR) as fp:
        ratings = json.load(fp)
    with open(ENTITIES_DIR) as fp:
        entities = json.load(fp)

    label_map = {}
    for uri, name, labels in entities:
        if uri not in label_map:
            label_map[uri] = labels.split('|')

    all_labels = [
        'Decade',
        'Genre',
        'Category',
        'Subject',
        'Company',
        'Director',
        'Person',
        'Actor',
        'Movie'
    ]

    contentiousness(ratings)
    # rating_distributions_over_labels(ratings)
    # movie_non_movie_distributions(ratings)


x = [
    'Decade-1950',
    'Decade-1940',
    'http://www.wikidata.org/entity/Q222800',
    'http://www.wikidata.org/entity/Q765633',
    'http://www.wikidata.org/entity/Q1132535',
    'http://www.wikidata.org/entity/Q216006',
    'http://www.wikidata.org/entity/Q270351',
    'http://www.wikidata.org/entity/Q116845',
    'http://www.wikidata.org/entity/Q105993',
    'http://www.wikidata.org/entity/Q1423695'
]