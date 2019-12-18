import json

if __name__ == '__main__':
    with open('../data_loading/mindreader/ratings_clean.json') as rp:
        ratings = json.load(rp)
    with open('../data_loading/mindreader/entities_clean.json') as ep:
        entities = json.load(ep)

    # Filter unknowns?
    ratings = [(u, e, r) for u, e, r in ratings if not r == 0]

    label_map = {}
    for uri, name, labels in entities:
        if uri not in label_map:
            label_map[uri] = labels.split('|')

    u_map = {}
    e_map = {}
    m_map = {}

    for user, entity, rating in ratings:
        if user not in u_map:
            u_map[user] = 0
        u_map[user] += 1

        if 'Movie' in label_map[entity]:
            if entity not in m_map:
                m_map[entity] = 0
            m_map[entity] += 1
        else:
            if entity not in e_map:
                e_map[entity] = 0
            e_map[entity] += 1

    movie_ratings = [(u, e, r) for u, e, r in ratings if 'Movie' in label_map[e]]

    print(f'Number of ratings: {len(ratings)}')
    print(f'Number of movie ratings: {len(movie_ratings)}')
    print(f'Number of users: {len(u_map)}')
    print(f'Number of entities: {len(e_map)}')
    print(f'Number of movies: {len(m_map)}')
    print(f'Density (all): {(len(ratings) / (len(u_map) * (len(m_map) + len(e_map)))) * 100}')
    print(f'Density (movies): {(len(movie_ratings) / (len(u_map) * (len(m_map)))) * 100}')

