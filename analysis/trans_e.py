import os
import json
import matplotlib.pyplot as plt
import numpy as np


def average_results(result_objects):
    n_recordings = len(result_objects[0]['train_loss'])

    train_loss = np.zeros(n_recordings)
    validation_hitrates = np.zeros(n_recordings)
    validation_dcgs = np.zeros(n_recordings)

    for obj in result_objects:
        for i in range(n_recordings):
            train_loss[i] += obj['train_loss'][i]
            validation_hitrates[i] += obj['validation']['hit_ratio'][i]
            validation_dcgs[i] += obj['validation']['dcg'][i]

    train_loss /= len(result_objects)
    validation_hitrates /= len(result_objects)
    validation_dcgs /= len(result_objects)

    print(f'Returning hitrates: {validation_hitrates}')

    return train_loss, validation_hitrates, validation_dcgs


def load_result_objects(path):
    file_directories = os.listdir(path)
    result_objects = {}
    for file_directory in file_directories:
        files = os.listdir(os.path.join(path, file_directory))
        result_objects[file_directory] = []
        for file in files:
            with open(os.path.join(path, file_directory, file)) as fp:
                result_objects[file_directory].append(json.load(fp))

    return result_objects


def plot(with_kg_triples=False, standard_corruption=True):
    result_objects = load_result_objects('../old/removing_movies_only')

    # First without any KG triples
    for group, objects in list(result_objects.items()):
        result_objects[group] = (
            [obj for obj in objects if obj['with_kg_triples']] if with_kg_triples else
            [obj for obj in objects if not obj['with_kg_triples']])
        result_objects[group] = (
            [obj for obj in objects if obj['standard_corruption']] if standard_corruption else
            [obj for obj in objects if not obj['standard_corruption']])

    grouped_models = {}
    for group, objects in result_objects.items():
        for obj in objects:
            if obj['movie_to_entity_ratio'] not in grouped_models:
                grouped_models[obj['movie_to_entity_ratio']] = []
            grouped_models[obj['movie_to_entity_ratio']].append(obj)

    avg_models = {
        movie_entity_ratio: average_results(models)
        for movie_entity_ratio, models
        in grouped_models.items()
    }

    # ----------- LOSS ----------
    for movie_entity_ratio, (train_loss, val_hitrate, test_hitrate) in avg_models.items():
        plt.plot(train_loss, label=f'{movie_entity_ratio} movie ratings')
    plt.title(f'TransE loss at varying dataset compositions {"(with KG triples)" if with_kg_triples else ("(no KG triples)")}')
    plt.ylabel('Distance between e(h) + e(r) and e(t) for recommendable items t')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()
    plt.show()

    # ------------- VAL HITRATE ----------
    for movie_entity_ratio, (train_loss, val_hitrate, test_hitrate) in avg_models.items():
        plt.plot(val_hitrate, label=f'{movie_entity_ratio} movie ratings')
    plt.title(f'Hit@10 at varying dataset compositions {"(with KG triples)" if with_kg_triples else ("(no KG triples)")}')
    plt.ylabel('Hit@10')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()
    plt.show()

    # ------------- VAL DCG ----------
    for movie_entity_ratio, (train_loss, val_hitrate, val_dcg) in avg_models.items():
        plt.plot(val_dcg, label=f'{movie_entity_ratio} movie ratings')
    plt.title(
        f'DCG@10 at varying dataset compositions {"(with KG triples)" if with_kg_triples else ("(no KG triples)")}')
    plt.ylabel('DCG@10')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    plot(with_kg_triples=False, standard_corruption=True)
    plot(with_kg_triples=True, standard_corruption=True)
    plot(with_kg_triples=True, standard_corruption=False)
