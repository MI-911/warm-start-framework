import json
import numpy as np
import os

BASE_DIR = '../results'


def get_model_choice():
    models = os.listdir(BASE_DIR)

    print(f'Which model do you want results for?')
    for model in models:
        print(f'    - {model}')

    choice = None
    while choice not in models:
        if choice is not None:
            print(f'Model "{choice}" not found, try again.')
        choice = input('Enter model name: ')

    return choice


def get_model_qualifier(model):
    dir = os.path.join(BASE_DIR, model)
    runs = [run for run in os.listdir(dir) if not run == 'training']

    assert len(runs) > 0, f'Found no runs for {model}'

    print(f'Which model qualifier do you want results for?')
    qualifiers = os.listdir(os.path.join(dir, runs[0]))
    qualifiers = [q.split(".")[0] for q in qualifiers]
    for q in qualifiers:
        print(f'    - {q}')

    choice = None
    while choice not in qualifiers:
        if choice is not None:
            print(f'Qualifier "{choice}" not found, try again.')
        choice = input('Enter the model qualifier: ')

    return choice + '.json'


def to_latex(value, std):
    return f'${value : .3f} \\pm {std :.3f}$'


def load_model_results(model):
    dir = os.path.join(BASE_DIR, model)
    runs = [run for run in os.listdir(dir) if not run == 'training']
    qualifier = get_model_qualifier(model)

    print(f'Loading results for {model} over {len(runs)} runs...')

    hits_at_k = [[] for i in range(50)]
    dcgs_at_k = [[] for i in range(50)]

    for run in runs:
        with open(os.path.join(dir, run, qualifier)) as fp:
            results = json.load(fp)
            for k in range(50):
                hits_at_k[k].append(results['hits_at_k'][k])
                dcgs_at_k[k].append(results['dcgs_at_k'][k])

    print(f'At which k(s) do you want Hit@k and DCG@k?')
    ks = input('Enter k (separate with space if more than one): ')
    ks = map(int, [k for k in ks.split()])
    print(f'Performance results for {qualifier}:')
    for k in ks:
        k -= 1
        hit_at_k = np.mean(hits_at_k[k])
        hit_std_dev = np.std(hits_at_k[k])
        dcg_at_k = np.mean(dcgs_at_k[k])
        dcg_std_dev = np.std(dcgs_at_k[k])
        print(f'    ---------------------')
        print(f'    Hit@{k + 1}: {hit_at_k : .3f} (+- {hit_std_dev : .3f})  - {to_latex(hit_at_k, hit_std_dev)}')
        print(f'    DCG@{k + 1}: {dcg_at_k : .3f} (+- {dcg_std_dev : .3f})  - {to_latex(dcg_at_k, dcg_std_dev)}')


if __name__ == '__main__':
    model = get_model_choice()
    load_model_results(model)
