from analysis.performance import get_model_choice
import numpy as np
import json
import os


BASE_DIR = '../results'

if __name__ == '__main__':
    # Method that generates a full LaTeX row
    # for some model

    model = get_model_choice()
    runs = [r for r in os.listdir(os.path.join(BASE_DIR, model)) if not r == 'training']
    qualifiers = [q for q in os.listdir(os.path.join(BASE_DIR, model, '1'))]

    print(f'At which k do you want Hit@k and DCG@k?')
    at_k = input('Enter k: ')
    at_k = int(at_k)

    hit_results = {
        '4-4': [],
        '3-4': [],
        '2-4': [],
        '1-4': []
    }

    hit_stds = {
        '4-4': [],
        '3-4': [],
        '2-4': [],
        '1-4': []
    }

    dcg_results = {
        '4-4': [],
        '3-4': [],
        '2-4': [],
        '1-4': []
    }

    dcg_stds = {
        '4-4': [],
        '3-4': [],
        '2-4': [],
        '1-4': []
    }

    for run in runs:
        for qualifier in qualifiers:
            file = os.path.join(BASE_DIR, model, run, qualifier)
            if 'WKG' in file:
                continue
            hit_lst = [ls for k, ls in hit_results.items() if k in file][0]
            dcg_lst = [ls for k, ls in dcg_results.items() if k in file][0]

            with open(file) as fp:
                metrics = json.load(fp)
                hit_lst.append(metrics['hits_at_k'][at_k-1])
                dcg_lst.append(metrics['dcgs_at_k'][at_k-1])

    for k, v in hit_results.items():
        hit_results[k] = np.mean(v)
        hit_stds[k] = np.std(v)
    for k, v in dcg_results.items():
        dcg_results[k] = np.mean(v)
        dcg_stds[k] = np.std(v)

    hit_latex = f'{model.upper()}            '
    for k, v in hit_results.items():
        std = hit_stds[k]
        hit_latex += f'& ${v : .3f} \\pm {std : .3f}$'

    hit_latex += '\\\\\\hline'

    dcg_latex = f'{model.upper()}            '
    for k, v in dcg_results.items():
        std = dcg_stds[k]
        dcg_latex += f'& ${v : .3f} \\pm {std : .3f}$'

    dcg_latex += '\\\\\\hline'

    print(f'{model.upper()}@{at_k}:')
    print(f'HIT: {hit_latex}')
    print(f'DCG: {dcg_latex}')