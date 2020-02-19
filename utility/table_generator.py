import json
import os
from collections import defaultdict
from typing import List

from loguru import logger
from scipy.stats import ttest_rel

alpha = 0.05

pretty_map = {
    'ndcg': 'NDCG',
    'hr': 'HR',
    'pr-collab': 'PPR-COLLAB',
    'pr-joint': 'PPR-JOINT',
    'pr-kg': 'PPR-KG',
    'item-knn': 'Item kNN',
    'user-knn': 'User kNN',
    'transe': 'TransE',
    'transe-kg': 'TransE-KG',
    'transh': 'TransH',
    'transh-kg': 'TransH-KG',
    'random': 'Random',
    'top-pop': 'TopPop',
    'svd': 'SVD',
    'bpr': 'BPR',
    'wtp-all_entities': 'All entities',
    'wtp-all_movies': 'All movies',
    'ntp-all_entities': 'All entities',
    'ntp-all_movies': 'All movies',
    'wtp-substituting-4-4': '4/4',
    'wtp-substituting-3-4': '3/4',
    'wtp-substituting-2-4': '2/4',
    'wtp-substituting-1-4': '1/4',
    'ntp-substituting-4-4': '4/4',
    'ntp-substituting-3-4': '3/4',
    'ntp-substituting-2-4': '2/4',
    'ntp-substituting-1-4': '1/4',
    'wtp-substituting-1-0': '1/4 (no DEs)',
    'mf': 'MF',
    'joint-mf': 'Joint-MF'
}


def line():
    return '\\\\\\hline'


def pretty(item):
    return pretty_map.get(item, item)


def get_p_val(results_base, model, metric, k_value, a, b):
    # Get intersecting splits
    metrics = list()
    for experiment in [a, b]:
        model_splits_path = os.path.join(os.path.join(results_base, experiment), model)
        model_splits = sorted([file for file in os.listdir(model_splits_path) if file != 'params.json'])

        model_metrics = list()
        for split in model_splits:
            split_path = os.path.join(model_splits_path, split)

            with open(split_path, 'r') as fp:
                model_metrics.append(json.load(fp)[metric][k_value])

        metrics.append(model_metrics)

    # Cutoff to minimum length
    min_length = len(min(metrics, key=len))
    metrics = [measure[:min_length] for measure in metrics]

    # t-test
    return ttest_rel(*metrics)[1]


def generate_table(results_base, experiments: List[str], metric='hr', test=None, k_value='10'):
    n_columns = 1 + len(experiments)

    table = """\\begin{table*}[ht!]\n\t\\centering\n"""

    # For each experiment, get summary files
    experiment_summary = defaultdict(dict)
    models = set()
    for experiment in experiments:
        experiment_results = dict()
        results_path = os.path.join(results_base, experiment)

        for file in os.listdir(results_path):
            if not file.startswith('summary') or not file.endswith('.json'):
                continue

            with open(os.path.join(results_path, file), 'r') as fp:
                summary = json.load(fp)

                for key, values in summary.items():
                    if key in experiment_results:
                        logger.warning(f'Duplicate model {key} for {experiment}')

                        continue

                    experiment_results[key] = values
                    models.add(key)

        experiment_summary[experiment] = experiment_results

        if not experiment_results:
            logger.error(f'No summaries for {experiment}')

            return

    models = sorted(models, key=pretty)

    # Add header
    column_layout = '|'.join('l' if not idx else 'c' for idx in range(n_columns))
    table += "\t\\begin{tabular}{" + column_layout + "}\n"

    # Add first row with model names
    table += "\t\t\\hline\n"
    columns = [f'& {pretty(experiment)}' for experiment in experiments]

    table += "\t\t\\multicolumn{1}{c|}{Models} " + ' '.join(columns) + "\n"
    table += "\t\t" + line() + "\n"

    # Get model-major results
    model_results = defaultdict(dict)
    highest_experiment_mean = defaultdict(float)
    for experiment, summary in experiment_summary.items():
        for model in models:
            if model not in summary:
                continue

            mean = round(summary[model][metric][k_value]['mean'], 2)
            model_results[model][experiment] = {
                'mean': mean,
                'std': summary[model][metric][k_value]['std']
            }

            if mean > highest_experiment_mean[experiment]:
                highest_experiment_mean[experiment] = mean

    for idx, model in enumerate(models):
        result_list = []

        for experiment in experiments:
            if experiment not in model_results[model]:
                result_list.append(' & N/A')

                continue

            mean = model_results[model][experiment]['mean']
            std = model_results[model][experiment]['std']

            base = f'{mean:.2f} \\pm {std:.2f}'
            significant = ''
            if test and experiment != test:
                significant = '^*' if get_p_val(results_base, model, metric, k_value, test, experiment) < alpha else ''
            if mean >= highest_experiment_mean[experiment]:
                result_list.append(" & $\\mathbf{" + base + significant + "}$")
            else:
                result_list.append(" & $" + base + significant + "$")

        table += "\t\t"

        # If not the first model row, add new line
        if idx:
            table += "\\\\"
        table += " " + pretty(model) + ''.join(result_list) + "\n"

    # Add footer
    table += "\t\t" + line() + "\n"
    table += "\t\\end{tabular}\n"
    table += "\t\\caption{" + metric.upper() + "@" + k_value + ".}\n"
    table += "\\end{table*}"

    return table
