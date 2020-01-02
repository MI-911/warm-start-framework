import json
import os
from collections import defaultdict
from typing import List

from loguru import logger

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
    'mf': 'MF',
    'joint-mf': 'Joint-MF'
}


def line():
    return '\\\\\\hline'


def pretty(item):
    return pretty_map.get(item, item)


def generate_table(results_base, experiments: List[str], metric='hr', k_values=None):
    n_columns = 1 + len(experiments)
    if not k_values:
        k_values = ['5', '10']
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
    column_layout = '|'.join('c' for _ in range(n_columns))
    table += "\t\\begin{tabular}{|" + column_layout + "|}\n"

    # Add first row with model names
    table += "\t\t\\hline\n"
    model_selection = [f'& {pretty(experiment)}' for experiment in experiments]
    table += "\t\t\\textbf{Model} " + ' '.join(model_selection) + "\n"

    # Add row for each k-value
    for k_value in k_values:
        table += "\t\t" + line() + "\\multicolumn{" + str(n_columns) + "}{|c|}{\\textbf{" + pretty(metric) + "@" + str(k_value) + "}}\n"

        # Get model-major results
        model_results = defaultdict(dict)
        highest_experiment_mean = defaultdict(float)
        for experiment, summary in experiment_summary.items():
            for model in models:
                if model not in summary:
                    continue

                mean = summary[model][metric][k_value]['mean']
                model_results[model][experiment] = {
                    'mean': mean,
                    'std': summary[model][metric][k_value]['std']
                }

                if mean > highest_experiment_mean[experiment]:
                    highest_experiment_mean[experiment] = mean

        for model in models:
            result_list = []

            for experiment in experiments:
                if experiment not in model_results[model]:
                    result_list.append(' & N/A')

                    continue

                mean = model_results[model][experiment]['mean']
                std = model_results[model][experiment]['std']

                base = f'{mean:.2f} \pm {std:.2f}'
                if mean >= highest_experiment_mean[experiment]:
                    result_list.append(" & $\\mathbf{" + base + "}$")
                else:
                    result_list.append(" & $" + base + "$")

            table += "\t\t" + line() + " " + pretty(model) + ''.join(result_list) + "\n"

    # Add footer
    table += "\t\t" + line() + "\n"
    table += "\t\\end{tabular}\n"
    table += "\\end{table*}"

    return table
