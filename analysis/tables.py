import numpy as np
import json
import os


BASE_DIR = '../results'


class Table:
    def __init__(self, experiments, summaries):
        self.experiments = experiments
        self.summaries = summaries
        self.n_columns = len(experiments)

        self.models = []
        for summary in summaries:
            self.models += summary.keys()
        self.models = set(self.models)

    def render_title(self):
        return f'''
            \\begin{{table}}[ht]
            \\centering
            \\begin{{tabular}}{{|c|c|c|}}
            \\hline 
            \\textbf{{Model}} & \\textbf{{Movies only}} & \\textbf{{All ratings}}   \\\\\\hline 
        '''


def load_summary(path):
    if 'summary.json' not in path:
        path = os.path.join(path, 'summary.json')
    if not os.path.exists(path):
        raise IOError(f'The summary file "{path}" does not exist.')
    with open(path) as fp:
        return json.load(fp)


def prompt_experiments():
    all_experiments = os.listdir(BASE_DIR)
    print(f'Available experiments:')
    for ex in all_experiments:
        print(f'  - {ex}')

    exs = input('Enter the experiment(s) to generate LaTeX for: ').split(' ')
    if len(exs) == 0:
        raise IOError('No experiments entered.')

    for ex in exs:
        if ex not in all_experiments:
            raise IOError(f'Experiment "{ex}" not found.')

    return exs


def get_name(o):
    if 'Movie' in label_map[o['uri']]:
        return f'\\textbf{{{o["name"]}}}'
    else:
        return o["name"]


def get_count(o):
    return f'({o["count"]})'


if __name__ == '__main__':
    # Ask for the experiments to generate LaTeX for
    with open('../data_loading/mindreader/entities_clean.json') as fp:
        entities = json.load(fp)

    label_map = {}
    for uri, name, labels in entities:
        if uri not in label_map:
            label_map[uri] = labels.split('|')

    with open('top.json') as fp:
        top = json.load(fp)['top']
        liked, disliked, unknown = top['liked'], top['disliked'], top['unknown']

        for l, d, u in zip(liked, disliked, unknown):
            print(f'''
            {get_name(l)}  & {get_name(d)}  & {get_name(u)} \\\\
            {get_count(l)} & {get_count(d)} & {get_count(u)} \\\\\\hline
            ''')


