from __future__ import print_function
import json
from os.path import isfile, join
from os import makedirs
import pandas as pd
import argparse

from loguru import logger

from models.entity2rec.node2vec import Node2Vec
import time
import shutil


class Entity2Vec(Node2Vec):

    """Generates a set of property-specific entity embeddings from a Knowledge Graph"""

    def __init__(self, is_directed, preprocessing, is_weighted, p, q, walk_length, num_walks, dimensions, window_size,
                 workers, iterations, feedback_file, split, training):

        Node2Vec.__init__(self, is_directed, preprocessing, is_weighted, p, q, walk_length, num_walks, dimensions,
                          window_size, workers, iterations)

        self.split = split
        self.reverse_map = {v: k for k,v in split.experiment.dataset.e_idx_map.items()}
        self.training = training
        self.feedback_file = feedback_file

    def e2v_walks_learn(self, properties_names, dataset):
        n = self.num_walks
        p = int(self.p)
        q = int(self.q)
        l = self.walk_length
        d = self.dimensions
        it = self.iter
        win = self.window_size

        try:
            makedirs('emb/%s' % dataset)
        except:
            pass

        feedback_edges = {'head_uri': [], 'tail_uri': []}
        for user, ratings in self.training:
            for rating in ratings:
                if rating.rating == 1 and rating.e_idx in self.reverse_map:
                    feedback_edges['head_uri'].append(str(user))
                    feedback_edges['tail_uri'].append(self.reverse_map[rating.e_idx])

        feedback_edges = pd.DataFrame.from_dict(feedback_edges)

        # copy define feedback_file, if declared
        if self.feedback_file:
            logger.debug('Copying feedback file %s' % self.feedback_file)
            shutil.copy2(self.feedback_file, "datasets/%s/graphs/feedback.edgelist" % dataset)

        # iterate through properties

        for prop_name in properties_names:
            logger.debug(prop_name)
            prop_short = prop_name

            if '/' in prop_name:
                prop_short = prop_name.split('/')[-1]

            graph = "datasets/%s/graphs/%s.edgelist" % (dataset, prop_short)

            try:
                makedirs('emb/%s/%s' % (dataset, prop_short))
            except:
                pass

            if prop_name == 'feedback':
                splitting = self.split.experiment.name
                number = self.split.name.split('.')[0]
                emb_output = "emb/%s/%s/num%d_p%d_q%d_l%d_d%d_iter%d_winsize%d%s-%s.emd" % (dataset,
                                                                                           prop_short, n, p, q, l, d,
                                                                                           it, win, splitting, number)
            else:
                emb_output = "emb/%s/%s/num%d_p%d_q%d_l%d_d%d_iter%d_winsize%d.emd" % (dataset,
                                                                                       prop_short, n, p, q, l, d, it,
                                                                                       win)

            # Always create new embeddings for feedback
            # TODO: Remove outcommenting
            # if prop_name == 'feedback':
            #     super(Entity2Vec, self).run(feedback_edges, emb_output)
            #     continue

            if not isfile(emb_output):  # check if embedding file already exists
                logger.debug('running with', graph)
                if prop_name == 'feedback':
                    edgelist = feedback_edges
                else:
                    idx_map = self.split.experiment.dataset.e_idx_map
                    triples = pd.read_csv(self.split.experiment.dataset.triples_path)
                    edgelist = triples.loc[triples['relation'] == prop_name][['head_uri', 'tail_uri']]

                    # TODO remove following line when data is correct.
                    edgelist = edgelist.loc[edgelist['head_uri'].isin(idx_map) & edgelist['tail_uri'].isin(idx_map)]

                # edgelist = edgelist.applymap(lambda x: idx_map[x])
                super(Entity2Vec, self).run(edgelist, emb_output)  # call the run function defined in parent class node2vec
            else:
                logger.debug('Embedding file already exist, going to next property...')
                continue

    @staticmethod
    def parse_args():

        """
        Parses the entity2vec arguments.
        """

        parser = argparse.ArgumentParser(description="Run entity2vec.")

        parser.add_argument('--walk_length', type=int, default=10,
                            help='Length of walk per source. Default is 10.')

        parser.add_argument('--num_walks', type=int, default=500,
                            help='Number of walks per source. Default is 40.')

        parser.add_argument('--p', type=float, default=1,
                            help='Return hyperparameter. Default is 1.')

        parser.add_argument('--q', type=float, default=1,
                            help='Inout hyperparameter. Default is 1.')

        parser.add_argument('--weighted', dest='weighted', action='store_true',
                            help='Boolean specifying (un)weighted. Default is unweighted.')
        parser.add_argument('--unweighted', dest='unweighted', action='store_false')
        parser.set_defaults(weighted=False)

        parser.add_argument('--directed', dest='directed', action='store_true',
                            help='Graph is (un)directed. Default is directed.')
        parser.set_defaults(directed=False)

        parser.add_argument('--no_preprocessing', dest='preprocessing', action='store_false',
                            help='Whether preprocess all transition probabilities or compute on the fly')
        parser.set_defaults(preprocessing=True)

        parser.add_argument('--dimensions', type=int, default=500,
                            help='Number of dimensions. Default is 128.')

        parser.add_argument('--window-size', type=int, default=10,
                            help='Context size for optimization. Default is 10.')

        parser.add_argument('--iter', default=5, type=int,
                            help='Number of epochs in SGD')

        parser.add_argument('--workers', type=int, default=8,
                            help='Number of parallel workers. Default is 8.')

        parser.add_argument('--config_file', nargs='?', default='config/properties.json',
                            help='Path to configuration file')

        parser.add_argument('--dataset', nargs='?', default='movielens_1m',
                            help='Dataset')

        parser.add_argument('--feedback_file', dest='feedback_file', default=False,
                            help='Path to a DAT file that contains all the couples user-item')

        return parser.parse_args()


if __name__ == '__main__':

    start_time = time.time()

    args = Entity2Vec.parse_args()

    logger.debug('Parameters:\n')

    logger.debug('walk length = %d\n' % args.walk_length)

    logger.debug('number of walks per entity = %d\n' % args.num_walks)

    logger.debug('p = %s\n' % args.p)

    logger.debug('q = %s\n' % args.q)

    logger.debug('weighted = %s\n' % args.weighted)

    logger.debug('directed = %s\n' % args.directed)

    logger.debug('no_preprocessing = %s\n' % args.preprocessing)

    logger.debug('dimensions = %s\n' % args.dimensions)

    logger.debug('iterations = %s\n' % args.iter)

    logger.debug('window size = %s\n' % args.window_size)

    logger.debug('workers = %s\n' % args.workers)

    logger.debug('config_file = %s\n' % args.config_file)

    logger.debug('dataset = %s\n' % args.dataset)

    logger.debug('feedback file = %s\n' % args.feedback_file)

    e2v = Entity2Vec(args.directed, args.preprocessing, args.weighted, args.p, args.q, args.walk_length, args.num_walks,
                     args.dimensions, args.window_size, args.workers, args.iter, args.config_file,
                     args.dataset, args.feedback_file)

    e2v.e2v_walks_learn()

    logger.debug("--- %s seconds ---" % (time.time() - start_time))