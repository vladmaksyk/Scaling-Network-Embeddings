#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import networkx as nx
import random
import time
import datetime
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging

from . import approximate
from . import exact
from . import parallelized
#import walks as serialized_walks
from gensim.models import Word2Vec
#from skipgram import Skipgram

from six import text_type as unicode
from six import iteritems
from six.moves import range

import psutil
from multiprocessing import cpu_count

p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def toyGraph():
    G = nx.Graph()
    G.add_nodes_from([1,2,3,4,5,6,7,8,9,10,11])
    G.add_edge(2, 1);G.add_edge(3, 1);G.add_edge(3, 2);G.add_edge(4, 1);
    G.add_edge(4, 2);G.add_edge(4, 3);G.add_edge(5, 1);G.add_edge(6, 1);
    G.add_edge(7, 1);G.add_edge(7, 5);G.add_edge(7, 6);G.add_edge(8, 1);
    G.add_edge(8, 2);G.add_edge(8, 3);G.add_edge(8, 4);G.add_edge(9, 1);
    G.add_edge(9, 3);G.add_edge(10, 3);G.add_edge(11, 1);G.add_edge(11, 5);
    print("Nodes: ", G.number_of_nodes()," Edges: ", G.number_of_edges())
    # Draw graph
    # nx.draw(G, with_labels = True)
    # plt.show()
    return G

def debug(type_, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        sys.__excepthook__(type_, value, tb)
    else:
        import traceback
        import pdb
        traceback.print_exception(type_, value, tb)
        print(u"\n")
        pdb.pm()

def updateContextPairs(node, context_node, context_pairs):
    if context_node != '':
        context_pair_str = node + " " + context_node
        if context_pair_str in context_pairs:
            context_pairs[context_pair_str] = context_pairs[context_pair_str] + 1
        else:
            context_pairs[context_pair_str] = 1

def convertPathsToContextPairs(max_paths, budget, window_size):
    context_pairs = {}
    fname = "output/CP_RandomWalks-"+"wl"+str(max_paths)+"-b"+str(budget)+"-ws"+str(window_size)+".txt"
    f = open(fname)
    for l in f:
        nodes = l.strip().split(' ')
        window = window_size
        i = 0
        while i < len(nodes):
            node = nodes[i].strip()
            if node == '':
                continue
            j = 1
            while j <= window:
                context_node = ''
                if i+j < len(nodes):
                    context_node =  nodes[i+j].strip()
                    updateContextPairs(node, context_node, context_pairs)
                if i-j >= 0:
                    context_node =  nodes[i-j].strip()
                    updateContextPairs(node, context_node, context_pairs)
                j = j + 1
            i = i + 1
    return context_pairs


def save_corpus(max_paths, budget, window_size, corpus):
    fname = "output/CP_RandomWalks-" + "wl" + str(max_paths) + "-b" + str(budget) + "-ws" + str(window_size) + ".txt"
    with open(fname, 'w+') as f:
        [f.writelines("%s\n" % ' '.join(walk)) for walk in corpus]
    print("Corpus saved on disk as " + fname)
    return


def countCP(context_pairs):
    countSum = 0
    for key, value in context_pairs.items():
        countSum += value
    print("Total value sums up to: ", countSum)

def process(args):

    if args.type == "exact":
        print("Running the exact version of DeepWalk")
        CP = exact.Runner(args.walk_length, args.budget, args.window_size, args.input, args.output)

    elif args.type == "approximate":

        print("Running the approximate version of DeepWalk")
        CP = approximate.Runner(args.walk_length, args.budget, args.window_size, args.input, args.output)

    elif args.type == "parallelized":
        print("Running the parallelized version of DeepWalk")
        CP = parallelized.Runner(args.walk_length, args.budget, args.window_size, args.input, args.output)

    else:
        print("Specify the correct version of DeepWalk (exact,approximate,parallelized)")

def main():
    parser = ArgumentParser("deepwalk",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument("--type", default="exact",
                        help="Choose the variation of DeepWalk")

    parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                        help="drop a debugger if an exception is raised.")

    parser.add_argument('--format', default='mat',
                        help='File format of input file')

    parser.add_argument('--input', nargs='?', required=True,
                        help='Input graph file')

    parser.add_argument("-l", "--log", dest="log", default="INFO",
                        help="log verbosity level")

    parser.add_argument('--matfile-variable-name', default='network',
                        help='variable name of adjacency matrix inside a .mat file.')

    parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
                        help='Size to start dumping walks to disk, instead of keeping them in memory.')

    parser.add_argument('--budget', default=10, type=int,
                        help='Number of random walks to start at each node')

    parser.add_argument('--output', required=True,
                        help='Output representation file')

    parser.add_argument('--representation-size', default=64, type=int,
                        help='Number of latent dimensions to learn for each node.')

    parser.add_argument('--seed', default=0, type=int,
                        help='Seed for random walk generator.')

    parser.add_argument('--undirected', default=True, type=bool,
                        help='Treat graph as undirected.')

    parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                        help='Use vertex degree to estimate the frequency of nodes '
                             'in the random walks. This option is faster than '
                             'calculating the vocabulary.')

    parser.add_argument('--walk-length', default=40, type=int,
                        help='Length of the random walk started at each node')

    parser.add_argument('--window-size', default=5, type=int,
                        help='Window size of skipgram model.')

    parser.add_argument('--workers', default=1, type=int,
                        help='Number of parallel processes.')

    #cmdargs = "--format edgelist --input input/BlogCatalog-edgelist.txt --max-memory-data-size 100000000 --representation-size 128 --undirected True --walk-length 40 --budget 1 --window-size 10 --workers 1 --output blogcatalog.embeddings"
    #cmdargs = "--type original --format edgelist --input input/BlogCatalog-edgelist.txt --max-memory-data-size 100000000 --representation-size 128 --undirected True --walk-length 40 --budget 1 --window-size 10 --workers 10 --output karate.embeddings"
    #cmdargs = "--type exact --format edgelist --input ../edgelists/BlogCatalog-edgelist.csv --max-memory-data-size 100000000 --representation-size 128 --undirected True --walk-length 40 --budget 1 --window-size 10 --workers 10 --output ../embeddings/BlogCatalog-approximate.txt.embeddings"

    args = parser.parse_args()
    numeric_level = getattr(logging, args.log.upper(), None)
    logging.basicConfig(format=LOGFORMAT)
    logger.setLevel(numeric_level)

    if args.debug:
        sys.excepthook = debug

    process(args)


if __name__ == "__main__":
    sys.exit(main())
