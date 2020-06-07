import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
import time

#set the parameters
NUM_RANDOM_WALKS = 80
EPSILON = 0.15

filepath = "input/BlogCatalog-edgelist.txt"
filepath_Youtube = "input/youtube.txt"
embeddingsiterative = "output/BlogCatalog-edgelist.txt.embeddings-iterative"

def parseEdgeList(graph_file, delimiter=" ", weighted=False, direction="undirected"):
    if(weighted == False):
        G = nx.read_edgelist(graph_file, delimiter=delimiter)
    else:
        G = nx.read_edgelist(graph_file, delimiter=delimiter, nodetype=int, data=(('weight',float),))
    print(G.number_of_nodes(), G.number_of_edges(), " loaded from ", graph_file)
    if(direction == "undirected"):
        return G.to_undirected()
    else:
        return G

def getAdjNPList(graph):
    adjdict = {}
    for vertex in graph:
        adjdict[vertex] = np.array([n for n in G.neighbors(vertex)])
        np.random.shuffle(adjdict[vertex])
    return adjdict

def getNodeContextSets(graph):
    all_sets = {}
    for node in graph:
        all_sets[node] = []
    return all_sets

#G = parseEdgeList(filepath)
G = parseEdgeList(filepath_Youtube)

def getPerNodeBudget(numNodes, budget):
    return math.floor(budget/numNodes)

def updateSetCount(curent_node, context_pair, context_budget):
    for i in range(0, context_budget):
        all_sets[curent_node].append(context_pair)


def countSum(contextPairs):
    countSum = 0
    for key, value in contextPairs.items():
        countSum += value
    print("Total value sums up to: ", countSum)

def countListSum(all_sets):
    countSum = 0
    for key, value in all_sets.items():
        countSum += len(value)
    print("Total value sums up to: ", countSum)

def WriteToFile(file):
    femb_iterative = open(file, 'w')
    for (key, value) in context_pairs.items():
        femb_iterative.write(key + " " + str(value) + "\n" )
    femb_iterative.close()

def chooseNodes(list_nodes, n):
    return random.sample(population=list_nodes, k=n)

def updateContextPairs(context_pair, num_rand_walks_ending_here, context_pairs):
    if context_pair not in context_pairs:
        context_pairs[context_pair] = num_rand_walks_ending_here
    else:
        context_pairs[context_pair] = context_pairs[context_pair] +  num_rand_walks_ending_here


def BFSRandomWalk(graph, startvertex, queue, context_pairs):
    while queue:
        vertex, budget = queue.pop(0)

        if all_sets[vertex] and vertex != startvertex:
            current_amount_of_cp = len(all_sets[startvertex])
            needed_amount_cp = len(all_sets["1"])
            cp_to_sample = needed_amount_cp - current_amount_of_cp
            sampled_cp = random.sample(all_sets[vertex], cp_to_sample)
            all_sets[startvertex].extend(sampled_cp)
            break

        vertex_neighbors = [n for n in G.neighbors(vertex)]
        num_neighbors = len(vertex_neighbors)
        m = getPerNodeBudget(num_neighbors, budget)
        remainder = budget - (m * num_neighbors)
        chosen_nodes = []
        if remainder > 0:
            chosen_nodes = chooseNodes(vertex_neighbors, remainder)
        for neighbor in vertex_neighbors:
            budget_for_this_node = m
            if neighbor in chosen_nodes:
                budget_for_this_node = budget_for_this_node + 1
            num_rand_walks_ending_here = math.floor(budget_for_this_node * EPSILON)
            context_pair = str(startvertex) + " " + str(neighbor)
            # print("context_pair ->", context_pair)
            if (num_rand_walks_ending_here > 0):
                # updateContextPairs(context_pair, num_rand_walks_ending_here, context_pairs)
                updateSetCount(startvertex, context_pair, num_rand_walks_ending_here)
            remaining_budget = budget_for_this_node - num_rand_walks_ending_here
            if remaining_budget > 0:
                if remaining_budget > 1:
                    queue.append((neighbor, remaining_budget))
                else:
                    randval = random.random()
                    if randval < EPSILON:
                        queue.append((neighbor, remaining_budget))
                    else:
                        # updateContextPairs(context_pair, 1, context_pairs)
                        updateSetCount(startvertex, context_pair, 1)

adjdict = getAdjNPList(G)
all_sets = getNodeContextSets(G)

start = time.time()
context_pairs = {}
print("Running BFS...")
for startvertex in adjdict:
    queue = [(startvertex, NUM_RANDOM_WALKS)]
    BFSRandomWalk(adjdict, startvertex, queue, context_pairs)
countListSum(all_sets)
WriteToFile(embeddingsiterative)

end = time.time()
result = end - start
print("Run in :",result)

