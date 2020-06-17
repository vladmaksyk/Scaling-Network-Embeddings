import networkx as nx
import numpy as np
#import matplotlib.pyplot as plt
import random
import pandas as pd
import math
import time
from collections import Counter
from tqdm import tqdm
import array as arr
import copy
from multiprocessing import Pool
from itertools import repeat
import tqdm

filepathTXT = "input/BlogCatalog-edgelist.txt"
filepathCSV = "input/BlogCatalog-edgelist.csv"
#embeddingsrecursive = "../embeddings/BlogCatalog-edgelist.txt.embeddings-recursive"
#embeddingsiterative = "../embeddings/BlogCatalog-edgelist.txt.embeddings-iterative"

def toyGraph():
    G = nx.Graph()
    G.add_nodes_from([1,2,3,4,5,6,7,8,9,10,11])
    G.add_edge(2, 1);G.add_edge(3, 1); G.add_edge(3, 2); G.add_edge(4, 1)
    G.add_edge(4, 2);G.add_edge(4, 3);G.add_edge(5, 1); G.add_edge(6, 1)
    G.add_edge(7, 1);G.add_edge(7, 5);G.add_edge(7, 6); G.add_edge(8, 1)
    G.add_edge(8, 2);G.add_edge(8, 3);G.add_edge(8, 4); G.add_edge(9, 1)
    G.add_edge(9, 3);G.add_edge(10, 3);G.add_edge(11, 1); G.add_edge(11, 5)
    print("Nodes: ", G.number_of_nodes()," Edges: ", G.number_of_edges())
    # Draw graph
    # nx.draw(G, with_labels = True)
    # plt.show()
    return G

def parseEdgeList2(graph_file, direction="undirected"):
    G = nx.Graph()
    colNames=["Start", "End"]
    edgeData = pd.read_csv(filepathCSV, names=colNames)
    nodes = []
    for i in range (0, edgeData.shape[0]):
        nodes.append(edgeData.iloc[i,0])
        nodes.append(edgeData.iloc[i,1])
    nodes = set(nodes)
    uniqueNodes = (list(nodes))
    uniqueNodes.sort()
    G.add_nodes_from(uniqueNodes)
    edgeCount = 0
    for i in range (0, edgeData.shape[0]):
        edgeCount += 1
        G.add_edge(edgeData.iloc[i,0], edgeData.iloc[i,1])
    print("Nodes: ", G.number_of_nodes()," Edges: ", G.number_of_edges(), " loaded from ", graph_file)
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

def getNodeContextSets(nodes):
    all_sets = {}
    for node in range(1, nodes+1):
        all_sets[node] = []
    return all_sets

def CountContextPairs(contextPairs):
    countSum = 0
    for key, value in contextPairs.items():
        countSum += value
    print("Total value sums up to: ", countSum)

def MergeDictionaries(lis):
    contextPairs = {}
    for dicset in lis:
        for CP,budget in dicset.items():
            storeContextPairs(CP,budget,contextPairs)
    return contextPairs

def updateSetCount(startvertex, context_pair, context_budget):
    for i in range(0, context_budget):
        all_sets[startvertex].append(context_pair)

def chooseNodes(list_nodes, sample_size):
    return random.sample(population=list_nodes, k=sample_size)


def getPerNodeBudget(numNodes, budget):
    return math.floor(budget / numNodes)


def storeContextPairs(context_pair, budget, context_pairs):
    if context_pair not in context_pairs:
        context_pairs[context_pair] = budget
    else:
        context_pairs[context_pair] = context_pairs[context_pair] + budget


def updateContextPairs(startvertex, window, window_count, context_pairs):
    lastNode = window[window_count]
    labelOfLastNode, budgetOfLastNode = lastNode
    index_count = 0
    window_except_last = window[:window_count]
    window_reversed = window_except_last[::-1]
    for node in window_reversed:
        node_label = node[0]
        if index_count == ORIGINAL_WINDOW_SIZE:
            break
        context_pair1 = (labelOfLastNode, node_label)
        context_pair2 = (node_label, labelOfLastNode)
        context_pairs.extend([context_pair1 for i in range(budgetOfLastNode)])
        context_pairs.extend([context_pair2 for i in range(budgetOfLastNode)])

        index_count = index_count + 1

        updateSetCount(startvertex, context_pair1, budgetOfLastNode)
        updateSetCount(startvertex, context_pair2, budgetOfLastNode)
    return context_pairs

def addNewNodeToWindow(tempwindow, temp_window_count, window_size, vertex, budget):
    newWindowElement = np.array([[vertex,budget]])
    if temp_window_count+1 == window_size:
        tempwindow = tempwindow[1:]
        tempwindow[temp_window_count] = newWindowElement
        return tempwindow, temp_window_count

    tempwindow[temp_window_count+1] = newWindowElement
    temp_window_count+=1
    return tempwindow, temp_window_count


def BFSRandomWalkWindow(startvertex, adjdict):
    global all_sets

    context_pairs = []

    window_count = 0
    WINDOW = np.zeros(shape=(WINDOW_SIZE + 20, 2), dtype=int)
    firstWindowElement = np.array([[startvertex, BUDGET]])
    WINDOW[window_count] = firstWindowElement

    queue_len = 0
    queue_pop_idx = 0
    queue_add_idx = 0
    queue = np.zeros(shape=(BUDGET, 5), dtype=object)
    firstQueueElement = np.array([[startvertex, BUDGET, 1, window_count, WINDOW]])
    queue[queue_len] = firstQueueElement
    queue_len += 1
    queue_add_idx += 1

    queue_buffer_size = (queue.size / 5) - 1

    while queue_len > 0:
        vertex, budget, current_walk_lenght, window_count, window = queue[queue_pop_idx]

        #if all_sets[vertex] and vertex != startvertex:
        #    current_amount_of_cp = len(all_sets[startvertex])
        #    needed_amount_cp = len(all_sets[1])
        #    cp_to_sample = needed_amount_cp - current_amount_of_cp
        #    sampled_cp = random.sample(all_sets[vertex], cp_to_sample)
        #    for cp in sampled_cp:
        #        storeContextPairs(cp, 1, context_pairs)
        #    all_sets[startvertex].extend(sampled_cp)
        #    break

        queue_len -= 1
        if queue_len > 0:
            if queue_pop_idx == queue_buffer_size:
                queue_pop_idx = 0
            else:
                queue_pop_idx += 1
        else:
            queue_add_idx = 0

        vertex_neighbors = adjdict[vertex]
        num_neighbors = vertex_neighbors.size
        per_node_budget = getPerNodeBudget(num_neighbors, budget)
        remainder = budget - (per_node_budget * num_neighbors)

        bonus = 0
        if remainder > 0:
            np.random.shuffle(vertex_neighbors)
            bonus = remainder

        current_walk_lenght += 1
        for neighbor in vertex_neighbors:

            budget_for_this_node = per_node_budget
            temp_window = np.copy(window)
            temp_window_count = window_count

            if bonus == 0 and budget_for_this_node == 0:
                break

            if bonus > 0:
                bonus -= 1
                budget_for_this_node += 1

            temp_window, temp_window_count = addNewNodeToWindow(temp_window, temp_window_count, WINDOW_SIZE, neighbor,
                                                                budget_for_this_node)
            context_pairs = updateContextPairs(startvertex ,temp_window, temp_window_count, context_pairs)

            if current_walk_lenght < WALK_LENGHT:
                newQueueElement = np.array(
                    [[neighbor, budget_for_this_node, current_walk_lenght, temp_window_count, temp_window]])
                queue[queue_add_idx] = newQueueElement
                queue_len += 1
                if queue_add_idx == queue_buffer_size:
                    queue_add_idx = 0
                    continue
                queue_add_idx += 1
    return context_pairs


# Set the actual parameters and graph
WALK_LENGHT = 40
BUDGET = 1
ORIGINAL_WINDOW_SIZE = 10

# Set toy parameters and graph
#WALK_LENGHT = 6
#BUDGET = 3
#ORIGINAL_WINDOW_SIZE = 2

WINDOW_SIZE = ORIGINAL_WINDOW_SIZE * 2 + 1
NODES = 10312
all_sets = getNodeContextSets(NODES)


if __name__ ==  '__main__':
    print("Running BFSRandomWalk...")

    # Load Toy graph
    #G = toyGraph()
    #adjdict = getAdjNPList(G)

    # Load real graph
    G = parseEdgeList2(filepathCSV)
    adjdict = getAdjNPList(G)

    start = time.time()
    num_processors = 4
    p = Pool(processes=num_processors)
    nodes = [i for i in adjdict.keys()]
    contextPairs = p.starmap(BFSRandomWalkWindow, zip(nodes, repeat(adjdict)))

    #contextPairs = MergeDictionaries(contextPairs)
    total_count = 0
    for cp_set in contextPairs:
        total_count += len(cp_set)
    print("Total amount of context pairs :", total_count)

    end = time.time()
    result = end - start
    print("The execution time ->", result)





