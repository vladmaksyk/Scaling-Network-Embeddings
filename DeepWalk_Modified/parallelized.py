import networkx as nx
import numpy as np
#import matplotlib.pyplot as plt
import random
import pandas as pd
import math
import time
import multiprocessing
import datetime
from collections import Counter
from tqdm import tqdm
import array as arr
import copy
from multiprocessing import Pool
from itertools import repeat
import tqdm

filepathCSV = "../edgelists/BlogCatalog-edgelist.csv"
embeddingsPath = "../embeddings/BlogCatalog-approximate.txt.embeddings"

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

def CountContextPairs(contextPairs):
    total_count = 0
    for cp_set in contextPairs:
        total_count += len(cp_set)
    print("Total amount of context pairs :", total_count)

def save_embeddings(path ,contextPairs):
    file = open(path, 'w')
    #Writing to file
    for (key, value) in contextPairs.items():
        file.write(str(key) + " " + str(value) + "\n" )
    file.close()
    print("Successfully written embeddings to file:", embeddingsPath)


def getAdjNPList(graph):
    adjdict = {}
    for vertex in graph:
        adjdict[vertex] = np.array([n for n in graph.neighbors(vertex)])
        np.random.shuffle(adjdict[vertex])
    return adjdict

def chooseNodes(list_nodes, sample_size):
    return random.sample(population=list_nodes, k=sample_size)


def getPerNodeBudget(numNodes, budget):
    return math.floor(budget / numNodes)


def storeContextPairs(context_pair, budget, context_pairs):
    if context_pair not in context_pairs:
        context_pairs[context_pair] = budget
    else:
        context_pairs[context_pair] = context_pairs[context_pair] + budget


def updateContextPairs(window, window_count, context_pairs):
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
        context_pairs.extend([context_pair1 for i in range(budgetOfLastNode)])
        #context_pairs.append(context_pair1)

        context_pair2 = (node_label, labelOfLastNode)
        context_pairs.extend([context_pair2 for i in range(budgetOfLastNode)])
        #context_pairs.append(context_pair2)

        index_count = index_count + 1
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
            context_pairs = updateContextPairs(temp_window, temp_window_count, context_pairs)

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




def Runner(wl,b,ows,input,output,workers):
    start = time.time()
    global WALK_LENGHT
    global BUDGET
    global ORIGINAL_WINDOW_SIZE
    global QUEUE_BUFFER_SIZE
    global WINDOW_SIZE
    global adjdict
    global all_sets

    WALK_LENGHT = wl
    BUDGET = b
    ORIGINAL_WINDOW_SIZE = ows

    WINDOW_SIZE = ORIGINAL_WINDOW_SIZE * 2 + 1
    QUEUE_BUFFER_SIZE = (BUDGET * WALK_LENGHT) - (BUDGET * 2) + 1

    #Load real graph
    print("Loading Data...")
    G = parseEdgeList2(input)
    adjdict = getAdjNPList(G)

    #num_processors = multiprocessing.cpu_count()
    print("Running BFSRandomWalk...")
    print("Walk lenght:", WALK_LENGHT, ", Budget:", BUDGET, ", Window size:", ORIGINAL_WINDOW_SIZE, ", Workers:",workers)
    #p = Pool(processes=workers)
    nodes = [i for i in adjdict.keys()]
    with Pool(workers) as p:
        contextPairs = p.starmap(BFSRandomWalkWindow, zip(nodes, repeat(adjdict)))

    end = time.time()
    result = end - start
    print("The execution time ->", str(datetime.timedelta(seconds=round(result))))

    CountContextPairs(contextPairs)
    save_embeddings(output, contextPairs)

    p.terminate()

if __name__ ==  '__main__':
    start = time.time()

    #RUNNER

    walklength = 40
    budget = 1
    windowsize = 10
    inputfile = filepathCSV
    outputfile = embeddingsPath
    direction = "undirected"
    workers = 4

    contextPairs = Runner(walklength,budget,windowsize,inputfile,outputfile,workers)





