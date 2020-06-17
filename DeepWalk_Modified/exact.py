import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
import time
#import graph
import numpy as np
from collections import Counter
from tqdm import tqdm
import array as arr
import copy


filepathCSV = "../edgelists/BlogCatalog-edgelist.csv"
embeddingsPath = "../embeddings/BlogCatalog-exact.txt.embeddings"

def parseEdgeList2(graph_file, direction="undirected"):
    # Create Graph
    G = nx.Graph()
    # Create head
    colNames=["Start", "End"]
    edgeData = pd.read_csv(filepathCSV, names=colNames)

    #Add nodes
    nodes = []
    #loop throug data records
    for i in range (0, edgeData.shape[0]):
        #append every node
        nodes.append(edgeData.iloc[i,0])
        nodes.append(edgeData.iloc[i,1])
    #creating a set of nodes
    nodes = set(nodes)
    #sorting the nodes in increasing order
    uniqueNodes = (list(nodes))
    uniqueNodes.sort()
    #adding the nodes to the graph
    G.add_nodes_from(uniqueNodes)

    # Add edges
    #loop from 0 to amount of records
    edgeCount = 0
    for i in range (0, edgeData.shape[0]):
        edgeCount += 1
        #add the edge to the graph
        G.add_edge(edgeData.iloc[i,0], edgeData.iloc[i,1])
    print("Nodes: ", G.number_of_nodes()," Edges: ", G.number_of_edges(), " loaded from ", graph_file)

    if(direction == "undirected"):
        return G.to_undirected()
    else:
        return G

def getAdjNPList(graph):
    adjdict = {}
    for vertex in graph:
        adjdict[vertex] = np.array([n for n in graph.neighbors(vertex)])
        np.random.shuffle(adjdict[vertex])
    return adjdict

def CountContextPairs(contextPairs):
    countSum = 0
    for key, value in contextPairs.items():
        countSum += value
    print("Total value sums up to: ", countSum)

def save_embeddings(path ,contextPairs):
    file = open(path, 'w')
    #Writing to file
    for (key, value) in contextPairs.items():
        file.write(str(key) + " " + str(value) + "\n" )
    file.close()
    print("Successfully written embeddings to file:", embeddingsPath)

def getToyGraph():
    G = nx.Graph()
    G.add_nodes_from(["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34"])
    inputfile = "input/karate.adjlist"
    lines = open(inputfile, "r")
    for line in lines:
        line = line.split()
        for node in line[1:]:
            G.add_edge(line[0], node);
    print("Nodes: ", G.number_of_nodes()," Edges: ", G.number_of_edges())
    #print("G.nodes ->", G.nodes)
    # Draw graph
    #nx.draw(G, with_labels = True)
    #plt.show()
    return G


def chooseNodes(list_nodes, sample_size):
    random.seed(0)
    return random.sample(population=list_nodes, k=sample_size)  # 2.22 s ± 179 ms per loop  (labesl str)


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
        context_pair1 = str(labelOfLastNode) + "," + str(node_label)
        context_pair2 = str(node_label) + "," + str(labelOfLastNode)
        storeContextPairs(context_pair1, budgetOfLastNode, context_pairs)
        storeContextPairs(context_pair2, budgetOfLastNode, context_pairs)
        index_count = index_count + 1


def addNewNodeToWindow(tempwindow, temp_window_count, window_size, vertex, budget):
    newWindowElement = np.array([[vertex, budget]])
    if temp_window_count + 1 == window_size:
        tempwindow = tempwindow[1:]
        tempwindow[temp_window_count] = newWindowElement
        return tempwindow, temp_window_count

    tempwindow[temp_window_count + 1] = newWindowElement
    temp_window_count += 1
    return tempwindow, temp_window_count


def BFSRandomWalkWindow(queue, queue_len, queue_pop_idx, queue_add_idx, context_pairs, window_size, walk_lenght):
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

            temp_window, temp_window_count = addNewNodeToWindow(temp_window, temp_window_count, window_size, neighbor,
                                                                budget_for_this_node)
            updateContextPairs(temp_window, temp_window_count, context_pairs)

            if current_walk_lenght < walk_lenght:
                newQueueElement = np.array(
                    [[neighbor, budget_for_this_node, current_walk_lenght, temp_window_count, temp_window]])
                queue[queue_add_idx] = newQueueElement

                queue_len += 1

                if queue_add_idx == queue_buffer_size:
                    queue_add_idx = 0
                    continue
                queue_add_idx += 1


def Runner(wl,b,ows,input,output):
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

    G = parseEdgeList2(input)  # Integer labels
    adjdict = getAdjNPList(G)

    start = time.time()
    random.seed(0)
    rand = random.Random(0)
    context_pairs = {}
    print("Running BFSRandomWalk...")

    for startvertex in adjdict.keys():
        # print("")
        # print("Running from -> ", startvertex, "with budget ->", BUDGET)
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

        BFSRandomWalkWindow(queue, queue_len, queue_pop_idx, queue_add_idx, context_pairs, WINDOW_SIZE, WALK_LENGHT)
    end = time.time()
    result = end - start
    print("Running time :", result)
    CountContextPairs(context_pairs)
    save_embeddings(output, context_pairs)
    return context_pairs


#RUNNER

# walklength = 40
# budget = 1
# windowsize = 10
# inputfile = filepathCSV
# outputfile = embeddingsPath
# direction = "undirected"

#contextPairs = Runner(walklength,budget,windowsize,inputfile,outputfile)




