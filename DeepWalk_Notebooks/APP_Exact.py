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
embeddingsiterative = "output/BlogCatalog-edgelist.txt"

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

G = parseEdgeList(filepath)

# G = nx.Graph()
# #Small example
# G.add_nodes_from(["A","B","C","D","E","F","G","H","I","J","K","L"])
# # G.add_nodes_from(["A","B","C","D","E"])
# G.add_edge("A", "D"),G.add_edge("A", "E"),G.add_edge("A", "I"),G.add_edge("A", "K")
# G.add_edge("B", "D"),G.add_edge("B", "C"),G.add_edge("B", "L"),G.add_edge("B", "K")
# G.add_edge("C", "D"),G.add_edge("D", "H"),G.add_edge("D", "G"),G.add_edge("D", "E")
# G.add_edge("E", "F"),G.add_edge("F", "G"),G.add_edge("I", "E"),G.add_edge("I", "J"),G.add_edge("K", "L")
# # Draw graph
# nx.draw(G, with_labels = True)
# plt.show()


def WriteToFile(file):
    femb_iterative = open(file, 'w')
    for (key, value) in context_pairs.items():
        femb_iterative.write(key + " " + str(value) + "\n" )
    femb_iterative.close()

def countSum(contextPairs):
    countSum = 0
    for key, value in contextPairs.items():
        countSum += value
    print("Total value sums up to: ", countSum)

def getPerNodeBudget(numNodes, budget):
    return math.floor(budget/numNodes)

def chooseNodes(list_nodes, n):
    return random.sample(population=list_nodes, k=n)

def updateContextPairs(context_pair, num_rand_walks_ending_here, context_pairs):
    if context_pair not in context_pairs:
        context_pairs[context_pair] = num_rand_walks_ending_here
    else:
        context_pairs[context_pair] = context_pairs[context_pair] +  num_rand_walks_ending_here

def BFSRandomWalk(graph, start, queue, context_pairs):
    random.seed(1)
    while queue:
        vertex, budget = queue.pop(0)
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
            num_rand_walks_ending_here =  math.floor(budget_for_this_node * EPSILON)
            context_pair = str(start) + " " + str(neighbor)
            if(num_rand_walks_ending_here > 0):
                updateContextPairs(context_pair, num_rand_walks_ending_here, context_pairs)
            #remaining_budget = math.floor(budget_for_this_node * (1 - EPSILON))
            remaining_budget = budget_for_this_node - num_rand_walks_ending_here
            if remaining_budget > 0:
                if remaining_budget > 1:
                    queue.append((neighbor, remaining_budget))
                else:
                    randval = random.random()
                    if randval < EPSILON:
                        queue.append((neighbor, remaining_budget))
                    else:
                        updateContextPairs(context_pair, 1, context_pairs)


print("Running BFS...")

context_pairs = {}
start = time.time()
for startvertex in G:
    queue = [(startvertex, NUM_RANDOM_WALKS)]
    BFSRandomWalk(G, startvertex, queue, context_pairs)
countSum(context_pairs)
WriteToFile(embeddingsiterative)

end = time.time()
result = end - start
print("Run in :",result)

