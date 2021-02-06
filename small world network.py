# -*- coding: utf-8 -*-
"""
Spyder Editor

- Generate small-world network
- Connect nodes to neighbours
- Pick nodes at random and rewire connections
- Eventually becomes "small world"

"""

import scipy as scipy
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from network_functionality import *

def create_ring_network(N):
    adj_matrix = np.zeros((N,N), dtype=np.int)
    # Connect to neighbours each side
    for i in range(N-1):
        for j in range(N-1):
            if i+1 == j+1:
                adj_matrix[i,j+1] = 1 
                adj_matrix[i+1,j] = 1 
                G.add_edge(i+1, j, weight=adj_matrix[i,j]) 
    # Connect to two neigbhours each side
    if N > 50:
        for i in range(N-2):
            for j in range(N-2):
                if i+2 == j+2:
                    adj_matrix[i,j+2] = 1 
                    adj_matrix[i+2,j] = 1 
                    G.add_edge(i+2, j, weight=adj_matrix[i,j])         
    # Connect to three neighbours each side
    if N > 100:
        for i in range(N-3):
            for j in range(N-3):
                if i+3 == j+3:
                    adj_matrix[i,j+3] = 1 
                    adj_matrix[i+3,j] = 1 
                    G.add_edge(i+3, j, weight=adj_matrix[i,j])                    
    # Close the loop
    adj_matrix[0, N-1] = 1
    adj_matrix[N-1, 0] = 1                
    return adj_matrix

def create_small_world_network(adj_matrix, B):
    # Watts-Strogatz model
    N = len(adj_matrix) # total number of nodes
    chosen_arcs_amount = int(B * N) # number of arcs chosen randomly
    
    for i in range(chosen_arcs_amount):         
        rewired_node = int(random.uniform(0, N))
        adj_matrix[rewired_node, rewired_node - 1] = 0
        adj_matrix[rewired_node - 1, rewired_node] = 0        
        # Check arc choice doesn't disconnect graph!
        while not (check_is_node_connected(adj_matrix, rewired_node)):
            rewired_node = int(random.uniform(0, N))
            adj_matrix[rewired_node, rewired_node - 1] = 0
            adj_matrix[rewired_node - 1, rewired_node] = 0
        random_node = int(random.uniform(0, N)) 
        adj_matrix[rewired_node, random_node] = 1
        adj_matrix[random_node, rewired_node] = 1                   
  
    return adj_matrix 
  
"""
----------------------------------------
Main program
----------------------------------------
"""

# Initialise
N = 100
B = 0.8
G = nx.Graph()
ring_network = create_ring_network(N)
small_world_network = create_small_world_network(ring_network, B)
create_graph(small_world_network,G,1)

# Random attack
random_attack = do_random_attack(small_world_network,1)
H = nx.Graph()
create_graph(random_attack, H,1)

degree_dist = tab_degree_dist(small_world_network)
line1 = plt.bar(degree_dist[0],degree_dist[1])
degree_dist = tab_degree_dist(random_attack)
line2 = plt.bar(degree_dist[0],degree_dist[1], alpha=0.75)
plt.title('Degree distribution with random attack')
plt.ylabel('Frequency')
plt.xlabel('Degree')
plt.legend((line1, line2),('Before attack','After attack'))
plt.show()

connectivity = nx.average_degree_connectivity(G, source='in+out', target='in+out', nodes=None, weight=None)
degree_list = list(connectivity.keys())
value_list = list(connectivity.values())
line1 = plt.bar(degree_list,value_list)
connectivity = nx.average_degree_connectivity(H, source='in+out', target='in+out', nodes=None, weight=None)
degree_list = list(connectivity.keys())
value_list = list(connectivity.values())
line2 = plt.bar(degree_list,value_list, alpha=0.75)
plt.title('Connectivity with random attack')
plt.ylabel('Connectivity')
plt.xlabel('Degree')
plt.legend((line1, line2),('Before attack','After attack'))
plt.show()

# Targeted attack
target_attack = do_target_attack(small_world_network,1)
I = nx.Graph()
create_graph(target_attack, I,1)

degree_dist = tab_degree_dist(small_world_network)
line1 = plt.bar(degree_dist[0],degree_dist[1])
degree_dist = tab_degree_dist(target_attack)
line2 = plt.bar(degree_dist[0],degree_dist[1], alpha=0.75)
plt.title('Degree distribution with targeted attack')
plt.ylabel('Frequency')
plt.xlabel('Degree')
plt.legend((line1, line2),('Before attack','After attack'))
plt.show()

connectivity = nx.average_degree_connectivity(G, source='in+out', target='in+out', nodes=None, weight=None)
degree_list = list(connectivity.keys())
value_list = list(connectivity.values())
line1 = plt.bar(degree_list,value_list)
connectivity = nx.average_degree_connectivity(I, source='in+out', target='in+out', nodes=None, weight=None)
degree_list = list(connectivity.keys())
value_list = list(connectivity.values())
line2 = plt.bar(degree_list,value_list, alpha=0.75)
plt.title('Connectivity with targeted attack')
plt.ylabel('Connectivity')
plt.xlabel('Degree')
plt.legend((line1, line2),('Before attack','After attack'))
plt.show()

# Important note, always remember that total number of edges in network is also affected by adjacent neighbours
    
    