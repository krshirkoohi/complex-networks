# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:41:57 2020

@author: kavianshirkoohi
"""

import scipy as scipy
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from network_functionality import *

"""
----------------------------------------
Special functions
----------------------------------------
"""

# Returns a random network
def create_random_network(N,P):
    adj_matrix = np.zeros((N,N), dtype=np.int)
    for i in range(N):
        for j in range(N):
            rand = random.uniform(0, 1)
            if rand <= P:
                adj_matrix[i,j] = 1
                adj_matrix[j,i] = 1
    return adj_matrix

# Scale free network
# Barabási–Albert model
# Returns new adjacency matrix
def create_scale_free_network(old_matrix, S,A):
    N = len(old_matrix)
    for i in range(S + 1):
        new_dim    = N + i
        new_matrix = np.zeros((new_dim, new_dim), dtype=np.int)
        
        # Port old matrix to new matrix
        for j in range(0, new_dim - 1):
            for k in range(0, new_dim - 1):
                new_matrix[j,k] = old_matrix[j,k]
                new_matrix[k,j] = old_matrix[k,j]
        
        # Get information about node degree
        degree_info = get_degree_info(new_matrix)
        degree_arr  = degree_info[0]
        prob_arr    = degree_info[1]
        
        # Choose node to attach new node to
        random_trial = random.uniform(0, max(prob_arr)) * (1-A)
        high_nodes   = np.where(random_trial < prob_arr)[0]
        chosen_node  = random.choice(high_nodes) 
        
        # Check node doesn't disconnect graph!
        while not (check_is_node_connected(new_matrix, chosen_node)):
            chosen_node  = random.choice(high_nodes)
            print(chosen_node)
    
        # Add chosen node to new 
        new_matrix[chosen_node, new_dim - 1] = 1
        new_matrix[new_dim - 1, chosen_node] = 1
        
        
        # Prepare for next step
        old_matrix = new_matrix
    return new_matrix

"""
----------------------------------------
Main program
----------------------------------------
"""

N = 5 # Nodes in original network
P = 0.5 # Connectivity of original network
S = 100 # How many new nodes to add for scale free network
A = 0.8 # Assortativity factor; how likely new hubs are to be formed  
G = nx.Graph()
random_network = create_random_network(N,P)
scale_free_network = create_scale_free_network(random_network, S, A)
create_graph(scale_free_network,G,0)

# Random attack
random_attack = do_random_attack(scale_free_network,1)
H = nx.Graph()
create_graph(random_attack, H, 0)

degree_dist = tab_degree_dist(scale_free_network)
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
target_attack = do_target_attack(scale_free_network,1)
I = nx.Graph()
create_graph(target_attack, I, 0)

degree_dist = tab_degree_dist(scale_free_network)
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


# Note: using bar charts due to handling discrete data, line graphs better for continuous data
