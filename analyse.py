#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 01:04:15 2020

@author: kavianshirkoohi
"""

from pylab import*
import scipy as scipy
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import pandas
from network_functionality import *

"""
----------------------------------------
Special functions
----------------------------------------
"""

# Read two column data from file separated by whitespace
def import_from_file(filename):
    data = (pandas.read_table(filename, delim_whitespace=True, skiprows=1)).to_numpy()
    data_arr = np.column_stack((data[:,0],data[:,1]))
    N = len(data_arr)
    return data_arr

# Port 2D data frame to adjacency matrix
# Each row represents an edge connection 
# Returns adjacency matrix as an array
def port_data_to_matrix(data_arr):
    max_data = np.arange(0,2,1)
    max_data[0] = max(data_arr[:,0])
    max_data[1] = max(data_arr[:,1])
    size = max(max_data) + 1
    adj_matrix = [[0 for i in range(size)] for j in range(size)]
    for row,column in data_arr:
        adj_matrix[int(row)][int(column)] = 1  
    return np.array(adj_matrix)

"""
----------------------------------------
Main program
----------------------------------------
"""

# Convert data frame to adjacency matrix and draw graph
# Graph will be used to apply NetworkX to analyse the properties

data_arr = import_from_file('facebook_combined.txt')
adj_matrix = port_data_to_matrix(data_arr)
G = nx.Graph()
create_graph(adj_matrix, G, 0)

"""
# Drawing graphs
plt.title('Degree distribution')
degree_dist = tab_degree_dist(adj_matrix)
plt.bar(degree_dist[0],degree_dist[1])
plt.ylabel('Frequency')
plt.xlabel('Degree')
plt.show()

plt.title('Clustering coefficient vs. degree')
clustering = tab_clust_coeff_vs_degree(G, adj_matrix)
plt.bar(clustering[0],clustering[1])
plt.ylabel('Clustering coefficent')
plt.xlabel('Degree')
plt.show()
"""

"""
plt.title('Betweenness vs. degree')
betweenness = tab_betweenness_vs_degree(G, adj_matrix)
plt.bar(betweenness[0],betweenness[1])
plt.ylabel('Betweeness')
plt.xlabel('Degree')
plt.show()

plt.title('Closeness vs. degree')
betweenness = tab_closeness_vs_degree(G, adj_matrix)
plt.bar(betweenness[0],betweenness[1])
plt.ylabel('Closeness')
plt.xlabel('Degree')
plt.show()

plt.title('Eigenvector centrality vs. degree')
betweenness = tab_eigencen_vs_degree(G, adj_matrix)
plt.bar(betweenness[0],betweenness[1])
plt.ylabel('Eigenvector centrality')
plt.xlabel('Degree')
plt.show()

plt.title('Degree centrality vs. degree')
centrality = tab_degree_centrality_vs_degree(G, adj_matrix)
plt.bar(centrality[0],centrality[1])
plt.ylabel('Degree centrality')
plt.xlabel('Degree')
plt.show()
"""

# Ascertaining properties

L = get_mean_path_len(adj_matrix)
print("\nMean path length, L =", L)

r = get_degree_assortativity(G)
print("\nAssortativity coefficient, r =", r)

# Random attack
random_attack = do_random_attack(adj_matrix,5)
H = nx.Graph()
create_graph(random_attack, H, 0)

degree_dist = tab_degree_dist(adj_matrix)
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
target_attack = do_target_attack(adj_matrix,5)
I = nx.Graph()
create_graph(target_attack, I, 0)

degree_dist = tab_degree_dist(adj_matrix)
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








