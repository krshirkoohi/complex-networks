# -*- coding: utf-8 -*-
"""
Spyder Editor

- For N nodes
- Connectivity P
- Adjacency matrix N-by-N array
- Connections are non-zero elements
- Weights are element values

"""

import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

N = 5 # total nodes
P = 0.3 # universal probability of connection
max_weight = 10 # max weight of nodes 

# create adjacency matrix
adj_matrix = np.zeros((N,N), dtype=np.int)
# create graph using NetworkX
G = nx.Graph()
G.add_nodes_from([0, N - 1])

# populate adjacency matrix
# connections according to P
for i in range(N):
    for j in range(N):
        # random connection chance
        # measured against P 
        # to decide if connection or not
        rand = random.uniform(0, 1)
        if rand <= P:
            adj_matrix[i,j] = random.uniform(1, max_weight) # assign random non-zero weight
            G.add_edge(i, j, weight=adj_matrix[i,j])
        # At some point might want to account for conjugate pair j,i too
        
# draw network using NetworkX 
# draw nodes with labels
pos=nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, labels=adj_matrix,node_color='r', node_size=250, alpha=1)

# draw arcs 
nx.draw_networkx_edges(G, pos, labels=adj_matrix,node_color='r', alpha=1)

# draw labels for nodes and weights of arcs
nx.draw_networkx_edge_labels(G, pos, nx.get_edge_attributes(G,'weight'))
nx.draw_networkx_labels(G, pos, 
                        labels=None, 
                        font_size=12, 
                        font_color='k', 
                        font_family='sans-serif', 
                        font_weight='normal', 
                        alpha=1.0, 
                        bbox=None, 
                        ax=None)

plt.savefig("simple_path.png") # save as png
plt.axis('off')
plt.show() # display

