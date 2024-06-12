# complex-networks
Investigations of Complex Networks submitted as my undergraduate dissertation.

The project involves developing software to investigate complex networks, focusing on their properties such as degree distribution, robustness, assortativity, and clustering factor. It also explores the application of complex networks in real-world phenomena.

## Key Sections

1. **Introduction**
   - **Definition of a Network**: A network is a mathematical model of a system of data points (nodes) connected by relations (arcs). Examples include transportation networks like the London Underground.
   - **Computer Representation**: Networks are represented using adjacency matrices, with nodes and arcs defined in a discrete manner. Python libraries such as Scipy, Numpy, Matplotlib, and NetworkX are utilised.

2. **Complex Networks**
   - **Characteristics**: Complex networks exhibit non-trivial topological properties not found in simple random or lattice networks. Key types include scale-free and small world networks.
   - **Properties**:
     - **Scale-Free Networks**: Characterised by a few highly connected hubs following a power-law degree distribution.
     - **Small World Networks**: Exhibit high connectivity with nodes connected by a small number of arcs, adhering to the six degrees of separation theory.

3. **Investigating Properties**
   - **Implementation**: Python programs were developed to generate and analyse scale-free and small world networks using the Barabási-Albert model and Watts-Strogatz model respectively.
   - **Mean Path Length**: Shows a logarithmic relationship with the number of nodes for both network types, decreasing with increased rewiring in small world networks.
   - **Degree Distribution**: Scale-free networks have high degree distribution due to hubs, while small world networks tend to show a Gaussian distribution.
   - **Measures of Centrality**: Degree, closeness, and betweenness centralities were analysed, with degree centrality being most consistent across network types.
   - **Clustering Coefficient and Assortativity**: Higher degree nodes tend to have lower clustering coefficients, and assortativity measures showed varied results based on network type and parameters.

4. **Robustness**
   - **Evaluation**: Both network types were tested for robustness against random and targeted attacks. Scale-free networks are resilient to random attacks but vulnerable to targeted ones, while small world networks show similar susceptibility to targeted attacks.

5. **Applications**
   - **Real-World Applications**: Complex networks can model various data sets in science and society. For example, social networks like Facebook can be analysed for properties such as degree distribution and clustering coefficient.
   - **Social Network Analysis**: A social network from Facebook data was analysed, showing properties similar to scale-free networks, with high degree distribution and clustering within cliques.

6. **Conclusion**
   - **Findings**: The study demonstrates the properties and robustness of complex networks, suggesting that many real-world networks are more random in nature. Scale-free networks tend to be closer to real social networks.
   - **Future Work**: Emphasises the evolving nature of the field of complex networks, highlighting the importance of understanding connected systems for efficient problem-solving in data science.

## Appendices

The appendices include Python code used for the various network analyses, demonstrating the implementation details for creating and investigating the properties of complex networks.

This report provides a comprehensive study on the properties and applications of complex networks, using computational methods to explore their robustness and practical applications in real-world data analysis.

Data sets provided as Fair Use for demonstrative purposes.

Please see .pdf for full documentation.
