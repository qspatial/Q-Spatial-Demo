import networkx as nx
import matplotlib.pyplot as plt
import pickle

# Load graph
with open('graph.gpickle', 'rb') as f:
    G = pickle.load(f)

# Define partition assignments (from your results)
partitions = {
    'A': 1, 'B': 1, 'C': 2, 'D': 2, 'E': 1,
    'F': 1, 'G': 1, 'H': 2, 'I': 0, 'J': 0,
    'K': 0, 'L': 2, 'M': 0, 'N': 0, 'O': 2
}

# Node positions (same as original)
pos = {
    'A': (0, 3), 'B': (1, 3), 'C': (2, 3), 'D': (3, 3),
    'E': (0, 2), 'F': (1, 2), 'G': (2, 2), 'H': (3, 2),
    'I': (0, 1), 'J': (1, 1), 'K': (2, 1), 'L': (3, 1),
    'M': (0.5, 0), 'N': (1.5, 0), 'O': (2.5, 0)
}

# Color map for partitions
colors = ['#FFB6C1', '#ADD8E6', '#90EE90']  # Light red, blue, green
node_colors = [colors[partitions[n]] for n in G.nodes()]

# Separate edges into internal and cut edges
internal_edges = [(u, v) for u, v in G.edges() if partitions[u] == partitions[v]]
cut_edges = [(u, v) for u, v in G.edges() if partitions[u] != partitions[v]]

# Create figure
plt.figure(figsize=(12, 10))

# Draw internal edges (thinner, gray)
nx.draw_networkx_edges(G, pos, edgelist=internal_edges, width=2, alpha=0.3, edge_color='gray')

# Draw cut edges (thicker, red)
nx.draw_networkx_edges(G, pos, edgelist=cut_edges, width=3, alpha=0.8, edge_color='red')

# Draw nodes with partition colors
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800,
                       edgecolors='black', linewidths=2)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')

# Draw edge weights
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors[0], edgecolor='black', label='Partition 0: I,J,K,M,N'),
    Patch(facecolor=colors[1], edgecolor='black', label='Partition 1: A,B,E,F,G'),
    Patch(facecolor=colors[2], edgecolor='black', label='Partition 2: C,D,H,L,O'),
    plt.Line2D([0], [0], color='red', linewidth=3, label='Cut edges (8)')
]
plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.title('Graph Partitioning Results (Weighted Cut: 13.70)', fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('partition_visualization.png', dpi=300, bbox_inches='tight')
print("Saved: partition_visualization.png")
plt.show()