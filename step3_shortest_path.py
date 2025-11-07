import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
edges = [
    (1, 2, 3),
    (1, 3, 4),
    (1, 5, 5),
    (2, 6, 3),
    (3, 4, 3),
    (3, 14, 4),
    (4, 6, 2),
    (4, 7, 3),
    (5, 6, 3),
    (5, 9, 5),
    (6, 7, 2),
    (6, 10, 3),
    (7, 11, 2),
    (8, 9, 2),
    (8, 10, 3),
    (9, 12, 3),
    (10, 11, 2),
    (11, 12, 2),
    (11, 13, 3),
    (12, 8, 2),
    (13, 14, 3),
]
for u, v, w in edges:
    G.add_edge(u, v, Weight=w)

# Use labels: City Name (Node Number)
node_labels = {
    1: "San Francisco (1)",
    2: "Palo Alto (2)",
    3: "Los Angeles (3)",
    4: "Denver (4)",
    5: "Houston (5)",
    6: "Chicago (6)",
    7: "Ann Arbor (7)",
    8: "Pittsburgh (8)",
    9: "Ithaca (9)",
    10: "New York (10)",
    11: "Princeton (11)",
    12: "College Park (12)",
    13: "Atlanta (13)",
    14: "San Diego (14)",
}

# SCF from CSV
scf = pd.read_csv("scf_230500226.csv")
source = int(scf.loc[0, "source"])
target = int(scf.loc[0, "destination"])

# Shortest Path
path = nx.shortest_path(G, source=source, target=target, weight="Weight")
cost = nx.shortest_path_length(G, source=source, target=target, weight="Weight")
print("Shortest path sequence:", path)
print("Total path cost (sum of weights):", cost)

# Visualization
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(12, 10))

# Draw nodes with labels "City Name (Node Number)"
nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=700)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

# Draw all edges in gray
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=2, edge_color="gray")

# Highlight shortest path edges in red
path_edges = list(zip(path, path[1:]))
nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=4, edge_color="red")

nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels={(u, v): d["Weight"] for u, v, d in G.edges(data=True)},
    font_color="green",
)

plt.title(
    f"NSFNET - Shortest Path {node_labels[source]} → {node_labels[target]}", fontsize=14
)
plt.tight_layout()
plt.axis("off")
plt.savefig("shortest_path.png")
plt.show()
