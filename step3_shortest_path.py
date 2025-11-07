import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# --- 1. Define NSFNET Topology (Nodes & Edges w/ Weights) ---
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

# --- 2. Load SCF CSV (Source & Dest Nodes) ---
scf = pd.read_csv("scf_230500226.csv")
source = int(scf.loc[0, "source"])
target = int(scf.loc[0, "destination"])
print(f"Source node: {source}, Destination node: {target}")

# --- 3. Compute Shortest Path (Dijkstra) ---
path = nx.shortest_path(G, source=source, target=target, weight="Weight")
cost = nx.shortest_path_length(G, source=source, target=target, weight="Weight")
print("Shortest path sequence:", path)
print("Total path cost (sum of weights):", cost)

# --- 4. Visualize and Highlight Path ---
city_labels = {
    1: "San Francisco",
    2: "Palo Alto",
    3: "Los Angeles",
    4: "Denver",
    5: "Houston",
    6: "Chicago",
    7: "Ann Arbor",
    8: "Pittsburgh",
    9: "Ithaca",
    10: "New York",
    11: "Princeton",
    12: "College Park",
    13: "Atlanta",
    14: "San Diego",
}
pos = nx.spring_layout(
    G, seed=42
)  # You can manually place nodes later for a custom map

plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue")
nx.draw_networkx_labels(G, pos, labels=city_labels, font_size=9)

# Draw all edges in grey
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=2, edge_color="gray")
# Highlight edges in path in red
path_edges = list(zip(path, path[1:]))
nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=4, edge_color="red")
nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels={(u, v): d["Weight"] for u, v, d in G.edges(data=True)},
    font_color="green",
)
plt.title(
    f"NSFNET - Shortest Path {city_labels[source]} ({source}) → {city_labels[target]} ({target})",
    fontsize=14,
)
plt.tight_layout()
plt.axis("off")
plt.savefig("shortest_path.png")
plt.show()
