
"""
Shortest Path Visualizer for NSFNET
Calculates and plots the shortest path for the SCF demand using NetworkX.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

def main():
    print("=" * 60)
    print("SHORTEST PATH VISUALIZER")
    print("=" * 60)
    
    os.makedirs("results", exist_ok=True)

    # --- 1. Load NSFNET Topology ---
    print("Loading NSFNET topology...")
    try:
        nsfnet_df = pd.read_csv("NSFNET_Links.csv")
    except FileNotFoundError:
        print("Error: NSFNET_Links.csv not found!")
        return

    G = nx.Graph()
    for _, row in nsfnet_df.iterrows():
        G.add_edge(int(row["Source"]), int(row["Destination"]), 
                   weight=int(row["Weight"]), capacity=int(row["Capacity_Mbps"]))

    # --- 2. Load SCF Demand ---
    try:
        scf = pd.read_csv("results/scf_demands.csv")
        source = int(scf.loc[0, "source"])
        target = int(scf.loc[0, "destination"])
        demand = int(scf.loc[0, "demand_Mbps"])
    except (FileNotFoundError, IndexError):
        print("Error reading results/scf_demands.csv. Using defaults (1 -> 14).")
        source, target, demand = 1, 14, 100

    print(f"Calculating shortest path: Node {source} -> Node {target}")

    # --- 3. Compute Shortest Path ---
    try:
        path = nx.shortest_path(G, source=source, target=target, weight="weight")
        cost = nx.shortest_path_length(G, source=source, target=target, weight="weight")
        print(f"Path: {' -> '.join(map(str, path))}")
        print(f"Total Cost: {cost}")
    except nx.NetworkXNoPath:
        print("No path found!")
        return

    # --- 4. Visualization ---
    print("Generating customization...")
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 10))

    # Node colors
    node_colors = []
    for node in G.nodes():
        if node == source: node_colors.append("green")
        elif node == target: node_colors.append("red")
        elif node in path: node_colors.append("orange")
        else: node_colors.append("lightblue")

    nx.draw_networkx_nodes(G, pos, node_size=800, node_color=node_colors, edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_weight="bold")

    # Edges
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", width=2)
    
    # Path Edges
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=4)

    # Edge Labels
    edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Legend
    legend_elements = [
        Patch(facecolor="green", label="Source"),
        Patch(facecolor="red", label="Destination"),
        Patch(facecolor="orange", label="Path Node"),
        plt.Line2D([0], [0], color="red", lw=4, label="Shortest Path")
    ]
    plt.legend(handles=legend_elements, loc="upper left")
    
    plt.title(f"Shortest Path: {source} -> {target} | Cost: {cost}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("results/shortest_path.png")
    print("Saved results/shortest_path.png")

if __name__ == "__main__":
    main()
