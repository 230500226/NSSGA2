
"""
Visualization Module for NSFNET Assignment
Generates plots and diagrams based on results from min_cost_solver.py.
"""

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import json
import os
import numpy as np
from matplotlib.patches import Patch

def load_data():
    try:
        scf_links = pd.read_csv("results/scf_link_utilization.csv")
        mcf_links = pd.read_csv("results/mcf_link_utilization.csv")
        mcf_detailed = pd.read_csv("results/mcf_detailed_flows.csv")
        with open("results/metrics.json", "r") as f:
            metrics = json.load(f)
        nsfnet = pd.read_csv("NSFNET_Links.csv")
        return scf_links, mcf_links, mcf_detailed, metrics, nsfnet
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None

def plot_histogram(scf_links, mcf_links):
    print("Generating Figure 7: Histogram...")
    plt.style.use("seaborn-v0_8-deep")
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(0, 101, 10)
    
    ax.hist(mcf_links["Utilization"], bins=bins, alpha=0.7, label="MCF Utilisation", color="skyblue", edgecolor="black")
    ax.hist(scf_links["Utilization"], bins=bins, alpha=0.9, label="SCF Utilisation", color="salmon", edgecolor="black")
    
    ax.set_title("Figure 7: Histogram of Link Utilisation Values (SCF vs MCF)", fontsize=16)
    ax.set_xlabel("Link Utilisation (%)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_xticks(bins)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("results/figure7_utilisation_histogram.png")
    plt.close()

def plot_cost_comparison(metrics):
    print("Generating Figure 8: Cost Comparison...")
    scf_cost = metrics["scf"]["total_cost"]
    mcf_cost = metrics["mcf"]["total_cost"]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(["SCF", "MCF"], [scf_cost, mcf_cost], color=["salmon", "skyblue"])
    
    plt.title("Figure 8: Total Cost Comparison", fontsize=16)
    plt.ylabel("Total Routing Cost")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:,.2f}", va="bottom", ha="center")
        
    plt.tight_layout()
    plt.savefig("results/figure8_cost_comparison.png")
    plt.close()

def plot_summary(metrics):
    print("Generating Figure 9: Summary...")
    labels = ["Total Cost", "Avg Utilisation"]
    scf_vals = [metrics["scf"]["total_cost"], metrics["scf"]["avg_utilization"]]
    mcf_vals = [metrics["mcf"]["total_cost"], metrics["mcf"]["avg_utilization"]]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, scf_vals, width, label="SCF", color="salmon")
    rects2 = ax.bar(x + width/2, mcf_vals, width, label="MCF", color="skyblue")
    
    ax.set_title("Figure 9: Comparative Summary", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')
    
    plt.tight_layout()
    plt.savefig("results/figure9_comparative_summary.png")
    plt.close()

def plot_network_flows(nsfnet_df, scf_links, mcf_detailed, metrics):
    print("Generating Network Flow Diagrams...")
    G = nx.Graph()
    for _, row in nsfnet_df.iterrows():
        G.add_edge(row["Source"], row["Destination"], pos=None) # pos will be calc by layout

    pos = nx.spring_layout(G, seed=42)
    
    # --- SCF Plot ---
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color="lightgray")
    
    # Draw SCF flows
    scf_edges = []
    scf_widths = []
    for _, row in scf_links.iterrows():
        if row["Flow"] > 0.01:
            scf_edges.append((row["Source"], row["Target"]))
            scf_widths.append(max(2, row["Flow"] / 50)) # Scale width
            
    nx.draw_networkx_edges(G, pos, edgelist=scf_edges, width=scf_widths, edge_color="red", arrows=True)
    plt.title(f"SCF Flow (Cost: {metrics['scf']['total_cost']:.2f})")
    plt.savefig("results/scf_flow_diagram.png")
    plt.close()
    
    # --- MCF Detailed Plot ---
    plt.figure(figsize=(14, 12))
    nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", alpha=0.3)
    
    commodities = mcf_detailed["commodity"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(commodities)))
    
    for i, k in enumerate(commodities):
        comm_flows = mcf_detailed[mcf_detailed["commodity"] == k]
        edges = []
        widths = []
        for _, row in comm_flows.iterrows():
            edges.append((row["source"], row["target"]))
            widths.append(max(2, row["flow"] / 10))
            
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=widths, edge_color=[colors[i]], label=f"Comm {k}", arrows=True)
        
    plt.legend()
    plt.title(f"MCF Detailed Flows (Cost: {metrics['mcf']['total_cost']:.2f})")
    plt.savefig("results/mcf_flow_diagram_colored.png")
    plt.close()

def plot_link_utilization(scf_links, mcf_links):
    print("Generating Figure 6: Link Utilization Bar Charts...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14))
    
    # Subplot 1: MCF Utilization
    ax1.bar(mcf_links['Link'], mcf_links['Utilization'], color='blue')
    ax1.set_title('MCF Link Utilisation')
    ax1.set_ylabel('Utilization (%)')
    ax1.set_ylim(0, 110)
    ax1.axhline(y=100, color='red', linestyle='--', label='100% Capacity')
    ax1.tick_params(axis='x', rotation=90)
    ax1.legend()
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    
    # Subplot 2: SCF Utilization
    ax2.bar(scf_links['Link'], scf_links['Utilization'], color='orange')
    ax2.set_title('SCF Link Utilisation')
    ax2.set_ylabel('Utilization (%)')
    ax2.set_ylim(0, 110)
    max_scf_util = scf_links['Utilization'].max()
    ax2.axhline(y=max_scf_util, color='darkgreen', linestyle='--', label=f'Max Utilisation ({max_scf_util:.1f}%)')
    ax2.tick_params(axis='x', rotation=90)
    ax2.legend()
    ax2.grid(axis='y', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("results/figure6_link_utilisation.png")
    plt.close()

if __name__ == "__main__":
    scf, mcf, mcf_det, met, nsf = load_data()
    if scf is not None:
        plot_histogram(scf, mcf)
        plot_cost_comparison(met)
        plot_summary(met)
        plot_link_utilization(scf, mcf)
        plot_network_flows(nsf, scf, mcf_det, met)
