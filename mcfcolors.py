"""
GA2 Project: Min-Cost Flow Solver for NSFNET
Student: 230500226
MCF Flow Visualization: Highlight path flows for each commodity with distinct colors
"""

import pulp
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# ------------------------------------------------------------
# STEP 1: Load NSFNET Topology from CSV
# ------------------------------------------------------------

nsfnet_df = pd.read_csv("NSFNET_Links.csv")
G = nx.Graph()
for _, row in nsfnet_df.iterrows():
    src = int(row["Source"])
    dst = int(row["Destination"])
    cap = int(row["Capacity_Mbps"])
    wt = int(row["Weight"])
    G.add_edge(src, dst, capacity=cap, weight=wt)
nodes = list(G.nodes())
edges = list(G.edges())

# ------------------------------------------------------------
# STEP 2: Multi-Commodity Flow (MCF) Problem (from CSV)
# ------------------------------------------------------------

mcf_df = pd.read_csv("demands_230500226.csv")
mcf_commodities = []
for idx, row in mcf_df.iterrows():
    mcf_commodities.append(
        {
            "k": int(row["commodity"]),
            "source": int(row["source"]),
            "dest": int(row["destination"]),
            "demand": int(row["demand_Mbps"]),
        }
    )

def solve_mcf_min_cost(G, commodities, scaling_factor=1.0):
    scaled_commodities = [
        {**c, "demand": c["demand"] * scaling_factor} for c in commodities
    ]
    prob = pulp.LpProblem("MCF_Min_Cost_Flow", pulp.LpMinimize)
    flow_vars = {}
    for comm in scaled_commodities:
        k = comm["k"]
        for i, j in G.edges():
            flow_vars[(k, i, j)] = pulp.LpVariable(f"f_{k}_{i}_{j}", lowBound=0)
            flow_vars[(k, j, i)] = pulp.LpVariable(f"f_{k}_{j}_{i}", lowBound=0)
    cost_expr = []
    for comm in scaled_commodities:
        k = comm["k"]
        for i, j in G.edges():
            weight = G[i][j]["weight"]
            cost_expr.append(weight * flow_vars[(k, i, j)])
            cost_expr.append(weight * flow_vars[(k, j, i)])
    prob += pulp.lpSum(cost_expr), "Total_Cost"
    for comm in scaled_commodities:
        k = comm["k"]
        source = comm["source"]
        dest = comm["dest"]
        demand = comm["demand"]
        for node in G.nodes():
            if node == source:
                supply_val = demand
            elif node == dest:
                supply_val = -demand
            else:
                supply_val = 0
            outgoing = [flow_vars[(k, node, neighbor)] for neighbor in G.neighbors(node)]
            incoming = [flow_vars[(k, neighbor, node)] for neighbor in G.neighbors(node)]
            prob += (pulp.lpSum(outgoing) - pulp.lpSum(incoming) == supply_val,
                     f"FlowConservation_Commodity_{k}_Node_{node}")
    for i, j in G.edges():
        total_flow_ij = [flow_vars[(comm["k"], i, j)] for comm in scaled_commodities]
        total_flow_ji = [flow_vars[(comm["k"], j, i)] for comm in scaled_commodities]
        capacity = G[i][j]["capacity"]
        prob += pulp.lpSum(total_flow_ij) <= capacity, f"Capacity_{i}_{j}"
        prob += pulp.lpSum(total_flow_ji) <= capacity, f"Capacity_{j}_{i}"
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]
    if status == "Optimal":
        return prob, flow_vars, scaled_commodities, True
    else:
        return prob, None, scaled_commodities, False

# ------------------------------------------------------------
# STEP 3: Solve MCF (use scaling if needed)
# ------------------------------------------------------------

scaling_factor = 1.0
max_iterations = 20
for iteration in range(max_iterations):
    prob_mcf, flow_vars_mcf, scaled_comms, is_feasible = solve_mcf_min_cost(
        G, mcf_commodities, scaling_factor
    )
    if is_feasible:
        print(f"✓ FEASIBLE at scaling factor: {scaling_factor:.3f}")
        print(
            f"  Total scaled demand: {sum(c['demand'] for c in scaled_comms):.1f} Mbps"
        )
        print(f"  Objective value (total cost): {pulp.value(prob_mcf.objective):.2f}")
        mcf_scaling = scaling_factor
        break
    else:
        print(
            f"✗ Infeasible at scaling {scaling_factor:.3f}, trying {scaling_factor * 0.9:.3f}..."
        )
        scaling_factor *= 0.9
else:
    print("ERROR: Could not find feasible solution for MCF")
    flow_vars_mcf = None
    mcf_scaling = 0

# ------------------------------------------------------------
# STEP 4: MCF Flow Visualization -- Each Commodity in Its Own Color
# ------------------------------------------------------------

if flow_vars_mcf:
    print("Generating MCF flow visualization (commodity-colored)...")
    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)

    # Color map for commodities
    base_colors = [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"
    ]
    # If more commodities, extend with random colors
    if len(scaled_comms) > len(base_colors):
        import random
        for _ in range(len(scaled_comms) - len(base_colors)):
            # pastel random colors
            r,g,b = np.random.uniform(0.3,0.9,3)
            base_colors.append((r, g, b))

    node_colors = []
    for node in G.nodes():
        if node in set(c["source"] for c in scaled_comms) and node in set(c["dest"] for c in scaled_comms):
            node_colors.append("purple")
        elif node in set(c["source"] for c in scaled_comms):
            node_colors.append("green")
        elif node in set(c["dest"] for c in scaled_comms):
            node_colors.append("red")
        else:
            node_colors.append("lightblue")

    nx.draw_networkx_nodes(
        G, pos, node_size=900, node_color=node_colors,
        edgecolors="black", linewidths=2, ax=ax
    )
    labels = {i: str(i) for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=3, edge_color="lightgray", alpha=0.2, ax=ax)

    legend_elements = [
        Patch(facecolor="green", edgecolor="black", label="Source Nodes"),
        Patch(facecolor="red", edgecolor="black", label="Destination Nodes"),
        Patch(facecolor="purple", edgecolor="black", label="Source & Dest"),
        Patch(facecolor="lightblue", edgecolor="black", label="Intermediate Nodes"),
    ]
    for idx, comm in enumerate(scaled_comms):
        color = base_colors[idx % len(base_colors)]
        legend_elements.append(plt.Line2D([0],[0], color=color, linewidth=4,
                                          label=f"Commodity {comm['k']} {comm['source']}→{comm['dest']}"))

    # Draw flows by commodity, using different color for each
    for idx, comm in enumerate(scaled_comms):
        k = comm["k"]
        color = base_colors[idx % len(base_colors)]
        flow_edges = []
        edge_widths = []
        edge_labels = {}

        for i, j in G.edges():
            flow_ij = pulp.value(flow_vars_mcf[(k, i, j)])
            flow_ji = pulp.value(flow_vars_mcf[(k, j, i)])

            if flow_ij > 0.01:
                flow_edges.append((i, j))
                edge_widths.append(max(2, flow_ij / 5))
                edge_labels[(i, j)] = f"{flow_ij:.1f}"
            if flow_ji > 0.01:
                flow_edges.append((j, i))
                edge_widths.append(max(2, flow_ji / 5))
                edge_labels[(j, i)] = f"{flow_ji:.1f}"

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=flow_edges,
            width=edge_widths,
            edge_color=color,
            alpha=0.8,
            arrows=True,
            arrowsize=16,
            arrowstyle="-|>",
            ax=ax,
        )

        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=10,
            font_color=color if isinstance(color, str) else "black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.4),
            ax=ax,
        )

    ax.legend(handles=legend_elements, loc="upper left", fontsize=11, framealpha=0.93)

    commodity_text = " | ".join(
        [f"K{c['k']}({c['source']}→{c['dest']})" for c in scaled_comms]
    )
    plt.title(
        f"Multi-Commodity Flow (MCF) Solution Paths\nEach path colored per commodity\n"
        f"{len(scaled_comms)} Commodities | Total Demand: {sum(c['demand'] for c in scaled_comms):.1f} Mbps | "
        f"Scaling: {mcf_scaling:.3f}\n{commodity_text}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("MCF_flow_diagram_multicolor.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: MCF_flow_diagram_multicolor.png")
    plt.close()
