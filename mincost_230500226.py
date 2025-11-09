"""
GA2 Project: Min-Cost Flow Solver for NSFNET
Student: 230500226
Using PuLP (not Gurobi)
"""

import pulp
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ============================================================
# STEP 1: Load NSFNET Topology from CSV
# ============================================================

print("=" * 60)
print("LOADING NSFNET TOPOLOGY FROM CSV")
print("=" * 60)

# Read NSFNET links from CSV file
nsfnet_df = pd.read_csv("NSFNET_Links.csv")
print(f"Loaded {len(nsfnet_df)} edges from NSFNET_Links.csv")
print()

# Create bidirectional graph from CSV
G = nx.Graph()
for _, row in nsfnet_df.iterrows():
    src = int(row["Source"])
    dst = int(row["Destination"])
    cap = int(row["Capacity_Mbps"])
    wt = int(row["Weight"])
    G.add_edge(src, dst, capacity=cap, weight=wt)

nodes = list(G.nodes())
edges = list(G.edges())

print("=" * 60)
print("NSFNET TOPOLOGY LOADED")
print("=" * 60)
print(f"Nodes: {len(nodes)}")
print(f"Edges: {len(edges)}")
print()

# ============================================================
# STEP 2: Your SCF Problem Data (from CSV)
# ============================================================

scf_df = pd.read_csv("scf_230500226.csv")
scf_source = int(scf_df.loc[0, "source"])
scf_dest = int(scf_df.loc[0, "destination"])
scf_demand = int(scf_df.loc[0, "demand_Mbps"])

print("=" * 60)
print("SINGLE-COMMODITY FLOW (SCF) PROBLEM")
print("=" * 60)
print(f"Source: Node {scf_source}")
print(f"Destination: Node {scf_dest}")
print(f"Demand: {scf_demand} Mbps")
print()

# ============================================================
# STEP 3: Solve SCF using Min-Cost Flow (PuLP)
# ============================================================


def solve_scf_min_cost(G, source, dest, demand, scaling_factor=1.0):
    """
    Solve Single-Commodity Min-Cost Flow using PuLP
    """
    scaled_demand = demand * scaling_factor

    prob = pulp.LpProblem("SCF_Min_Cost_Flow", pulp.LpMinimize)

    flow_vars = {}
    for i, j in G.edges():
        flow_vars[(i, j)] = pulp.LpVariable(
            f"f_{i}_{j}", lowBound=0, upBound=G[i][j]["capacity"]
        )
        flow_vars[(j, i)] = pulp.LpVariable(
            f"f_{j}_{i}", lowBound=0, upBound=G[i][j]["capacity"]
        )

    cost_expr = []
    for i, j in G.edges():
        weight = G[i][j]["weight"]
        cost_expr.append(weight * flow_vars[(i, j)])
        cost_expr.append(weight * flow_vars[(j, i)])

    prob += pulp.lpSum(cost_expr), "Total_Cost"

    for node in G.nodes():
        if node == source:
            supply_val = scaled_demand
        elif node == dest:
            supply_val = -scaled_demand
        else:
            supply_val = 0

        outgoing = []
        incoming = []

        for neighbor in G.neighbors(node):
            outgoing.append(flow_vars[(node, neighbor)])
            incoming.append(flow_vars[(neighbor, node)])

        prob += (
            pulp.lpSum(outgoing) - pulp.lpSum(incoming) == supply_val,
            f"FlowConservation_Node_{node}",
        )

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]

    if status == "Optimal":
        return prob, flow_vars, scaled_demand, True
    else:
        return prob, None, scaled_demand, False


# Try solving with scaling if needed
print("Attempting to solve SCF with original demand...")
scaling_factor = 1.0
max_iterations = 20

for iteration in range(max_iterations):
    prob, flow_vars, scaled_demand, is_feasible = solve_scf_min_cost(
        G, scf_source, scf_dest, scf_demand, scaling_factor
    )

    if is_feasible:
        print(f"✓ FEASIBLE at scaling factor: {scaling_factor:.3f}")
        print(f"  Scaled demand: {scaled_demand:.1f} Mbps")
        print(f"  Objective value (total cost): {pulp.value(prob.objective):.2f}")
        scf_scaling = scaling_factor
        break
    else:
        print(
            f"✗ Infeasible at scaling {scaling_factor:.3f}, trying {scaling_factor * 0.9:.3f}..."
        )
        scaling_factor *= 0.9
else:
    print("ERROR: Could not find feasible solution after scaling")
    flow_vars = None
    scf_scaling = 0

print()

# ============================================================
# STEP 4: Display SCF Results
# ============================================================

scf_flow_data = {}  # Store for visualization

if flow_vars:
    print("=" * 60)
    print("SCF SOLUTION - LINK UTILIZATION")
    print("=" * 60)

    results = []
    total_flow_used = 0

    for i, j in G.edges():
        flow_ij = pulp.value(flow_vars[(i, j)])
        flow_ji = pulp.value(flow_vars[(j, i)])
        capacity = G[i][j]["capacity"]
        weight = G[i][j]["weight"]

        scf_flow_data[(i, j)] = flow_ij
        scf_flow_data[(j, i)] = flow_ji

        if flow_ij > 0.01 or flow_ji > 0.01:
            if flow_ij > flow_ji:
                net_flow = flow_ij
                direction = f"{i}→{j}"
            else:
                net_flow = flow_ji
                direction = f"{j}→{i}"

            utilization = (net_flow / capacity) * 100

            results.append(
                {
                    "Link": direction,
                    "Flow (Mbps)": net_flow,
                    "Capacity (Mbps)": capacity,
                    "Utilization (%)": utilization,
                    "Cost": net_flow * weight,
                }
            )

            total_flow_used += net_flow * weight

    df_scf = pd.DataFrame(results)
    print(df_scf.to_string(index=False))
    print()
    print(f"Total Cost: {total_flow_used:.2f}")
    print(f"Average Utilization: {df_scf['Utilization (%)'].mean():.2f}%")
    print(f"Max Utilization: {df_scf['Utilization (%)'].max():.2f}%")
    print()

    # Save to CSV
    df_scf.to_csv("SCF_result.csv", index=False)
    print("✓ Saved: SCF_result.csv")
    print()

# ============================================================
# STEP 5: Multi-Commodity Flow (MCF) Problem (from CSV)
# ============================================================

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

print("=" * 60)
print("MULTI-COMMODITY FLOW (MCF) PROBLEM")
print("=" * 60)
for comm in mcf_commodities:
    print(
        f"Commodity {comm['k']}: Node {comm['source']} → {comm['dest']}, Demand = {comm['demand']} Mbps"
    )
print(f"Total Demand: {sum(c['demand'] for c in mcf_commodities)} Mbps")
print()


def solve_mcf_min_cost(G, commodities, scaling_factor=1.0):
    """
    Solve Multi-Commodity Min-Cost Flow using PuLP
    """
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

            outgoing = [
                flow_vars[(k, node, neighbor)] for neighbor in G.neighbors(node)
            ]
            incoming = [
                flow_vars[(k, neighbor, node)] for neighbor in G.neighbors(node)
            ]

            prob += (
                pulp.lpSum(outgoing) - pulp.lpSum(incoming) == supply_val,
                f"FlowConservation_Commodity_{k}_Node_{node}",
            )

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


print("Attempting to solve MCF...")
scaling_factor = 1.0

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

print()

# ============================================================
# STEP 6: Display MCF Results
# ============================================================

mcf_flow_data = {}  # Store for visualization

if flow_vars_mcf:
    print("=" * 60)
    print("MCF SOLUTION - LINK UTILIZATION")
    print("=" * 60)

    link_results = []

    for i, j in G.edges():
        capacity = G[i][j]["capacity"]
        weight = G[i][j]["weight"]

        total_flow_ij = sum(
            pulp.value(flow_vars_mcf[(c["k"], i, j)]) for c in scaled_comms
        )
        total_flow_ji = sum(
            pulp.value(flow_vars_mcf[(c["k"], j, i)]) for c in scaled_comms
        )

        mcf_flow_data[(i, j)] = total_flow_ij
        mcf_flow_data[(j, i)] = total_flow_ji

        if total_flow_ij > 0.01 or total_flow_ji > 0.01:
            net_flow = max(total_flow_ij, total_flow_ji)
            utilization = (net_flow / capacity) * 100

            link_results.append(
                {
                    "Link": f"{i}↔{j}",
                    "Flow (Mbps)": net_flow,
                    "Capacity (Mbps)": capacity,
                    "Utilization (%)": utilization,
                }
            )

    df_mcf = pd.DataFrame(link_results)
    print(df_mcf.to_string(index=False))
    print()
    print(f"Average Utilization: {df_mcf['Utilization (%)'].mean():.2f}%")
    print(f"Max Utilization: {df_mcf['Utilization (%)'].max():.2f}%")
    print()

    # Save to CSV
    df_mcf.to_csv("MCF_result.csv", index=False)
    df_mcf.to_csv("LinkUtilisation.csv", index=False)
    print("✓ Saved: MCF_result.csv")
    print("✓ Saved: LinkUtilisation.csv")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

# ============================================================
# STEP 7: VISUALIZATION - SCF Flow Diagram
# ============================================================

if flow_vars and scf_flow_data:
    print("\nGenerating SCF visualization...")

    fig, ax = plt.subplots(figsize=(16, 12))

    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)

    node_colors = []
    for node in G.nodes():
        if node == scf_source:
            node_colors.append("green")
        elif node == scf_dest:
            node_colors.append("red")
        else:
            node_colors.append("lightblue")

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=900,
        node_color=node_colors,
        edgecolors="black",
        linewidths=2,
        ax=ax,
    )

    labels = {i: str(i) for i in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels=labels, font_size=12, font_weight="bold", ax=ax
    )

    nx.draw_networkx_edges(
        G, pos, edgelist=G.edges(), width=3, edge_color="black", alpha=0.3, ax=ax
    )

    flow_edges = []
    edge_widths = []
    edge_labels = {}

    for i, j in G.edges():
        flow_ij = scf_flow_data.get((i, j), 0)
        flow_ji = scf_flow_data.get((j, i), 0)

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
        edge_color="red",
        alpha=0.7,
        arrows=True,
        arrowsize=20,
        arrowstyle="->",
        ax=ax,
    )

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        font_color="darkred",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        ax=ax,
    )

    legend_elements = [
        Patch(facecolor="green", edgecolor="black", label=f"Source: Node {scf_source}"),
        Patch(
            facecolor="red", edgecolor="black", label=f"Destination: Node {scf_dest}"
        ),
        Patch(facecolor="lightblue", edgecolor="black", label="Other Nodes"),
        plt.Line2D(
            [0], [0], color="red", linewidth=4, label="Flow Path (thickness ∝ flow)"
        ),
        plt.Line2D([0], [0], color="lightgray", linewidth=2, label="Unused Links"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=11, framealpha=0.95)

    plt.title(
        f"Single-Commodity Flow (SCF) Solution\n"
        f"Node {scf_source} → Node {scf_dest} | Demand: {scaled_demand:.1f} Mbps | "
        f"Scaling: {scf_scaling:.3f} | Total Cost: {total_flow_used:.2f}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig("SCF_flow_diagram.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: SCF_flow_diagram.png")
    plt.close()

# ============================================================
# STEP 8: VISUALIZATION - MCF Flow Diagram
# ============================================================

if flow_vars_mcf and mcf_flow_data:
    print("Generating MCF visualization...")

    fig, ax = plt.subplots(figsize=(16, 12))

    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)

    mcf_sources = set(c["source"] for c in scaled_comms)
    mcf_dests = set(c["dest"] for c in scaled_comms)

    node_colors = []
    for node in G.nodes():
        if node in mcf_sources and node in mcf_dests:
            node_colors.append("purple")
        elif node in mcf_sources:
            node_colors.append("green")
        elif node in mcf_dests:
            node_colors.append("red")
        else:
            node_colors.append("lightblue")

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=900,
        node_color=node_colors,
        edgecolors="black",
        linewidths=2,
        ax=ax,
    )

    labels = {i: str(i) for i in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels=labels, font_size=12, font_weight="bold", ax=ax
    )

    nx.draw_networkx_edges(
        G, pos, edgelist=G.edges(), width=3, edge_color="gray", alpha=0.3, ax=ax
    )

    flow_edges = []
    edge_widths = []
    edge_labels = {}

    for i, j in G.edges():
        flow_ij = mcf_flow_data.get((i, j), 0)
        flow_ji = mcf_flow_data.get((j, i), 0)

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
        edge_color="blue",
        alpha=0.6,
        arrows=True,
        arrowsize=20,
        arrowstyle="->",
        ax=ax,
    )

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        font_color="darkblue",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        ax=ax,
    )

    legend_elements = [
        Patch(facecolor="green", edgecolor="black", label="Source Nodes"),
        Patch(facecolor="red", edgecolor="black", label="Destination Nodes"),
        Patch(facecolor="purple", edgecolor="black", label="Source & Dest"),
        Patch(facecolor="lightblue", edgecolor="black", label="Intermediate Nodes"),
        plt.Line2D(
            [0],
            [0],
            color="blue",
            linewidth=4,
            label="Aggregate Flow (thickness ∝ flow)",
        ),
        plt.Line2D([0], [0], color="lightgray", linewidth=2, label="Unused Links"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=11, framealpha=0.95)

    commodity_text = " | ".join(
        [f"K{c['k']}({c['source']}→{c['dest']})" for c in scaled_comms]
    )
    plt.title(
        f"Multi-Commodity Flow (MCF) Solution\n"
        f"{len(scaled_comms)} Commodities | Total Demand: {sum(c['demand'] for c in scaled_comms):.1f} Mbps | "
        f"Scaling: {mcf_scaling:.3f}\n{commodity_text}",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig("MCF_flow_diagram.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: MCF_flow_diagram.png")
    plt.close()

print("\n" + "=" * 60)
print("ALL VISUALIZATIONS COMPLETE")
print("=" * 60)
print("Generated files:")
print("  • SCF_result.csv")
print("  • MCF_result.csv")
print("  • LinkUtilisation.csv")
print("  • SCF_flow_diagram.png")
print("  • MCF_flow_diagram.png")
