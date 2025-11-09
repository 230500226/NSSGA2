"""
GA2 Project: Min-Cost Flow Solver for NSFNET
Student: 230500226
Using PuLP (not Gurobi)
"""

import pulp
import pandas as pd
import networkx as nx
import numpy as np

# ============================================================
# STEP 1: Load NSFNET Topology
# ============================================================

# NSFNET Links from project brief (Appendix C)
links_data = [
    (1, 2, 45, 3),
    (1, 3, 45, 4),
    (1, 5, 45, 5),
    (2, 6, 45, 3),
    (3, 4, 45, 3),
    (3, 14, 45, 4),
    (4, 6, 45, 2),
    (4, 7, 45, 3),
    (5, 6, 45, 3),
    (5, 9, 45, 5),
    (6, 7, 45, 2),
    (6, 10, 45, 3),
    (7, 11, 45, 2),
    (8, 9, 45, 2),
    (8, 10, 45, 3),
    (9, 12, 45, 3),
    (10, 11, 45, 2),
    (11, 12, 45, 2),
    (11, 13, 45, 3),
    (12, 8, 45, 2),
    (13, 14, 45, 3),
]

# Build bidirectional network
G = nx.Graph()
for src, dst, cap, wt in links_data:
    G.add_edge(src, dst, capacity=cap, weight=wt)

nodes = list(G.nodes())
edges = list(G.edges())

print("=" * 60)
print("NSFNET TOPOLOGY LOADED")
print("=" * 60)
print(f"Nodes: {len(nodes)}")
print(f"Edges: {len(edges)}")
print(f"Total Capacity: {len(edges) * 45} Mbps")
print()

# ============================================================
# STEP 2: Load SCF and MCF Demand Data
# ============================================================

scf_df = pd.read_csv("scf_230500226.csv")
mcf_df = pd.read_csv("demands_230500226.csv")

scf_source = int(scf_df.loc[0, "source"])
scf_dest = int(scf_df.loc[0, "destination"])
scf_demand = float(scf_df.loc[0, "demand_Mbps"])

print("=" * 60)
print("SINGLE-COMMODITY FLOW (SCF) PROBLEM")
print("=" * 60)
print(f"Source: Node {scf_source}")
print(f"Destination: Node {scf_dest}")
print(f"Demand: {scf_demand} Mbps")
print()

# ============================================================
# STEP 3: Solve SCF Using Min-Cost Flow (PuLP)
# ============================================================


def solve_scf_min_cost(G, source, dest, demand, scaling_factor=1.0, verbose=False):
    """
    Solve Single-Commodity Min-Cost Flow using PuLP
    Returns the solution, flow_vars, scaled demand, status
    """
    scaled_demand = demand * scaling_factor

    prob = pulp.LpProblem("SCF_Min_Cost_Flow", pulp.LpMinimize)
    # Decision variables: flow in both directions
    flow_vars = {}
    for i, j in G.edges():
        flow_vars[(i, j)] = pulp.LpVariable(
            f"f_{i}_{j}", lowBound=0, upBound=G[i][j]["capacity"]
        )
        flow_vars[(j, i)] = pulp.LpVariable(
            f"f_{j}_{i}", lowBound=0, upBound=G[i][j]["capacity"]
        )

    # Objective: Minimize total cost
    cost_expr = []
    for i, j in G.edges():
        cost_expr.append(G[i][j]["weight"] * flow_vars[(i, j)])
        cost_expr.append(G[i][j]["weight"] * flow_vars[(j, i)])
    prob += pulp.lpSum(cost_expr), "Total_Cost"

    # Flow conservation at each node
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
    if verbose:
        print(f"Solver status: {status}")
    if status == "Optimal":
        return prob, flow_vars, scaled_demand, True
    else:
        return prob, None, scaled_demand, False


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
        break
    else:
        print(
            f"✗ Infeasible at scaling {scaling_factor:.3f}, trying {scaling_factor * 0.9:.3f}..."
        )
        scaling_factor *= 0.9
else:
    print("ERROR: Could not find feasible solution after scaling")
    flow_vars = None

print()

# ============================================================
# STEP 4: Display SCF Results (non-zero flows only)
# ============================================================

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
        # Only output non-zero flows (tolerance 0.01 Mbps)
        if flow_ij > 0.01 or flow_ji > 0.01:
            # Pick net direction and show only one direction
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
    df_scf.to_csv("SCF_result.csv", index=False)
    print("✓ Saved: SCF_result.csv")
    print()

# ============================================================
# STEP 5: Solve and display Multi-Commodity Flow (MCF)
# ============================================================

print("=" * 60)
print("MULTI-COMMODITY FLOW (MCF) PROBLEM")
print("=" * 60)
commodities = []
for idx, row in mcf_df.iterrows():
    k = idx + 1
    s = int(row["source"])
    t = int(row["destination"])
    d = float(row["demand_Mbps"])
    print(f"Commodity {k}: Node {s} → {t}, Demand = {d} Mbps")
    commodities.append({"k": k, "source": s, "dest": t, "demand": d})
print(f"Total Demand: {sum(c['demand'] for c in commodities)} Mbps")
print()


def solve_mcf_min_cost(G, commodities, scaling_factor=1.0, verbose=False):
    # Scale all demands
    scaled_commodities = [
        {**c, "demand": c["demand"] * scaling_factor} for c in commodities
    ]
    prob = pulp.LpProblem("MCF_Min_Cost_Flow", pulp.LpMinimize)
    # Decision variables for each commodity/direction
    flow_vars = {}
    for comm in scaled_commodities:
        k = comm["k"]
        for i, j in G.edges():
            flow_vars[(k, i, j)] = pulp.LpVariable(f"f_{k}_{i}_{j}", lowBound=0)
            flow_vars[(k, j, i)] = pulp.LpVariable(f"f_{k}_{j}_{i}", lowBound=0)
    # Objective: min total cost
    cost_expr = []
    for comm in scaled_commodities:
        k = comm["k"]
        for i, j in G.edges():
            wt = G[i][j]["weight"]
            cost_expr.append(wt * flow_vars[(k, i, j)])
            cost_expr.append(wt * flow_vars[(k, j, i)])
    prob += pulp.lpSum(cost_expr), "Total_Cost"
    # Flow conservation
    for comm in scaled_commodities:
        k = comm["k"]
        s = comm["source"]
        t = comm["dest"]
        d = comm["demand"]
        for node in G.nodes():
            if node == s:
                supply_val = d
            elif node == t:
                supply_val = -d
            else:
                supply_val = 0
            outg = [flow_vars[(k, node, neighbor)] for neighbor in G.neighbors(node)]
            incg = [flow_vars[(k, neighbor, node)] for neighbor in G.neighbors(node)]
            prob += (
                pulp.lpSum(outg) - pulp.lpSum(incg) == supply_val,
                f"FlowCons_k{k}_node{node}",
            )
    # Capacity: total flow on each link ≤ capacity
    for i, j in G.edges():
        total_ij = [flow_vars[(c["k"], i, j)] for c in scaled_commodities]
        total_ji = [flow_vars[(c["k"], j, i)] for c in scaled_commodities]
        cap = G[i][j]["capacity"]
        prob += pulp.lpSum(total_ij) <= cap, f"Cap_{i}_{j}"
        prob += pulp.lpSum(total_ji) <= cap, f"Cap_{j}_{i}"
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]
    if verbose:
        print(f"Solver status: {status}")
    if status == "Optimal":
        return prob, flow_vars, scaled_commodities, True
    else:
        return prob, None, scaled_commodities, False


print("Attempting to solve MCF...")
scaling_factor = 1.0
for iteration in range(max_iterations):
    prob, flow_vars, scaled_comms, is_feasible = solve_mcf_min_cost(
        G, commodities, scaling_factor
    )
    if is_feasible:
        print(f"✓ FEASIBLE at scaling factor: {scaling_factor:.3f}")
        print(
            f"  Total scaled demand: {sum(c['demand'] for c in scaled_comms):.1f} Mbps"
        )
        print(f"  Objective value (total cost): {pulp.value(prob.objective):.2f}")
        break
    else:
        print(
            f"✗ Infeasible at scaling {scaling_factor:.3f}, trying {scaling_factor * 0.9:.3f}..."
        )
        scaling_factor *= 0.9
else:
    print("ERROR: Could not find feasible solution for MCF")
    flow_vars = None

print()

# ============================================================
# STEP 6: Display MCF Results (non-zero flows only)
# ============================================================

if flow_vars:
    print("=" * 60)
    print("MCF SOLUTION - LINK UTILIZATION")
    print("=" * 60)
    link_results = []
    # Aggregate link flows over all five commodities
    for i, j in G.edges():
        capacity = G[i][j]["capacity"]
        weight = G[i][j]["weight"]
        # Sum flows across all commodities, both directions
        total_flow_ij = sum(pulp.value(flow_vars[(c["k"], i, j)]) for c in scaled_comms)
        total_flow_ji = sum(pulp.value(flow_vars[(c["k"], j, i)]) for c in scaled_comms)
        # Only output non-zero links (allowing for tolerance)
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
    df_mcf.to_csv("MCF_result.csv", index=False)
    df_mcf.to_csv("LinkUtilisation.csv", index=False)
    print("✓ Saved: MCF_result.csv")
    print("✓ Saved: LinkUtilisation.csv")
    print()

print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
