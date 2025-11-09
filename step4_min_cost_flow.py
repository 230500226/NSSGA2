import pandas as pd      # Used for loading input data and saving output results
import networkx as nx    # Used for representing the network as a directed graph
import pulp              # Linear programming package used for optimization
import numpy as np       # Used for numerical operations, but not directly used here

# --- 1. Build NSFNET Graph ---
# The network topology is encoded as a directed graph (DiGraph).
# Each tuple: (source_node, destination_node, link_weight)
# Both directions of each link are listed to allow for asymmetric routing.

G = nx.DiGraph()  # Directed graph allows us to model flows on arcs with direction

edges = [
    # Each pair (u, v, w) represents an arc from node u to node v with cost 'w' per unit flow
    (1, 2, 3),
    (2, 1, 3),
    (1, 3, 4),
    (3, 1, 4),
    (1, 5, 5),
    (5, 1, 5),
    (2, 6, 3),
    (6, 2, 3),
    (3, 4, 3),
    (4, 3, 3),
    (3, 14, 4),
    (14, 3, 4),
    (4, 6, 2),
    (6, 4, 2),
    (4, 7, 3),
    (7, 4, 3),
    (5, 6, 3),
    (6, 5, 3),
    (5, 9, 5),
    (9, 5, 5),
    (6, 7, 2),
    (7, 6, 2),
    (6, 10, 3),
    (10, 6, 3),
    (7, 11, 2),
    (11, 7, 2),
    (8, 9, 2),
    (9, 8, 2),
    (8, 10, 3),
    (10, 8, 3),
    (9, 12, 3),
    (12, 9, 3),
    (10, 11, 2),
    (11, 10, 2),
    (11, 12, 2),
    (12, 11, 2),
    (11, 13, 3),
    (13, 11, 3),
    (12, 8, 2),
    (8, 12, 2),
    (13, 14, 3),
    (14, 13, 3),
]

# Add each edge into the network graph with the specified capacity and cost
for u, v, w in edges:
    G.add_edge(
        u, v, capacity=45, weight=w
    )
    # capacity: maximum flow allowed through this link (45 Mbps for all links per NSFNET)
    # weight: cost per unit flow for this link

# --- 2. Load Demands ---
# Load SCF and MCF demand instances from CSV files.
# These files contain the source, destination, and required demand for each instance.

scf = pd.read_csv("scf_230500226.csv")
# SCF CSV: Should have at least columns 'source', 'destination', 'demand_Mbps'

mcf = pd.read_csv("demands_230500226.csv")
# MCF CSV: Should have columns 'source', 'destination', 'demand_Mbps' for each commodity

# --- 3. SCF - Min Cost Flow ---
def run_scf(G, scf):
    """
    Solves the Single-Commodity Min-Cost Flow problem on graph G
    using source, destination, and demand from the SCF CSV file.

    Returns:
        scf_result: DataFrame containing flow values for each arc
    """

    prob = pulp.LpProblem("SCF", pulp.LpMinimize)  # Create a minimization LP

    # Variables: flow on each arc (directed edge)
    flow_vars = {}
    for u, v in G.edges():
        # For each directed edge, create a non-negative flow variable
        flow_vars[u, v] = pulp.LpVariable(f"f_{u}_{v}", lowBound=0)

    # Objective: minimize total flow cost across all arcs
    prob += pulp.lpSum([G[u][v]["weight"] * flow_vars[u, v] for u, v in G.edges()])

    # --- Flow Conservation Constraints ---
    # Enforce that at every node, total flow out - flow in matches demand
    s = int(scf.loc[0, "source"])         # source node
    t = int(scf.loc[0, "destination"])    # sink/destination node
    d = float(scf.loc[0, "demand_Mbps"])  # required amount to send

    for n in G.nodes():
        in_flow = pulp.lpSum([flow_vars[u, n] for u in G.predecessors(n)])    # incoming flow to node n
        out_flow = pulp.lpSum([flow_vars[n, v] for v in G.successors(n)])     # outgoing flow from node n
        if n == s:
            # Source: net outgoing flow equals total demand
            prob += out_flow - in_flow == d
        elif n == t:
            # Sink: net incoming flow equals total demand
            prob += in_flow - out_flow == d
        else:
            # All others: net flow balanced (no supply/demand)
            prob += out_flow - in_flow == 0

    # --- Capacity Constraints ---
    # Flow on each arc cannot exceed its capacity
    for u, v in G.edges():
        prob += flow_vars[u, v] <= G[u][v]["capacity"]

    # --- Solve the Problem ---
    prob.solve()
    # pulp.value(prob.objective) gives total minimum cost

    # --- Extract Results ---
    result = []
    for u, v in G.edges():
        result.append({"source": u, "destination": v, "flow": flow_vars[u, v].varValue})
    scf_result = pd.DataFrame(result)
    # Save results so they can be analyzed or plotted
    scf_result.to_csv("SCF_result.csv", index=False)
    return scf_result


scf_result_df = run_scf(G, scf)  # Run SCF and get result table

# --- 4. MCF - Min Cost Flow for 5 commodities ---
def run_mcf(G, mcf):
    """
    Solves the Multi-Commodity Min-Cost Flow problem.
    Each row in mcf contains source, destination, and the demand for a commodity.

    Returns:
        mcf_result: DataFrame with all commodity flows per arc
        link_util_df: DataFrame with per-link utilization (sum of all flows / capacity)
    """
    prob = pulp.LpProblem("MCF", pulp.LpMinimize)  # Create minimization LP

    K = mcf.shape[0]  # Number of commodities (rows in the CSV)
    flow_vars = {}

    # --- Variables: flow for every commodity on every arc ---
    for k in range(K):
        for u, v in G.edges():
            # Each commodity's flow on each directed edge is independent
            flow_vars[k, u, v] = pulp.LpVariable(f"f_{k}_{u}_{v}", lowBound=0)

    # --- Objective: total cost over all commodities and arcs ---
    prob += pulp.lpSum(
        [G[u][v]["weight"] * flow_vars[k, u, v] for k in range(K) for u, v in G.edges()]
    )

    # --- Flow Conservation Constraints for Each Commodity ---
    for k in range(K):
        s = int(mcf.loc[k, "source"])
        t = int(mcf.loc[k, "destination"])
        d = float(mcf.loc[k, "demand_Mbps"])
        for n in G.nodes():
            in_flow = pulp.lpSum([flow_vars[k, u, n] for u in G.predecessors(n)])
            out_flow = pulp.lpSum([flow_vars[k, n, v] for v in G.successors(n)])
            if n == s:
                prob += out_flow - in_flow == d     # Source: net out-flow = demand
            elif n == t:
                prob += in_flow - out_flow == d     # Sink: net in-flow = demand
            else:
                prob += out_flow - in_flow == 0     # Others: balanced

    # --- Capacity Constraints: sum of flows ≤ capacity ---
    for u, v in G.edges():
        # For each directed edge, sum over all commodities' flows cannot exceed capacity
        prob += (
            pulp.lpSum([flow_vars[k, u, v] for k in range(K)]) <= G[u][v]["capacity"]
        )

    # --- Solve the Problem ---
    prob.solve()
    # pulp.value(prob.objective) gives minimal network cost

    # --- Extract Per-Commodity Flows for Each Arc ---
    result = []
    for k in range(K):
        for u, v in G.edges():
            result.append(
                {
                    "commodity": k + 1,                 # 1-based commodity ID
                    "source": u,
                    "destination": v,
                    "flow": flow_vars[k, u, v].varValue,
                }
            )
    mcf_result = pd.DataFrame(result)
    mcf_result.to_csv("MCF_result.csv", index=False)

    # --- Compute Link Utilisation ---
    # For each arc, sum all flows across commodities; compare to capacity
    link_util = []
    for u, v in G.edges():
        total_flow = sum([flow_vars[k, u, v].varValue for k in range(K)])
        utilisation = total_flow / G[u][v]["capacity"]
        link_util.append(
            {
                "source": u,
                "destination": v,
                "total_flow": total_flow,
                "utilisation": utilisation,    # as a fraction of capacity
            }
        )
    link_util_df = pd.DataFrame(link_util)
    link_util_df.to_csv("LinkUtilisation.csv", index=False)
    return mcf_result, link_util_df


mcf_result_df, link_util_df = run_mcf(G, mcf)  # Run MCF and get results
