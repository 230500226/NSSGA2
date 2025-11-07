import pandas as pd
import networkx as nx
import pulp
import numpy as np

# --- 1. Build NSFNET Graph ---
G = nx.DiGraph()  # Use directed for more flexibility
edges = [
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
for u, v, w in edges:
    G.add_edge(
        u, v, capacity=45, weight=w
    )  # Use 45 Mbps as per NSFNET, update if needed.

# --- 2. Load Demands ---
scf = pd.read_csv("scf_230500226.csv")
mcf = pd.read_csv("demands_230500226.csv")


# --- 3. SCF - Min Cost Flow ---
def run_scf(G, scf):
    prob = pulp.LpProblem("SCF", pulp.LpMinimize)
    # Variables: flow on each arc
    flow_vars = {}
    for u, v in G.edges():
        flow_vars[u, v] = pulp.LpVariable(f"f_{u}_{v}", lowBound=0)

    # Objective: minimize total cost
    prob += pulp.lpSum([G[u][v]["weight"] * flow_vars[u, v] for u, v in G.edges()])

    # Flow conservation
    s = int(scf.loc[0, "source"])
    t = int(scf.loc[0, "destination"])
    d = float(scf.loc[0, "demand_Mbps"])
    for n in G.nodes():
        in_flow = pulp.lpSum([flow_vars[u, n] for u in G.predecessors(n)])
        out_flow = pulp.lpSum([flow_vars[n, v] for v in G.successors(n)])
        if n == s:
            prob += out_flow - in_flow == d
        elif n == t:
            prob += in_flow - out_flow == d
        else:
            prob += out_flow - in_flow == 0

    # Capacity constraints
    for u, v in G.edges():
        prob += flow_vars[u, v] <= G[u][v]["capacity"]

    prob.solve()
    # Get flows for solution
    result = []
    for u, v in G.edges():
        result.append({"source": u, "destination": v, "flow": flow_vars[u, v].varValue})
    scf_result = pd.DataFrame(result)
    scf_result.to_csv("SCF_result.csv", index=False)
    return scf_result


scf_result_df = run_scf(G, scf)


# --- 4. MCF - Min Cost Flow for 5 commodities ---
def run_mcf(G, mcf):
    prob = pulp.LpProblem("MCF", pulp.LpMinimize)
    K = mcf.shape[0]  # number of commodities
    flow_vars = {}
    # Create var for each commodity and edge
    for k in range(K):
        for u, v in G.edges():
            flow_vars[k, u, v] = pulp.LpVariable(f"f_{k}_{u}_{v}", lowBound=0)

    # Objective: minimize sum of all commodity costs
    prob += pulp.lpSum(
        [G[u][v]["weight"] * flow_vars[k, u, v] for k in range(K) for u, v in G.edges()]
    )

    # Flow conservation for each commodity
    for k in range(K):
        s = int(mcf.loc[k, "source"])
        t = int(mcf.loc[k, "destination"])
        d = float(mcf.loc[k, "demand_Mbps"])
        for n in G.nodes():
            in_flow = pulp.lpSum([flow_vars[k, u, n] for u in G.predecessors(n)])
            out_flow = pulp.lpSum([flow_vars[k, n, v] for v in G.successors(n)])
            if n == s:
                prob += out_flow - in_flow == d
            elif n == t:
                prob += in_flow - out_flow == d
            else:
                prob += out_flow - in_flow == 0

    # Capacity constraints: sum of all commodities <= capacity per edge
    for u, v in G.edges():
        prob += (
            pulp.lpSum([flow_vars[k, u, v] for k in range(K)]) <= G[u][v]["capacity"]
        )

    prob.solve()
    # Get flows for solution
    result = []
    for k in range(K):
        for u, v in G.edges():
            result.append(
                {
                    "commodity": k + 1,
                    "source": u,
                    "destination": v,
                    "flow": flow_vars[k, u, v].varValue,
                }
            )
    mcf_result = pd.DataFrame(result)
    mcf_result.to_csv("MCF_result.csv", index=False)

    # Link Utilisation: sum flows per link
    link_util = []
    for u, v in G.edges():
        total_flow = sum([flow_vars[k, u, v].varValue for k in range(K)])
        utilisation = total_flow / G[u][v]["capacity"]
        link_util.append(
            {
                "source": u,
                "destination": v,
                "total_flow": total_flow,
                "utilisation": utilisation,
            }
        )
    link_util_df = pd.DataFrame(link_util)
    link_util_df.to_csv("LinkUtilisation.csv", index=False)
    return mcf_result, link_util_df


mcf_result_df, link_util_df = run_mcf(G, mcf)
