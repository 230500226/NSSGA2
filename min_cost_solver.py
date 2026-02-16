
"""
Min-Cost Flow Solver for NSFNET
Solves Single-Commodity and Multi-Commodity Minimum Cost Flow problems using PuLP.
"""

import pulp
import pandas as pd
import networkx as nx
import json
import os

def solve():
    print("=" * 60)
    print("MIN-COST FLOW SOLVER STARTING")
    print("=" * 60)
    
    os.makedirs("results", exist_ok=True)

    # ============================================================
    # STEP 1: Load NSFNET Topology from CSV
    # ============================================================
    print("Loading NSFNET topology...")
    try:
        nsfnet_df = pd.read_csv("NSFNET_Links.csv")
    except FileNotFoundError:
        print("Error: NSFNET_Links.csv not found!")
        return

    G = nx.Graph()
    for _, row in nsfnet_df.iterrows():
        src = int(row["Source"])
        dst = int(row["Destination"])
        cap = int(row["Capacity_Mbps"])
        wt = int(row["Weight"])
        G.add_edge(src, dst, capacity=cap, weight=wt)

    print(f"Topology loaded: {len(G.nodes())} nodes, {len(G.edges())} edges.\n")

    # ============================================================
    # STEP 2: Solve SCF
    # ============================================================
    print("Solving Single-Commodity Flow (SCF)...")
    try:
        scf_df = pd.read_csv("results/scf_demands.csv")
    except FileNotFoundError:
        print("Error: results/scf_demands.csv not found. Run traffic_generator.py first.")
        return

    scf_source = int(scf_df.loc[0, "source"])
    scf_dest = int(scf_df.loc[0, "destination"])
    scf_demand = int(scf_df.loc[0, "demand_Mbps"])
    
    # Solve SCF logic (extracted function for reuse/clarity)
    prob_scf, flow_vars_scf, scaled_scf_demand, scf_scaling, scf_feasible = solve_scf(G, scf_source, scf_dest, scf_demand)
    
    if not scf_feasible:
        print("SCF Infeasible even after scaling.")
        return

    print(f"SCF Solved. Total Cost: {pulp.value(prob_scf.objective):.2f}")

    # Save SCF Results
    scf_link_data = []
    total_cost_scf = 0
    
    for i, j in G.edges():
        flow_ij = pulp.value(flow_vars_scf[(i, j)])
        flow_ji = pulp.value(flow_vars_scf[(j, i)])
        capacity = G[i][j]["capacity"]
        weight = G[i][j]["weight"]
        
        net_flow = 0
        direction = f"{i}<->{j}"
        
        if flow_ij > 0.01 or flow_ji > 0.01:
            if flow_ij > flow_ji:
                net_flow = flow_ij
                direction = f"{i}->{j}"
            else:
                net_flow = flow_ji
                direction = f"{j}->{i}"
        
        utilization = (net_flow / capacity) * 100
        cost = net_flow * weight
        total_cost_scf += cost
        
        scf_link_data.append({
            "Link": direction,
            "Source": i if flow_ij >= flow_ji else j,
            "Target": j if flow_ij >= flow_ji else i,
            "Flow": net_flow,
            "Capacity": capacity,
            "Utilization": utilization,
            "Cost": cost,
            "Weight": weight
        })

    pd.DataFrame(scf_link_data).to_csv("results/scf_link_utilization.csv", index=False)
    print("Saved results/scf_link_utilization.csv")

    # ============================================================
    # STEP 3: Solve MCF
    # ============================================================
    print("\nSolving Multi-Commodity Flow (MCF)...")
    try:
        mcf_df = pd.read_csv("results/mcf_demands.csv")
    except FileNotFoundError:
        print("Error: results/mcf_demands.csv not found.")
        return

    mcf_commodities = []
    for _, row in mcf_df.iterrows():
        mcf_commodities.append({
            "k": int(row["commodity"]),
            "source": int(row["source"]),
            "dest": int(row["destination"]),
            "demand": int(row["demand_Mbps"])
        })

    prob_mcf, flow_vars_mcf, scaled_comms, mcf_scaling, mcf_feasible = solve_mcf(G, mcf_commodities)
    
    if not mcf_feasible:
        print("MCF Infeasible.")
        return

    print(f"MCF Solved. Total Cost: {pulp.value(prob_mcf.objective):.2f}")

    # Save MCF Aggregate Results
    mcf_link_data = []
    mcf_detailed_data = [] # For colored plotting
    total_cost_mcf = pulp.value(prob_mcf.objective) # Using objective for total cost (accurate)

    for i, j in G.edges():
        capacity = G[i][j]["capacity"]
        weight = G[i][j]["weight"]
        
        total_flow_ij = sum(pulp.value(flow_vars_mcf[(c["k"], i, j)]) for c in scaled_comms)
        total_flow_ji = sum(pulp.value(flow_vars_mcf[(c["k"], j, i)]) for c in scaled_comms)
        
        # Aggregate
        net_flow_max = max(total_flow_ij, total_flow_ji)
        utilization = (net_flow_max / capacity) * 100
        
        mcf_link_data.append({
            "Link": f"{i}<->{j}",
            "Source": i,
            "Target": j,
            "Flow": net_flow_max,
            "Capacity": capacity,
            "Utilization": utilization,
            "Weight": weight
        })

        # Detailed per commodity
        for c in scaled_comms:
            k = c["k"]
            f_ij = pulp.value(flow_vars_mcf[(k, i, j)])
            f_ji = pulp.value(flow_vars_mcf[(k, j, i)])
            
            if f_ij > 0.01:
                mcf_detailed_data.append({"commodity": k, "source": i, "target": j, "flow": f_ij})
            if f_ji > 0.01:
                mcf_detailed_data.append({"commodity": k, "source": j, "target": i, "flow": f_ji})

    temp_mcf_df = pd.DataFrame(mcf_link_data)
    temp_mcf_df.to_csv("results/mcf_link_utilization.csv", index=False)
    pd.DataFrame(mcf_detailed_data).to_csv("results/mcf_detailed_flows.csv", index=False)
    print("Saved results/mcf_link_utilization.csv")
    print("Saved results/mcf_detailed_flows.csv")

    # ============================================================
    # STEP 4: Save Metrics
    # ============================================================
    metrics = {
        "scf": {
            "total_cost": total_cost_scf,
            "max_utilization": max([d["Utilization"] for d in scf_link_data]),
            "avg_utilization": sum([d["Utilization"] for d in scf_link_data]) / len(scf_link_data),
            "scaling_factor": scf_scaling,
            "demand": scf_demand
        },
        "mcf": {
            "total_cost": total_cost_mcf,
            "max_utilization": temp_mcf_df["Utilization"].max(),
            "avg_utilization": temp_mcf_df["Utilization"].mean(),
            "scaling_factor": mcf_scaling,
            "total_demand": sum(c["demand"] for c in scaled_comms)
        }
    }
    
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Saved results/metrics.json")
    print("=" * 60)
    print("SOLVER COMPLETE")
    print("=" * 60)


def solve_scf(G, source, dest, demand):
    """Solve SCF with scaling"""
    scaling_factor = 1.0
    for _ in range(20):
        # ... logic similar to original ...
        scaled_demand = demand * scaling_factor
        prob = pulp.LpProblem("SCF", pulp.LpMinimize)
        flow_vars = {}
        cost_expr = []
        
        for i, j in G.edges():
            flow_vars[(i, j)] = pulp.LpVariable(f"f_{i}_{j}", 0, G[i][j]["capacity"])
            flow_vars[(j, i)] = pulp.LpVariable(f"f_{j}_{i}", 0, G[i][j]["capacity"])
            cost_expr.extend([G[i][j]["weight"] * flow_vars[(i, j)], G[i][j]["weight"] * flow_vars[(j, i)]])
            
        prob += pulp.lpSum(cost_expr)
        
        for node in G.nodes():
            net_in = []
            net_out = []
            for nbr in G.neighbors(node):
                net_out.append(flow_vars[(node, nbr)])
                net_in.append(flow_vars[(nbr, node)])
            
            supply = 0
            if node == source: supply = scaled_demand
            elif node == dest: supply = -scaled_demand
            
            prob += (pulp.lpSum(net_out) - pulp.lpSum(net_in) == supply)
            
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob.status] == "Optimal":
            return prob, flow_vars, scaled_demand, scaling_factor, True
        
        scaling_factor *= 0.9
        
    return None, None, 0, 0, False

def solve_mcf(G, commodities):
    """Solve MCF with scaling"""
    scaling_factor = 1.0
    for _ in range(20):
        scaled_comms = [{**c, "demand": c["demand"] * scaling_factor} for c in commodities]
        prob = pulp.LpProblem("MCF", pulp.LpMinimize)
        flow_vars = {}
        cost_expr = []
        
        # Create vars
        for c in scaled_comms:
            k = c["k"]
            for i, j in G.edges():
                flow_vars[(k, i, j)] = pulp.LpVariable(f"f_{k}_{i}_{j}", 0)
                flow_vars[(k, j, i)] = pulp.LpVariable(f"f_{k}_{j}_{i}", 0)
                cost_expr.extend([G[i][j]["weight"] * flow_vars[(k, i, j)], G[i][j]["weight"] * flow_vars[(k, j, i)]])
        
        prob += pulp.lpSum(cost_expr)
        
        # Flow conservation
        for c in scaled_comms:
            k = c["k"]
            for node in G.nodes():
                supply = 0
                if node == c["source"]: supply = c["demand"]
                elif node == c["dest"]: supply = -c["demand"]
                
                net_out = [flow_vars[(k, node, nbr)] for nbr in G.neighbors(node)]
                net_in = [flow_vars[(k, nbr, node)] for nbr in G.neighbors(node)]
                prob += (pulp.lpSum(net_out) - pulp.lpSum(net_in) == supply)
                
        # Capacity constraints
        for i, j in G.edges():
            total_ij = [flow_vars[(c["k"], i, j)] for c in scaled_comms]
            total_ji = [flow_vars[(c["k"], j, i)] for c in scaled_comms]
            cap = G[i][j]["capacity"]
            prob += pulp.lpSum(total_ij) <= cap
            prob += pulp.lpSum(total_ji) <= cap
            
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob.status] == "Optimal":
            return prob, flow_vars, scaled_comms, scaling_factor, True
            
        scaling_factor *= 0.9
        
    return None, None, [], 0, False

if __name__ == "__main__":
    solve()
