import networkx as nx
import matplotlib.pyplot as plt
import pulp

# (Replace these stubs with actual file loading)
# Example network data
edges = [
    ("A", "B", 10),
    ("B", "C", 20),
    ("A", "C", 15),
    ("C", "D", 45),
    ("B", "D", 45),
    ("A", "D", 5),
]
capacities = {(u, v): 45 for u, v, _ in edges}
costs = {(u, v): w for u, v, w in edges}
demands = {("A", "D"): 60, ("B", "D"): 30}  # Example; update with your true MCF demands
MAX_CAPACITY = 45


def solve_mcf_and_plot(edges, capacities, costs, demands, max_capacity, plot_path):
    scale_factor = 1.0
    while True:
        prob = pulp.LpProblem("MinCostFlow", pulp.LpMinimize)
        flow_vars = {}
        for u, v, _ in edges:
            flow_vars[(u, v)] = pulp.LpVariable(f"f_{u}_{v}", 0, capacities[(u, v)])

        # Objective: minimize sum of cost * flow
        prob += pulp.lpSum([costs[(u, v)] * flow_vars[(u, v)] for u, v, _ in edges])

        # Flow conservation for each node
        nodes = set([u for u, v, _ in edges] + [v for u, v, _ in edges])
        for n in nodes:
            net_demand = sum(
                [scale_factor * d for (src, dst), d in demands.items() if n == dst]
            ) - sum([scale_factor * d for (src, dst), d in demands.items() if n == src])
            in_flow = pulp.lpSum([flow_vars[(u, v)] for u, v, _ in edges if v == n])
            out_flow = pulp.lpSum([flow_vars[(u, v)] for u, v, _ in edges if u == n])
            prob += (in_flow - out_flow == net_demand), f"node_{n}_balance"

        # Capacity constraints are in var definitions

        prob.solve()

        # Check feasibility: flows <= max_capacity on any link
        over_cap = False
        actual_flows = {}
        for u, v, _ in edges:
            flow_val = (
                flow_vars[(u, v)].varValue
                if flow_vars[(u, v)].varValue is not None
                else 0
            )
            actual_flows[(u, v)] = flow_val
            if flow_val > max_capacity + 1e-3:
                over_cap = True

        if not over_cap:
            break
        scale_factor *= 0.9

    # Now plot the network with flows
    G = nx.DiGraph()
    for u, v, w in edges:
        G.add_edge(
            u, v, weight=w, capacity=capacities[(u, v)], flow=actual_flows[(u, v)]
        )

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue")
    nx.draw_networkx_labels(G, pos)
    # Draw all edges with edge labels (flow/capacity)
    edge_colors = ["red" if G[u][v]["flow"] > 0 else "grey" for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, pos, arrowstyle="->", arrowsize=20, edge_color=edge_colors
    )
    edge_labels = {
        (u, v): f"{G[u][v]['flow']:.1f}/{G[u][v]['capacity']} (w={G[u][v]['weight']})"
        for u, v in G.edges()
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    plt.title(f"MCF Solution (scaled demands x {scale_factor:.3f})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"MCF diagram saved as: {plot_path}")


if __name__ == "__main__":
    solve_mcf_and_plot(edges, capacities, costs, demands, MAX_CAPACITY, "mcf_path.png")
