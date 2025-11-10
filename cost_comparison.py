import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
try:
    mcf_df = pd.read_csv('MCF_result.csv')
    scf_df = pd.read_csv('SCF_result.csv')
except FileNotFoundError:
    print("Ensure 'MCF_result.csv' and 'SCF_result.csv' are in the same directory.")
    exit()

# --- Calculating Costs for Figure 8 ---
print("\nCalculating Total Costs for Figure 8...")

# 1. Calculate Total SCF Cost (Easy)
total_cost_scf = scf_df['Cost'].sum()
print(f"Total SCF Cost: {total_cost_scf:.2f}")

# 2. Derive weights from SCF data to apply to MCF
# Weight = Cost / Flow
scf_df['Weight'] = scf_df['Cost'] / scf_df['Flow (Mbps)']
# Handle potential division by zero (if flow is 0, cost is 0, weight is irrelevant)
scf_df.replace([np.inf, -np.inf], 0, inplace=True)
scf_df['Weight'] = scf_df['Weight'].fillna(0)

# Create a dictionary of weights for easy lookup
# e.g., weight_dict['1→2'] = 3
weight_dict = {}
for _, row in scf_df.iterrows():
    # Store weight for directed link, e.g., '1→2'
    weight_dict[row['Link']] = row['Weight']

# 3. Calculate Total MCF Cost (Harder)
# We need to match undirected links (e.g., '1↔2') with directed weights
mcf_cost_list = []
for _, row in mcf_df.iterrows():
    # Split undirected link '1↔2' into ['1', '2']
    nodes = row['Link'].split('↔')
    if len(nodes) == 2:
        n1, n2 = nodes[0], nodes[1]
        
        # Form directed links, e.g., '1→2' and '2→1'
        link_fwd = f"{n1}→{n2}"
        link_bwd = f"{n2}→{n1}"
        
        # Find the weight. Assume it's symmetric and stored in one of
        # the directed links. Use 0 as a default if not found.
        weight = weight_dict.get(link_fwd, weight_dict.get(link_bwd, 0))
        
        # MCF Cost = Total flow on undirected link * symmetric weight
        link_cost = row['Flow (Mbps)'] * weight
        mcf_cost_list.append(link_cost)

total_cost_mcf = sum(mcf_cost_list)
print(f"Total MCF Cost: {total_cost_mcf:.2f}")

# --- Plotting Figure 8 ---
print("Generating Figure 8: Total Cost Comparison...")

plt.figure(figsize=(8, 7))
models = ['MCF (Min-Cost)', 'SCF (Min-Max Util)']
costs = [total_cost_mcf, total_cost_scf]

bars = plt.bar(models, costs, color=['blue', 'orange'])
plt.title('Figure 8: Bar Chart Comparing SCF and MCF Total Costs')
plt.ylabel('Total Cost (Cost Units)')
plt.grid(axis='y', linestyle=':', alpha=0.7)

# Add text labels on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:,.2f}', 
             va='bottom', ha='center')

plt.savefig('figure_8_total_cost_comparison.png')
print("Saved 'figure_8_total_cost_comparison.png'")
plt.show()
