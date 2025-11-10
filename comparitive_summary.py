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

# --- Aggregating Data for Figure 9 ---
print("\nAggregating data for Figure 9...")

# 1. Get Utilisation Metrics
max_util_mcf = mcf_df['Utilization (%)'].max()
avg_util_mcf = mcf_df['Utilization (%)'].mean()

max_util_scf = scf_df['Utilization (%)'].max()
avg_util_scf = scf_df['Utilization (%)'].mean()

# 2. Get Cost Metrics (requires running the code from Figure 8 first)
# For simplicity, we'll re-run the cost calculation here.
total_cost_scf = scf_df['Cost'].sum()
scf_df['Weight'] = scf_df['Cost'] / scf_df['Flow (Mbps)']
scf_df.replace([np.inf, -np.inf], 0, inplace=True)
scf_df['Weight'] = scf_df['Weight'].fillna(0)
weight_dict = {}
for _, row in scf_df.iterrows():
    weight_dict[row['Link']] = row['Weight']
mcf_cost_list = []
for _, row in mcf_df.iterrows():
    nodes = row['Link'].split('↔')
    if len(nodes) == 2:
        n1, n2 = nodes[0], nodes[1]
        link_fwd = f"{n1}→{n2}"
        link_bwd = f"{n2}→{n1}"
        weight = weight_dict.get(link_fwd, weight_dict.get(link_bwd, 0))
        link_cost = row['Flow (Mbps)'] * weight
        mcf_cost_list.append(link_cost)
total_cost_mcf = sum(mcf_cost_list)

# --- Plotting Figure 9 ---
print("Generating Figure 9: Comparative Summary Plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Figure 9: Comparative Summary of SCF and MCF Performance', fontsize=16)

# --- Plot 1: Utilization Metrics ---
metrics_util = ['Max Utilization', 'Average Utilization']
mcf_util_values = [max_util_mcf, avg_util_mcf]
scf_util_values = [max_util_scf, avg_util_scf]

x_util = np.arange(len(metrics_util))
width = 0.35

rects1 = ax1.bar(x_util - width/2, mcf_util_values, width, label='MCF', color='blue')
rects2 = ax1.bar(x_util + width/2, scf_util_values, width, label='SCF (Min-Max)', color='orange')

ax1.set_ylabel('Utilization (%)')
ax1.set_title('Utilisation Performance')
ax1.set_xticks(x_util)
ax1.set_xticklabels(metrics_util)
ax1.set_ylim(0, 110)
ax1.legend()
ax1.grid(axis='y', linestyle=':', alpha=0.7)

# Add labels to bars
ax1.bar_label(rects1, padding=3, fmt='%.1f%%')
ax1.bar_label(rects2, padding=3, fmt='%.1f%%')

# --- Plot 2: Cost Metric ---
metrics_cost = ['Total Cost']
mcf_cost_values = [total_cost_mcf]
scf_cost_values = [total_cost_scf]

x_cost = np.arange(len(metrics_cost))

rects3 = ax2.bar(x_cost - width/2, mcf_cost_values, width, label='MCF', color='blue')
rects4 = ax2.bar(x_cost + width/2, scf_cost_values, width, label='SCF (Min-Max)', color='orange')

ax2.set_ylabel('Total Cost (Cost Units)')
ax2.set_title('Cost Performance')
ax2.set_xticks(x_cost)
ax2.set_xticklabels(metrics_cost)
ax2.legend()
ax2.grid(axis='y', linestyle=':', alpha=0.7)

# Add labels to bars
ax2.bar_label(rects3, padding=3, fmt='{:,.0f}')
ax2.bar_label(rects4, padding=3, fmt='{:,.0f}')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('figure_9_comparative_summary.png')
print("Saved 'figure_9_comparative_summary.png'")
plt.show()
