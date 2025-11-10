import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
try:
    mcf_df = pd.read_csv('MCF_result.csv')
    scf_df = pd.read_csv('SCF_result.csv')
except FileNotFoundError:
    print("Ensure 'MCF_result.csv' and 'SCF_result.csv' are in the same directory.")
    exit()

# --- Plotting Figure 6 ---
print("Generating Figure 6: Link Utilisation Bar Charts...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14))
fig.suptitle('Figure 6: Solver Output Showing Link Utilisation', fontsize=16, y=1.02)

# Subplot 1: MCF Utilization
ax1.bar(mcf_df['Link'], mcf_df['Utilization (%)'], color='blue')
ax1.set_title('MCF Link Utilisation')
ax1.set_ylabel('Utilization (%)')
ax1.set_ylim(0, 110) # Set Y-axis to 110% to see the 100% bars clearly
ax1.axhline(y=100, color='red', linestyle='--', label='100% Capacity')
ax1.tick_params(axis='x', rotation=90)
ax1.legend()
ax1.grid(axis='y', linestyle=':', alpha=0.7)

# Subplot 2: SCF Utilization
# Note: SCF_result.csv has 42 links (directed) vs MCF's 21 (undirected)
ax2.bar(scf_df['Link'], scf_df['Utilization (%)'], color='orange')
ax2.set_title('SCF (Min-Max) Link Utilisation')
ax2.set_ylabel('Utilization (%)')
ax2.set_ylim(0, 110)
# Find max SCF util to draw a line
max_scf_util = scf_df['Utilization (%)'].max()
ax2.axhline(y=max_scf_util, color='darkgreen', linestyle='--', 
            label=f'Max Utilisation ({max_scf_util:.1f}%)')
ax2.tick_params(axis='x', rotation=90)
ax2.legend()
ax2.grid(axis='y', linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig('figure_6_link_utilisation.png')
print("Saved 'figure_6_link_utilisation.png'")
plt.show()
