import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
try:
    mcf_df = pd.read_csv('MCF_result.csv')
    scf_df = pd.read_csv('SCF_result.csv')
except FileNotFoundError:
    print("Ensure 'MCF_result.csv' and 'SCF_result.csv' are in the same directory.")
    exit()

# --- Plotting Figure 7 ---
print("\nGenerating Figure 7: Histogram of Link Utilisation...")

plt.figure(figsize=(12, 7))

# Plot histograms
# We use the raw flow data, as the SCF file has 42 links and MCF has 21.
# This shows the distribution of all link *directions* in SCF.
plt.hist(mcf_df['Utilization (%)'], bins=20, alpha=0.7, label='MCF', color='blue', edgecolor='black')
plt.hist(scf_df['Utilization (%)'], bins=20, alpha=0.7, label='SCF (Min-Max)', color='orange', edgecolor='black')

plt.title('Figure 7: Histogram of Link Utilisation Values')
plt.xlabel('Link Utilisation (%)')
plt.ylabel('Number of Links')
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.7)

plt.savefig('figure_7_utilisation_histogram.png')
print("Saved 'figure_7_utilisation_histogram.png'")
plt.show()
