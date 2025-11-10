import matplotlib.pyplot as plt
import numpy as np

# Data from your results
scf_utilisation = [91.3, 100.0, 91.3, 100.0, 91.3, 100.0]
mcf_utilisation = [
    44.0,
    17.0,
    100.0,
    100.0,
    100.0,
    57.7,
    100.0,
    100.0,
    81.3,
    1.7,
    100.0,
    66.0,
    100.0,
    1.7,
    1.7,
    66.0,
]

# Create the plot
plt.style.use("seaborn-v0_8-deep")
fig, ax = plt.subplots(figsize=(10, 6))

# Define bins for the histogram
bins = np.arange(0, 101, 10)

# Plot histograms
ax.hist(
    mcf_utilisation,
    bins=bins,
    alpha=0.7,
    label="MCF Utilisation",
    color="skyblue",
    edgecolor="black",
)
ax.hist(
    scf_utilisation,
    bins=bins,
    alpha=0.9,
    label="SCF Utilisation",
    color="salmon",
    edgecolor="black",
)

# Add titles and labels
ax.set_title("Figure 7: Histogram of Link Utilisation Values (SCF vs MCF)", fontsize=16)
ax.set_xlabel("Link Utilisation (%)", fontsize=12)
ax.set_ylabel("Number of Links (Frequency)", fontsize=12)
ax.set_xticks(bins)
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.legend()

# Save the figure
plt.tight_layout()
plt.savefig("figure7_utilisation_histogram.png")

print("Generated figure7_utilisation_histogram.png")
