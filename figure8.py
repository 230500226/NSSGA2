import matplotlib.pyplot as plt

# Data from your results
models = ["SCF", "MCF"]
costs = [767.03, 1814.86]

# Create the plot
plt.style.use("seaborn-v0_8-deep")
fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(models, costs, color=["salmon", "skyblue"])

# Add titles and labels
ax.set_title("Figure 8: Bar Chart Comparing SCF and MCF Total Costs", fontsize=16)
ax.set_ylabel("Total Routing Cost", fontsize=12)
ax.set_xlabel("Optimisation Model", fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Add cost labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval,
        f"{yval:,.2f}",
        va="bottom",
        ha="center",
    )

# Save the figure
plt.tight_layout()
plt.savefig("figure8_cost_comparison.png")

print("Generated figure8_cost_comparison.png")
