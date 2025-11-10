import matplotlib.pyplot as plt
import numpy as np

# Data from your results
labels = ["Total Cost", "Avg. Utilisation (%)", "Bottleneck Links"]
scf_metrics = [767.03, 94.21, 9]
mcf_metrics = [1814.86, 64.82, 7]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

plt.style.use("seaborn-v0_8-deep")
fig, ax = plt.subplots(figsize=(12, 7))

# Create bars for SCF and MCF
rects1 = ax.bar(x - width / 2, scf_metrics, width, label="SCF", color="salmon")
rects2 = ax.bar(x + width / 2, mcf_metrics, width, label="MCF", color="skyblue")

# Add some text for labels, title and axes ticks
ax.set_ylabel("Value")
ax.set_title(
    "Figure 9: Comparative Summary of SCF and MCF Performance Metrics", fontsize=16
)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend()


# Function to attach a text label above each bar
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("figure9_comparative_summary.png")

print("Generated figure9_comparative_summary.png")
