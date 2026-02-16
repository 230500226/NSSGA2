
# 🌐 NSFNET Flow Optimizer

> **High-performance network traffic optimization using Linear Programming.**

This repository hosts a Python-based solver for Minimum Cost Flow (MCF) problems on large-scale network topologies like NSFNET. It leverages PuLP for linear optimization and NetworkX for graph visualization to model, solve, and analyze network traffic demands.

![Language: Python](https://img.shields.io/badge/Language-Python-blue.svg?logo=python&logoColor=white)
![Library: PuLP](https://img.shields.io/badge/Optimization-PuLP-orange)
![Library: NetworkX](https://img.shields.io/badge/Graph_Theory-NetworkX-green)
![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen)

## 📂 Repository Structure

The project is organized into modular scripts for generation, solving, and visualization:

| Component | Description |
| :--- | :--- |
| **traffic_generator.py** | Generates synthetic Single and Multi-Commodity flow demands based on seed parameters. |
| **min_cost_solver.py** | Core linear programming solver using PuLP to optimize routing for minimal cost. |
| **visualization.py** | Generates comprehensive plots (Link Utilization, Cost Comparison, Flow Diagrams). |
| **shortest_path.py** | Dijkstra's algorithm implementation for quick shortest-path verification. |
| **NSFNET_Links.csv** | Topology definition file containing nodes, links, capacities, and weights. |
| **results/** | Directory for all generated CSV data, JSON metrics, and visualization images. |

---

## 🚀 Installation & Usage

### 1. Prerequisites
*Requires Python 3.8+*

<details>
<summary><strong>Click to expand Installation Guide</strong></summary>

#### Phase 1: Environment Setup
1.  **Clone Repository:**
    ```bash
    git clone https://github.com/rustinsystems/nsfnet-flow-optimizer.git
    cd nsfnet-flow-optimizer
    ```
2.  **Install Dependencies:**
    ```bash
    pip install pandas networkx pulp matplotlib
    ```

</details>

### 2. Running the Optimization Pipeline
*Follow these steps to generate data, solve flow problems, and visualize results.*

<details>
<summary><strong>Click to expand Usage Guide</strong></summary>

#### Step 1: Generate Traffic Demands
Run the generator to create synthetic demand data. You can optionally specify a student ID seed.
```bash
python traffic_generator.py --student_id 230500226
```
*Outputs: `results/scf_demands.csv`, `results/mcf_demands.csv`*

#### Step 2: Solve Optimization Problems
Execute the solver to compute optimal routing strategies for Single-Commodity (SCF) and Multi-Commodity (MCF) flows.
```bash
python min_cost_solver.py
```
*Outputs: `results/scf_link_utilization.csv`, `results/mcf_detailed_flows.csv`, `results/metrics.json`*

#### Step 3: Visualize Results
Generate insightful charts and network diagrams to analyze the solver's performance.
```bash
python visualization.py
```
*Outputs: `results/figure7_utilisation_histogram.png`, `results/mcf_flow_diagram_colored.png`, etc.*

#### Step 4: Verify Shortest Path (Optional)
Run Dijkstra's algorithm to compare theoretical shortest paths against the optimized flow.
```bash
python shortest_path.py
```

</details>

## 📊 Visualization Gallery

The `visualization.py` script automatically generates detailed analytics:

*   **Link Utilization Histogram**: Compares congestion levels between SCF and MCF.
*   **Cost Comparison**: Bar chart showing total routing costs for different strategies.
*   **Network Flow Diagrams**: Visual representation of traffic flow, with multi-commodity flows color-coded for clarity.

<div align="center">
<p><i>Managed by <a href="https://rustinsystems.com">Rustin Systems</a></i></p>
<p>Bridging Theoretical Algorithms & Practical Implementation</p>
</div>