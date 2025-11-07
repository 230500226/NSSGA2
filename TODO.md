# Traffic Engineering Project To-Do List

## Preliminaries
- [ ] Read the entire project description and requirements (see traffic_eng_project.pdf).
- [ ] Note the examiner, moderator names, and assessment criteria.

## Step 1: Topology Assignment
- [ ] Run the provided Python Flow Generator (Appendix A) with your student number.
- [ ] Record your assigned topology (NSFNET or GEANT2) and its scale (number of nodes/links).
- [ ] Write a short paragraph about your assigned topology:
    - [ ] Describe its origin, scale, and research relevance.
    - [ ] Include one reference figure from Appendix C.

## Step 2: Traffic Flow Generation
- [ ] Capture the SCF (Single-Commodity Flow) and MCF (Multi-Commodity Flow) results displayed by the script.
- [ ] Insert a screenshot or recreate a table of these results for your report.
- [ ] Ensure these files are generated:
    - [ ] `scf_<student_number>.csv`
    - [ ] `demands_<student_number>.csv`
- [ ] Check all feasibility criteria:
    - [ ] Both SCF and MCF printed to console
    - [ ] Two CSV files produced
    - [ ] Correct assigned topology displayed
    - [ ] Demands in hundreds of Mbps, not zero

### 2.3 If Model is Infeasible
- [ ] If optimisation model is infeasible, apply demand scaling (reduce all demands by multiplying by 0.9 until feasible).
- [ ] Report the final scaling factor and scaled demand values.

## Step 3: Shortest Path Routing
- [ ] Use NetworkX to compute Dijkstra’s shortest path for your assigned topology.
    - [ ] Use source and destination nodes from your SCF data.
    - [ ] Report path cost/latency (sum of weights).
    - [ ] Auto-highlight the shortest path in your topology figure (edges in red).
    - [ ] Include the labelled figure in your report.
    - [ ] Discuss whether this path aligns with network expectations.

## Step 4: Linear Programming Models (Flows)
- [ ] Formulate and solve these models using Python & libraries (NetworkX, PuLP, SciPy):
    - [ ] SCF model for (s, t, d₁)
    - [ ] MCF model for five commodities
    - [ ] Two objectives: Minimise total cost (Min-Cost Flow) and Minimise max link utilisation (Min-Max-U)
- [ ] Re-apply scaling if infeasible.
- [ ] Save & include:
    - [ ] SCF_result.csv
    - [ ] MCF_result.csv
    - [ ] LinkUtilisation.csv
    - [ ] All relevant result figures

## Step 5: Statistical Analysis
- [ ] Analyse and present:
    - [ ] Average path length
    - [ ] Utilisation variance
    - [ ] Bottleneck links
- [ ] Create tables and histograms for your analysis.

## Step 6: Interpretation
- [ ] Discuss all figures and tables.
- [ ] Relate your results to Traffic Engineering principles.

## Step 7: Reflection
- [ ] Discuss any limitations, difficulties encountered, and possible improvements.

## Step 8: Optional (+10%)
- [ ] If desired, implement a simple ML method (e.g., regression for traffic prediction).

## Step 9: Formatting and Submission
- [ ] Write the report with proper structure:
    1. Introduction & Mathematical Terms (10%)
    2. Theory (10%)
    3. Mathematical Methods (10%)
    4. Calculus / Linear Algebra (10%)
    5. Calculations & Solver Results (20%)
    6. Statistical Analysis (20%)
    7. Interpretation (20%)
    8. Conclusion + Reflection
    9. References (IEEE ≥ 5)
    10. Appendices (code, outputs, screenshots)
- [ ] Max length: 12 pages (excluding appendices).
- [ ] Ensure IEEE-style references (at least 5).
- [ ] Submit as PDF via Blackboard for full marks (penalty for other formats).
- [ ] Attach Python source code.
- [ ] Double-check everything before final submission: outputs, results, file formats, and completeness.

## Appendices
- [ ] Include:
    - [ ] Python code (with clear documentation/comments)
    - [ ] All result figures and screenshots

---
**Tip:** Consult the troubleshooting guide (Appendix D) if you run into problems like missing packages or infeasible models.
