
"""
Traffic Generator for NSFNET/GEANT2 Assignment
Generates Single-Commodity (SCF) and Multi-Commodity (MCF) flow demands based on a student ID seed.
"""

import math
import pandas as pd
import argparse
import os

def generate_traffic(student_number="230500226", output_dir="results"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # === Derived values ===
    digits = [int(x) for x in student_number]
    S = sum(digits)
    L = digits[-1]
    SL = digits[-2] if len(digits) >= 2 else 0
    TL = digits[-3] if len(digits) >= 3 else 0

    # === Automatic topology assignment ===
    if S % 2 == 0:
        topology = "NSFNET"
        N = 14
    else:
        topology = "GEANT2"
        N = 32

    print("=====================================================")
    print("=== Assigned Topology ===")
    print("=====================================================")
    print(f"Student number : {student_number}")
    print(f"Sum of digits  : {S}")
    print(f"Assigned topology: {topology} (N = {N} nodes)")
    print("-----------------------------------------------------\n")

    # === Single-Commodity Flow (SCF) ===
    S2 = 10 * SL + L
    s = (S2 % N) + 1
    S3 = 100 * TL + 10 * SL + L
    t = (S3 % N) + 1
    if s == t:
        t = ((t) % N) + 1
    d1 = 10 * S

    print("=====================================================")
    print("=== Single-Commodity Flow (SCF) ===")
    print("=====================================================")
    print(f"Source node (s)     : {s}")
    print(f"Destination node (t): {t}")
    print(f"Demand (d1)         : {d1} Mbps")
    print("-----------------------------------------------------\n")

    scf_df = pd.DataFrame([{"source": s, "destination": t, "demand_Mbps": d1}])
    scf_filename = os.path.join(output_dir, "scf_demands.csv")
    scf_df.to_csv(scf_filename, index=False)

    # === Multi-Commodity Flow (MCF) ===
    K = 5
    p = 1 + (S % 4)
    alphas = [0.40, 0.60, 0.80, 1.00, 1.20]
    rows = []
    for k in range(1, K + 1):
        sk = ((s + (k - 1) * p - 1) % N) + 1
        tk = ((t + (k - 1) * 2 * p - 1) % N) + 1
        if sk == tk:
            tk = (tk % N) + 1
        dk = round(alphas[k - 1] * d1)
        rows.append({"commodity": k, "source": sk, "destination": tk, "demand_Mbps": dk})

    mcf_df = pd.DataFrame(rows)
    mcf_filename = os.path.join(output_dir, "mcf_demands.csv")
    mcf_df.to_csv(mcf_filename, index=False)

    print("=====================================================")
    print("=== Multi-Commodity Flow (MCF) ===")
    print("=====================================================")
    print(f"Step size (p): {p}\n")
    print(mcf_df.to_string(index=False))
    print("-----------------------------------------------------")
    print(f"Saved SCF file: {scf_filename}")
    print(f"Saved MCF file: {mcf_filename}")
    print("=====================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Network Traffic Demands")
    parser.add_argument("--student_id", type=str, default="230500226", help="Student ID for seeding")
    args = parser.parse_args()
    
    generate_traffic(args.student_id)
