"""
PersistRank — Phase 8: Full Evaluation
Computes P@1, P@3, P@5 for all 5 methods across ALL anomaly rows.
Also runs ablation study (3 component combinations).
Outputs: results tables for the paper.
"""
import pandas as pd
import numpy as np
import time
from criticality import CRITICALITY, MAX_CRIT

# =====================================================================
# Setup
# =====================================================================
print("=" * 60)
print("Phase 8: Full Evaluation (All Methods, All Anomaly Rows)")
print("=" * 60)

print("\n[8.0] Loading risk data...")
t0 = time.time()
df = pd.read_csv("persistence_data.csv")

SERVICES = list(CRITICALITY.keys())

s_min = df["anomaly_score"].min()
s_max = df["anomaly_score"].max()

print(f"  Loaded {len(df)} rows in {time.time()-t0:.1f}s")

# =====================================================================
# Step 8.1 — Determine ground truth service per anomaly row
# =====================================================================
print("\n[8.1] Determining ground truth services...")

# target_service is already in the data from load_data.py
# For rows where target is known, use it directly
# For ambiguous cases, use service with highest raw anomaly score as fallback
anomaly_df = df[df["label"] != "normal"].copy()
print(f"  Anomaly rows: {len(anomaly_df)}")
print(f"  Target services: {anomaly_df['target_service'].value_counts().to_dict()}")

# Filter to rows with known target services
anomaly_df = anomaly_df[anomaly_df["target_service"] != "none"].copy()
print(f"  Rows with known target: {len(anomaly_df)}")

# =====================================================================
# Step 8.2 — Compute Precision@K for all methods
# =====================================================================
print("\n[8.2] Evaluating all methods (this may take a minute)...")

np.random.seed(42)  # reproducible random baseline

def rank_services(row, mode):
    """Rank services by score for a given mode."""
    scores = {}
    for svc in SERVICES:
        a = (row[f"score_{svc}"] - s_min) / (s_max - s_min + 1e-8)
        c = CRITICALITY[svc] / MAX_CRIT
        p = row[f"persist_{svc}"]

        if mode == "persistrank":
            scores[svc] = a * c * p
        elif mode == "no_persist":      # AlertRank-style
            scores[svc] = a * c
        elif mode == "no_criticality":  # Ablation
            scores[svc] = a * p
        elif mode == "anom_only":
            scores[svc] = a
        elif mode == "crit_only":
            scores[svc] = c
        elif mode == "random":
            scores[svc] = np.random.rand()
    return sorted(scores, key=scores.get, reverse=True)

modes = ["persistrank", "no_persist", "no_criticality", "anom_only", "crit_only", "random"]
k_values = [1, 3, 5]

# Pre-compute all rankings (vectorized where possible)
results = {mode: {f"P@{k}": [] for k in k_values} for mode in modes}

t1 = time.time()
total_rows = len(anomaly_df)
checkpoint = max(1, total_rows // 10)

for i, (idx, row) in enumerate(anomaly_df.iterrows()):
    true_svc = row["target_service"]

    for mode in modes:
        ranked = rank_services(row, mode)
        for k in k_values:
            hit = 1 if true_svc in ranked[:k] else 0
            results[mode][f"P@{k}"].append(hit)

    if (i + 1) % checkpoint == 0:
        print(f"  Progress: {i+1}/{total_rows} rows ({(i+1)/total_rows*100:.0f}%)")

elapsed = time.time() - t1
print(f"  Evaluation completed in {elapsed:.1f}s ({total_rows/elapsed:.0f} rows/sec)")

# =====================================================================
# Step 8.3 — Main Results Table (Table 3 in paper)
# =====================================================================
print("\n" + "=" * 60)
print("TABLE 3: Main Results — Precision@K Across All Methods")
print("=" * 60)

print(f"\n  {'Method':<25s} {'P@1':>8s} {'P@3':>8s} {'P@5':>8s}")
print(f"  {'-'*50}")

method_labels = {
    "persistrank":    "PersistRank (ours)",
    "no_persist":     "No Persistence (AlertRank)",
    "no_criticality": "No Criticality",
    "anom_only":      "Anomaly Only",
    "crit_only":      "Criticality Only",
    "random":         "Random",
}

main_results = {}
for mode in modes:
    p1 = np.mean(results[mode]["P@1"])
    p3 = np.mean(results[mode]["P@3"])
    p5 = np.mean(results[mode]["P@5"])
    main_results[mode] = {"P@1": p1, "P@3": p3, "P@5": p5}
    marker = " ***" if mode == "persistrank" else ""
    print(f"  {method_labels[mode]:<25s} {p1:>8.3f} {p3:>8.3f} {p5:>8.3f}{marker}")

# Improvement over baselines
pr_p3 = main_results["persistrank"]["P@3"]
np_p3 = main_results["no_persist"]["P@3"]
ao_p3 = main_results["anom_only"]["P@3"]
print(f"\n  PersistRank P@3 improvement over AlertRank-style: {(pr_p3 - np_p3)*100:+.1f} points")
print(f"  PersistRank P@3 improvement over Anomaly-only:    {(pr_p3 - ao_p3)*100:+.1f} points")

# =====================================================================
# Step 8.4 — Ablation Study (Table 4 in paper)
# =====================================================================
print("\n" + "=" * 60)
print("TABLE 4: Ablation Study")
print("=" * 60)

ablation_modes = ["persistrank", "no_persist", "no_criticality", "anom_only"]
ablation_labels = {
    "persistrank":    "A x C x P (Full PersistRank)",
    "no_persist":     "A x C     (No Persistence)",
    "no_criticality": "A x P     (No Criticality)",
    "anom_only":      "A         (Anomaly Only)",
}

print(f"\n  {'Components':<35s} {'P@1':>8s} {'P@3':>8s} {'P@5':>8s}")
print(f"  {'-'*60}")
for mode in ablation_modes:
    r = main_results[mode]
    marker = " ***" if mode == "persistrank" else ""
    print(f"  {ablation_labels[mode]:<35s} {r['P@1']:>8.3f} {r['P@3']:>8.3f} {r['P@5']:>8.3f}{marker}")

# =====================================================================
# Per-anomaly-type breakdown
# =====================================================================
print("\n" + "=" * 60)
print("TABLE 5: Per-Anomaly-Type P@3 (PersistRank vs AlertRank-style)")
print("=" * 60)

print(f"\n  {'Anomaly Type':<30s} {'Rows':>6s} {'PersistRank':>12s} {'AlertRank':>10s} {'Delta':>8s} {'Winner'}")
print(f"  {'-'*80}")

anomaly_types = sorted(anomaly_df["anomaly_type"].unique())
for atype in anomaly_types:
    mask = anomaly_df["anomaly_type"] == atype
    type_indices = anomaly_df[mask].index

    # Get P@3 for each method on this anomaly type
    pr_hits = []
    np_hits = []
    for idx in type_indices:
        row = df.loc[idx]
        true_svc = row["target_service"]

        pr_ranked = rank_services(row, "persistrank")
        np_ranked = rank_services(row, "no_persist")

        pr_hits.append(1 if true_svc in pr_ranked[:3] else 0)
        np_hits.append(1 if true_svc in np_ranked[:3] else 0)

    pr_p3 = np.mean(pr_hits)
    np_p3 = np.mean(np_hits)
    delta = pr_p3 - np_p3
    winner = "PR" if delta > 0.01 else ("AR" if delta < -0.01 else "TIE")
    print(f"  {atype:<30s} {mask.sum():>6d} {pr_p3:>12.3f} {np_p3:>10.3f} {delta:>+8.3f}   {winner}")

# =====================================================================
# Save results to CSV
# =====================================================================
print(f"\n{'=' * 60}")
print("Saving results...")

# Main results
results_df = pd.DataFrame(main_results).T
results_df.index.name = "method"
results_df.to_csv("results_main.csv")
print(f"  Saved results_main.csv")

# Per-type results
type_results = []
for atype in anomaly_types:
    mask = anomaly_df["anomaly_type"] == atype
    row_data = {"anomaly_type": atype, "rows": mask.sum()}
    for mode in modes:
        type_indices = list(anomaly_df[mask].index)
        hits = []
        for idx in type_indices:
            row = df.loc[idx]
            true_svc = row["target_service"]
            ranked = rank_services(row, mode)
            hits.append(1 if true_svc in ranked[:3] else 0)
        row_data[f"P@3_{mode}"] = np.mean(hits)
    type_results.append(row_data)

type_df = pd.DataFrame(type_results)
type_df.to_csv("results_per_type.csv", index=False)
print(f"  Saved results_per_type.csv")

total_time = time.time() - t0
print(f"\n  Total runtime: {total_time:.1f}s")
print(f"{'=' * 60}")
print("Phase 8 complete!")
