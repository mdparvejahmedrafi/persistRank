"""
PersistRank — Phase 7: Risk Score + Ranking
Computes: Risk(v,t) = anomaly_norm × criticality_norm × persistence_norm
Ranks services per row and tests on sample anomaly rows.
Outputs: df with risk_{service} columns, saves to risk_data.csv
"""
import pandas as pd
import numpy as np
import time
from criticality import CRITICALITY, MAX_CRIT

# =====================================================================
# Step 7.1 — Load persistence data and compute risk scores
# =====================================================================
print("=" * 60)
print("Phase 7: Risk Score + Ranking (PersistRank Formula)")
print("=" * 60)

print("\n[7.1] Loading persistence data...")
t0 = time.time()
df = pd.read_csv("persistence_data.csv")

SERVICES = list(CRITICALITY.keys())

# Global anomaly score min/max for normalization
s_min = df["anomaly_score"].min()
s_max = df["anomaly_score"].max()

print(f"  Loaded {len(df)} rows in {time.time()-t0:.1f}s")
print(f"  Anomaly score range: [{s_min:.4f}, {s_max:.4f}]")

# Compute risk per service per row
print("\n  Computing risk scores: Risk = anomaly_norm x criticality_norm x persistence_norm")
for svc in SERVICES:
    a = (df[f"score_{svc}"] - s_min) / (s_max - s_min + 1e-8)   # anomaly_norm
    c = CRITICALITY[svc] / MAX_CRIT                               # criticality_norm
    p = df[f"persist_{svc}"]                                       # persistence_norm
    df[f"risk_{svc}"] = a * c * p

print(f"  Done. Added {len(SERVICES)} risk columns.")

# =====================================================================
# Step 7.2 — Ranking function for all 5 modes
# =====================================================================
print("\n[7.2] Defining ranking modes...")

def get_ranked_services(row, mode="persistrank"):
    """Rank services by score for a given row. Returns sorted list."""
    scores = {}
    for svc in SERVICES:
        a = (row[f"score_{svc}"] - s_min) / (s_max - s_min + 1e-8)
        c = CRITICALITY[svc] / MAX_CRIT
        p = row[f"persist_{svc}"]

        if mode == "persistrank":
            scores[svc] = a * c * p
        elif mode == "no_persist":
            scores[svc] = a * c          # AlertRank-style baseline
        elif mode == "no_criticality":
            scores[svc] = a * p           # Ablation: no criticality
        elif mode == "anom_only":
            scores[svc] = a               # Anomaly score only
        elif mode == "crit_only":
            scores[svc] = c               # Criticality only (static)
        elif mode == "random":
            scores[svc] = np.random.rand()
    return sorted(scores, key=scores.get, reverse=True)

modes = ["persistrank", "no_persist", "no_criticality", "anom_only", "crit_only", "random"]
print(f"  Modes defined: {modes}")

# =====================================================================
# Step 7.3 — Test on sample anomaly rows
# =====================================================================
print("\n[7.3] Testing on sample anomaly rows...")

# Pick one row from each anomaly label
anomaly_labels = [l for l in df["label"].unique() if l != "normal"]

print(f"\n  {'Anomaly Label':<35s} {'Target':<10s} | {'PersistRank Top-3':<40s} | {'No-Persist Top-3':<40s} | {'PR Hit':>6s} {'NP Hit':>6s}")
print(f"  {'-'*150}")

pr_hits = 0
np_hits = 0
total = 0

for label in sorted(anomaly_labels):
    subset = df[df["label"] == label]
    target = subset["target_service"].iloc[0]
    if target == "none":
        continue

    # Use median row (middle of the anomaly window)
    mid_idx = subset.index[len(subset) // 2]
    row = df.loc[mid_idx]

    pr_ranked = get_ranked_services(row, "persistrank")
    np_ranked = get_ranked_services(row, "no_persist")

    pr_hit = target in pr_ranked[:3]
    np_hit = target in np_ranked[:3]
    if pr_hit: pr_hits += 1
    if np_hit: np_hits += 1
    total += 1

    pr_mark = "HIT" if pr_hit else "MISS"
    np_mark = "HIT" if np_hit else "MISS"

    print(f"  {label:<35s} {target:<10s} | {str(pr_ranked[:3]):<40s} | {str(np_ranked[:3]):<40s} | {pr_mark:>6s} {np_mark:>6s}")

print(f"\n  Sample hit rate: PersistRank={pr_hits}/{total} ({pr_hits/total*100:.0f}%)  |  No-Persist={np_hits}/{total} ({np_hits/total*100:.0f}%)")

# Show risk score breakdown for one key example: memory-leak-cart
print(f"\n  --- Detailed: memory-leak-cart (mid-window row) ---")
ml_subset = df[df["label"] == "memory-leak-cart"]
if len(ml_subset) > 0:
    mid_idx = ml_subset.index[len(ml_subset) // 2]
    row = df.loc[mid_idx]
    print(f"  {'Service':<15s} {'Anom(a)':>8s} {'Crit(c)':>8s} {'Pers(p)':>8s} {'Risk(axcxp)':>12s} {'NoPersist(axc)':>15s}")
    print(f"  {'-'*70}")
    for svc in SERVICES:
        a = (row[f"score_{svc}"] - s_min) / (s_max - s_min + 1e-8)
        c = CRITICALITY[svc] / MAX_CRIT
        p = row[f"persist_{svc}"]
        risk = a * c * p
        no_p = a * c
        marker = " <-- TARGET" if svc == "cart" else ""
        print(f"  {svc:<15s} {a:>8.4f} {c:>8.2f} {p:>8.4f} {risk:>12.4f} {no_p:>15.4f}{marker}")

# =====================================================================
# Save results
# =====================================================================
print(f"\n{'=' * 60}")
print("Saving risk-scored data...")
df.to_csv("risk_data.csv", index=False)
print(f"  Saved to risk_data.csv ({len(df)} rows, {df.shape[1]} columns)")

print(f"\n  Total runtime: {time.time()-t0:.1f}s")
print(f"{'=' * 60}")
print("Phase 7 complete!")
