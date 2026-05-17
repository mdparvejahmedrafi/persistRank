"""
PersistRank — Phase 4: Per-Service Anomaly Scores
Computes an anomaly score per service (12 services x 19 metrics each).
Verifies that the injected service scores highest for each anomaly type.
Outputs: df with score_{service} columns, saves to per_service_data.csv
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time

# =====================================================================
# Step 4.1 — Load scored data and define services
# =====================================================================
print("=" * 60)
print("Phase 4: Per-Service Anomaly Scores")
print("=" * 60)

print("\n[4.1] Loading scored data...")
t0 = time.time()
df = pd.read_csv("scored_data.csv")
feature_cols = [c for c in df.columns
                if c not in ["label", "anomaly_type", "target_service", "anomaly_score"]]
X = df[feature_cols].values

# Re-normalize (need Xs for per-service scoring)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# 12 services in RS-Anomic
SERVICES = ['payment', 'shipping', 'redis', 'mongodb', 'dispatch',
            'rabbitmq', 'user', 'mysql', 'catalogue', 'ratings', 'web', 'cart']

print(f"  Loaded {len(df)} rows x {len(feature_cols)} features in {time.time()-t0:.1f}s")

# =====================================================================
# Step 4.2 — For each service, find its columns
# =====================================================================
print("\n[4.2] Mapping columns to services...")

service_col_map = {}
for svc in SERVICES:
    # Columns are prefixed: {service}_container_...
    cols = [c for c in feature_cols if c.startswith(svc + "_")]
    service_col_map[svc] = cols
    print(f"  {svc:<15s}: {len(cols)} columns")

# Check for unmapped columns
mapped = set()
for cols in service_col_map.values():
    mapped.update(cols)
unmapped = set(feature_cols) - mapped
if unmapped:
    print(f"\n  WARNING: {len(unmapped)} unmapped columns: {list(unmapped)[:5]}...")
else:
    print(f"\n  All {len(feature_cols)} columns mapped to services.")

# =====================================================================
# Step 4.3 — Compute per-service anomaly score
# =====================================================================
print("\n[4.3] Computing per-service anomaly scores...")

for svc in SERVICES:
    cols = service_col_map[svc]
    if not cols:
        # Fallback: use global anomaly score
        df[f"score_{svc}"] = df["anomaly_score"]
        print(f"  {svc}: using global score (no columns found)")
        continue

    col_idx = [feature_cols.index(c) for c in cols]
    # Average absolute normalized value = how unusual this service is
    df[f"score_{svc}"] = np.abs(Xs[:, col_idx]).mean(axis=1)

print(f"  Done. Added {len(SERVICES)} score columns.")

# =====================================================================
# Step 4.4 — Verify per-service scores
# =====================================================================
print("\n[4.4] Verifying per-service scores...")

# For each anomaly type, check if the target service has the highest score
print(f"\n  {'Anomaly Label':<35s} {'Target':<12s} {'Top-1 Svc':<12s} {'Top-2 Svc':<12s} {'Top-3 Svc':<12s} {'Hit?'}")
print(f"  {'-'*95}")

anomaly_labels = [l for l in df["label"].unique() if l != "normal"]
hit_count = 0
total_count = 0

for label in sorted(anomaly_labels):
    subset = df[df["label"] == label]
    target_svc = subset["target_service"].iloc[0]

    # Average score across all rows of this anomaly type
    avg_scores = {}
    for svc in SERVICES:
        avg_scores[svc] = subset[f"score_{svc}"].mean()

    ranked = sorted(avg_scores, key=avg_scores.get, reverse=True)
    hit = target_svc in ranked[:3]
    if hit:
        hit_count += 1
    total_count += 1

    marker = "<-- HIT" if hit else "<-- MISS"
    print(f"  {label:<35s} {target_svc:<12s} {ranked[0]:<12s} {ranked[1]:<12s} {ranked[2]:<12s} {marker}")

print(f"\n  Target service in top-3: {hit_count}/{total_count} ({hit_count/total_count*100:.0f}%)")

# Detailed: per-row hit rate
print(f"\n  Per-row verification (target service in top-3 per individual row):")
row_hits = 0
row_total = 0
anomaly_df = df[df["label"] != "normal"]

for idx, row in anomaly_df.iterrows():
    target_svc = row["target_service"]
    if target_svc == "none":
        continue
    svc_scores = {svc: row[f"score_{svc}"] for svc in SERVICES}
    ranked = sorted(svc_scores, key=svc_scores.get, reverse=True)
    if target_svc in ranked[:3]:
        row_hits += 1
    row_total += 1

print(f"  Row-level target in top-3: {row_hits}/{row_total} ({row_hits/row_total*100:.1f}%)")

# Show score summary table
print(f"\n  Mean per-service scores (normal vs anomaly):")
print(f"  {'Service':<15s} {'Normal':>10s} {'Anomaly':>10s} {'Delta':>10s}")
print(f"  {'-'*45}")
for svc in SERVICES:
    n_score = df.loc[df["label"] == "normal", f"score_{svc}"].mean()
    a_score = df.loc[df["label"] != "normal", f"score_{svc}"].mean()
    print(f"  {svc:<15s} {n_score:>10.4f} {a_score:>10.4f} {a_score - n_score:>+10.4f}")

# =====================================================================
# Save results
# =====================================================================
print(f"\n{'=' * 60}")
print("Saving per-service scored data...")
df.to_csv("per_service_data.csv", index=False)
print(f"  Saved to per_service_data.csv ({len(df)} rows, {df.shape[1]} columns)")
print(f"  New columns: {[f'score_{s}' for s in SERVICES]}")

print(f"\n  Total runtime: {time.time()-t0:.1f}s")
print(f"{'=' * 60}")
print("Phase 4 complete!")
