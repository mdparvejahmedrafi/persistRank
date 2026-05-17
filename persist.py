"""
PersistRank — Phase 6: Persistence Score
Counts consecutive anomalous windows per service.
This is the CORE CONTRIBUTION of PersistRank.
Memory leaks build persistence over many windows; CPU spikes don't.
Outputs: df with persist_{service} columns, saves to persistence_data.csv
"""
import pandas as pd
import numpy as np
import time

# =====================================================================
# Step 6.1 — Load per-service data and define thresholds
# =====================================================================
print("=" * 60)
print("Phase 6: Persistence Score (Core Contribution)")
print("=" * 60)

print("\n[6.1] Loading per-service scored data...")
t0 = time.time()
df = pd.read_csv("per_service_data.csv")

SERVICES = ['payment', 'shipping', 'redis', 'mongodb', 'dispatch',
            'rabbitmq', 'user', 'mysql', 'catalogue', 'ratings', 'web', 'cart']

print(f"  Loaded {len(df)} rows in {time.time()-t0:.1f}s")

# Define threshold: 75th percentile of NORMAL data scores per service
print("\n  Computing per-service thresholds (P75 of normal data)...")
thresholds = {}
normal_df = df[df["label"] == "normal"]

print(f"  {'Service':<15s} {'P75 Threshold':>15s}")
print(f"  {'-'*30}")
for svc in SERVICES:
    normal_scores = normal_df[f"score_{svc}"]
    thresholds[svc] = normal_scores.quantile(0.75)
    print(f"  {svc:<15s} {thresholds[svc]:>15.4f}")

# =====================================================================
# Step 6.2 — Count consecutive anomalous windows per service
# =====================================================================
print("\n[6.2] Computing persistence scores...")

CAP = 10  # max windows for normalization (hyperparameter)

for svc in SERVICES:
    col = f"score_{svc}"
    thresh = thresholds[svc]
    is_anomalous = (df[col] > thresh).astype(int).values

    persistence = np.zeros(len(df))
    count = 0
    for i in range(len(is_anomalous)):
        if is_anomalous[i]:
            count += 1
        else:
            count = 0
        persistence[i] = min(count / CAP, 1.0)

    df[f"persist_{svc}"] = persistence

print(f"  Done. Added {len(SERVICES)} persistence columns (CAP={CAP}).")

# =====================================================================
# Step 6.3 — Verify persistence makes sense
# =====================================================================
print("\n[6.3] Verifying persistence scores...")

# Per anomaly type: average persistence across all services
print(f"\n  Average persistence by anomaly type:")
print(f"  {'Anomaly Type':<35s} {'Avg Persist':>12s} {'Max Persist':>12s} {'Rows':>8s}")
print(f"  {'-'*70}")

type_persist = {}
for atype in sorted(df["anomaly_type"].unique()):
    subset = df[df["anomaly_type"] == atype]
    # Average persistence across all services
    persist_cols = [f"persist_{svc}" for svc in SERVICES]
    avg_p = subset[persist_cols].mean().mean()
    max_p = subset[persist_cols].max().max()
    type_persist[atype] = avg_p
    print(f"  {atype:<35s} {avg_p:>12.4f} {max_p:>12.4f} {len(subset):>8d}")

# Key insight: memory-leak should have HIGH persistence, high-cpu should be LOWER
print(f"\n  Key comparisons:")
ml = type_persist.get("memory-leak", 0)
hc = type_persist.get("high-cpu", 0)
sd = type_persist.get("service-down", 0)
pl = type_persist.get("packetloss", 0)
hl = type_persist.get("high-latency", 0)

print(f"    memory-leak  persistence: {ml:.4f}")
print(f"    high-cpu     persistence: {hc:.4f}")
print(f"    service-down persistence: {sd:.4f}")
print(f"    packetloss   persistence: {pl:.4f}")
print(f"    high-latency persistence: {hl:.4f}")

if ml > hc:
    print(f"\n  [PASS] memory-leak ({ml:.4f}) > high-cpu ({hc:.4f})")
else:
    print(f"\n  [NOTE] memory-leak ({ml:.4f}) vs high-cpu ({hc:.4f}) - check data ordering")

# Per anomaly: show persistence of the TARGET service specifically
print(f"\n  Persistence of TARGET service per anomaly:")
print(f"  {'Anomaly Label':<35s} {'Target':<12s} {'Target Persist':>15s} {'Avg Other':>12s}")
print(f"  {'-'*75}")

anomaly_labels = [l for l in df["label"].unique() if l != "normal"]
for label in sorted(anomaly_labels):
    subset = df[df["label"] == label]
    target = subset["target_service"].iloc[0]
    if target == "none":
        continue
    target_persist = subset[f"persist_{target}"].mean()
    other_persist = np.mean([subset[f"persist_{s}"].mean() for s in SERVICES if s != target])
    marker = " <-- GOOD" if target_persist > other_persist else ""
    print(f"  {label:<35s} {target:<12s} {target_persist:>15.4f} {other_persist:>12.4f}{marker}")

# Distribution of persistence values
print(f"\n  Persistence distribution (all services, all rows):")
all_persist = np.concatenate([df[f"persist_{svc}"].values for svc in SERVICES])
print(f"    Zero (no persistence):  {(all_persist == 0).sum() / len(all_persist) * 100:.1f}%")
print(f"    Low  (0 < p <= 0.3):    {((all_persist > 0) & (all_persist <= 0.3)).sum() / len(all_persist) * 100:.1f}%")
print(f"    Med  (0.3 < p <= 0.7):  {((all_persist > 0.3) & (all_persist <= 0.7)).sum() / len(all_persist) * 100:.1f}%")
print(f"    High (0.7 < p < 1.0):   {((all_persist > 0.7) & (all_persist < 1.0)).sum() / len(all_persist) * 100:.1f}%")
print(f"    Max  (p == 1.0):        {(all_persist == 1.0).sum() / len(all_persist) * 100:.1f}%")

# =====================================================================
# Save results
# =====================================================================
print(f"\n{'=' * 60}")
print("Saving persistence data...")
df.to_csv("persistence_data.csv", index=False)
print(f"  Saved to persistence_data.csv ({len(df)} rows, {df.shape[1]} columns)")
print(f"  New columns: {[f'persist_{s}' for s in SERVICES]}")

print(f"\n  Total runtime: {time.time()-t0:.1f}s")
print(f"{'=' * 60}")
print("Phase 6 complete!")
