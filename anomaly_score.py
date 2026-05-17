"""
PersistRank — Phase 3: Anomaly Scoring with Isolation Forest
Trains IF on normal data only, scores all rows.
Outputs: df with 'anomaly_score' column, saves to scored_data.csv
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import time

# =====================================================================
# Step 3.1 — Load processed data and separate features from labels
# =====================================================================
print("=" * 60)
print("Phase 3: Anomaly Scoring (Isolation Forest)")
print("=" * 60)

print("\n[3.1] Loading processed data...")
t0 = time.time()
df = pd.read_csv("processed_data.csv")
feature_cols = [c for c in df.columns if c not in ["label", "anomaly_type", "target_service"]]
X = df[feature_cols].values

print(f"  Loaded {len(df)} rows × {len(feature_cols)} features in {time.time()-t0:.1f}s")
print(f"  Normal rows:  {(df['label'] == 'normal').sum()}")
print(f"  Anomaly rows: {(df['label'] != 'normal').sum()}")

# =====================================================================
# Step 3.2 — Normalize features with StandardScaler
# =====================================================================
print("\n[3.2] Normalizing features with StandardScaler...")
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
print(f"  Done. Shape: {Xs.shape}")
print(f"  Mean range: [{Xs.mean(axis=0).min():.4f}, {Xs.mean(axis=0).max():.4f}]")
print(f"  Std range:  [{Xs.std(axis=0).min():.4f}, {Xs.std(axis=0).max():.4f}]")

# =====================================================================
# Step 3.3 — Train Isolation Forest on normal data ONLY
# =====================================================================
print("\n[3.3] Training Isolation Forest on normal data only...")
normal_mask = df["label"] == "normal"
X_normal = Xs[normal_mask]
print(f"  Training samples: {X_normal.shape[0]}")

t1 = time.time()
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42,
    n_jobs=-1  # use all CPU cores
)
model.fit(X_normal)
print(f"  Training completed in {time.time()-t1:.1f}s")

# =====================================================================
# Step 3.4 — Score ALL rows (higher = more anomalous)
# =====================================================================
print("\n[3.4] Scoring all rows...")
t2 = time.time()
raw_scores = -model.decision_function(Xs)  # negate: higher = more anomalous
df["anomaly_score"] = raw_scores
print(f"  Scoring completed in {time.time()-t2:.1f}s")

# =====================================================================
# Step 3.5 — Verify scores make sense
# =====================================================================
print("\n[3.5] Verifying anomaly scores...")

# Normal vs anomaly score comparison
normal_scores = df.loc[df["label"] == "normal", "anomaly_score"]
anomaly_scores = df.loc[df["label"] != "normal", "anomaly_score"]

print(f"\n  {'Metric':<25s} {'Normal':>10s} {'Anomaly':>10s} {'Delta':>10s}")
print(f"  {'-'*55}")
print(f"  {'Mean score':<25s} {normal_scores.mean():>10.4f} {anomaly_scores.mean():>10.4f} {anomaly_scores.mean()-normal_scores.mean():>+10.4f}")
print(f"  {'Median score':<25s} {normal_scores.median():>10.4f} {anomaly_scores.median():>10.4f} {anomaly_scores.median()-normal_scores.median():>+10.4f}")
print(f"  {'Std score':<25s} {normal_scores.std():>10.4f} {anomaly_scores.std():>10.4f}")
print(f"  {'Min score':<25s} {normal_scores.min():>10.4f} {anomaly_scores.min():>10.4f}")
print(f"  {'Max score':<25s} {normal_scores.max():>10.4f} {anomaly_scores.max():>10.4f}")

# Check: anomaly mean should be HIGHER than normal mean
if anomaly_scores.mean() > normal_scores.mean():
    print(f"\n  [PASS] Anomaly scores (mean={anomaly_scores.mean():.4f}) > Normal scores (mean={normal_scores.mean():.4f})")
else:
    print(f"\n  [FAIL] Anomaly scores are NOT higher than normal! Check data loading.")

# Per-anomaly-type breakdown
print(f"\n  Per-anomaly-type mean scores:")
print(f"  {'Anomaly Type':<35s} {'Mean Score':>10s} {'Rows':>8s}")
print(f"  {'-'*55}")
for atype in sorted(df["anomaly_type"].unique()):
    subset = df[df["anomaly_type"] == atype]
    print(f"  {atype:<35s} {subset['anomaly_score'].mean():>10.4f} {len(subset):>8d}")

# Score distribution stats
print(f"\n  Score distribution:")
for pct in [25, 50, 75, 90, 95, 99]:
    print(f"    P{pct}: {np.percentile(df['anomaly_score'], pct):.4f}")

# =====================================================================
# Save results
# =====================================================================
print(f"\n{'=' * 60}")
print("Saving scored data...")
df.to_csv("scored_data.csv", index=False)
print(f"  Saved to scored_data.csv ({len(df)} rows, {df.shape[1]} columns)")
print(f"  New column added: 'anomaly_score'")

# Also save scaler and model for reproducibility
import joblib
joblib.dump(scaler, "scaler.joblib")
joblib.dump(model, "isolation_forest.joblib")
print(f"  Saved scaler.joblib and isolation_forest.joblib")

print(f"\n  Total runtime: {time.time()-t0:.1f}s")
print(f"{'=' * 60}")
print("Phase 3 complete!")
