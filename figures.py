"""
PersistRank — Phase 9: Generate Publication-Quality Figures
Creates all figures for the EBISION 2026 paper.
Saves as PNG (300 DPI) for paper insertion.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from criticality import CRITICALITY, MAX_CRIT
import time

# =====================================================================
# Setup
# =====================================================================
print("=" * 60)
print("Phase 9: Generate Publication-Quality Figures")
print("=" * 60)

t0 = time.time()

# Load data
print("\n[9.0] Loading data...")
df = pd.read_csv("persistence_data.csv")
results_main = pd.read_csv("results_main.csv", index_col=0)
results_type = pd.read_csv("results_per_type.csv")

SERVICES = list(CRITICALITY.keys())
s_min = df["anomaly_score"].min()
s_max = df["anomaly_score"].max()

# Style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'persistrank': '#2196F3',
    'no_persist': '#FF9800',
    'no_criticality': '#9C27B0',
    'anom_only': '#4CAF50',
    'crit_only': '#F44336',
    'random': '#9E9E9E',
}

# =====================================================================
# Figure 2: Bar Chart — P@K Comparison (Main Results)
# =====================================================================
print("\n[9.1] Figure 2: P@K bar chart...")

fig, ax = plt.subplots(figsize=(10, 5))

methods = ['persistrank', 'no_persist', 'no_criticality', 'anom_only', 'crit_only', 'random']
labels = ['PersistRank\n(Ours)', 'No Persistence\n(AlertRank)', 'No\nCriticality', 'Anomaly\nOnly', 'Criticality\nOnly', 'Random']
k_metrics = ['P@1', 'P@3', 'P@5']
x = np.arange(len(methods))
width = 0.25

for i, k in enumerate(k_metrics):
    vals = [results_main.loc[m, k] for m in methods]
    bars = ax.bar(x + i * width, vals, width, label=k,
                  color=[COLORS[m] for m in methods], alpha=0.4 + i * 0.25,
                  edgecolor='white', linewidth=0.5)
    # Add value labels on bars
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_ylabel('Precision')
ax.set_title('Alert Prioritization Performance: Precision@K Across Methods')
ax.set_xticks(x + width)
ax.set_xticklabels(labels)
ax.set_ylim(0, 0.85)
ax.legend(title='Metric', loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig2_bar_p_at_k.png')
plt.close()
print("  Saved fig2_bar_p_at_k.png")

# =====================================================================
# Figure 3: Persistence Over Time — Memory Leak vs CPU Spike
# =====================================================================
print("\n[9.2] Figure 3: Persistence over time...")

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

# Memory leak (cart) — should build up over time
ml_df = df[df["label"] == "memory-leak-cart"].reset_index(drop=True)
ax1 = axes[0]
ax1.plot(ml_df.index, ml_df["persist_cart"], color='#E53935', linewidth=1.5, label='cart (target)')
ax1.plot(ml_df.index, ml_df["persist_payment"], color='#1E88E5', linewidth=1, alpha=0.6, label='payment')
ax1.plot(ml_df.index, ml_df["persist_user"], color='#43A047', linewidth=1, alpha=0.6, label='user')
ax1.set_ylabel('Persistence Score')
ax1.set_title('Memory Leak (cart) - Persistence Builds Over Time')
ax1.legend(loc='upper left', fontsize=9)
ax1.set_ylim(-0.05, 1.1)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

# CPU spike (dispatch) — more bursty
cpu_df = df[df["label"] == "high-cpu-dispatch"].reset_index(drop=True)
ax2 = axes[1]
ax2.plot(cpu_df.index, cpu_df["persist_dispatch"], color='#E53935', linewidth=1.5, label='dispatch (target)')
ax2.plot(cpu_df.index, cpu_df["persist_payment"], color='#1E88E5', linewidth=1, alpha=0.6, label='payment')
ax2.plot(cpu_df.index, cpu_df["persist_user"], color='#43A047', linewidth=1, alpha=0.6, label='user')
ax2.set_xlabel('Time Window')
ax2.set_ylabel('Persistence Score')
ax2.set_title('High CPU (dispatch) - Persistence Pattern')
ax2.legend(loc='upper left', fontsize=9)
ax2.set_ylim(-0.05, 1.1)
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('fig3_persistence_over_time.png')
plt.close()
print("  Saved fig3_persistence_over_time.png")

# =====================================================================
# Figure 4: Per-Anomaly-Type P@3 (Grouped Bar Chart)
# =====================================================================
print("\n[9.3] Figure 4: Per-anomaly-type P@3 grouped bar...")

fig, ax = plt.subplots(figsize=(12, 5))

types = results_type["anomaly_type"].values
pr_vals = results_type["P@3_persistrank"].values
ar_vals = results_type["P@3_no_persist"].values

x = np.arange(len(types))
width = 0.35

bars1 = ax.bar(x - width/2, pr_vals, width, label='PersistRank (Ours)',
               color='#2196F3', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, ar_vals, width, label='AlertRank-style (No Persist)',
               color='#FF9800', edgecolor='white', linewidth=0.5)

# Highlight improvements
for i in range(len(types)):
    delta = pr_vals[i] - ar_vals[i]
    if delta > 0.01:
        ax.annotate(f'+{delta:.2f}', xy=(x[i], max(pr_vals[i], ar_vals[i]) + 0.02),
                   ha='center', fontsize=7, color='#2196F3', fontweight='bold')

ax.set_ylabel('Precision@3')
ax.set_title('Per-Anomaly-Type Performance: PersistRank vs AlertRank-style')
ax.set_xticks(x)
ax.set_xticklabels([t.replace('-', '\n') for t in types], fontsize=8)
ax.set_ylim(0, 1.15)
ax.legend(loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig4_per_type_p3.png')
plt.close()
print("  Saved fig4_per_type_p3.png")

# =====================================================================
# Figure 5: Ablation Bar Chart
# =====================================================================
print("\n[9.4] Figure 5: Ablation study bar chart...")

fig, ax = plt.subplots(figsize=(8, 5))

ablation_modes = ['persistrank', 'no_persist', 'no_criticality', 'anom_only']
ablation_labels = [
    'A x C x P\n(Full PersistRank)',
    'A x C\n(No Persistence)',
    'A x P\n(No Criticality)',
    'A\n(Anomaly Only)',
]
ablation_colors = ['#2196F3', '#FF9800', '#9C27B0', '#4CAF50']

p3_vals = [results_main.loc[m, 'P@3'] for m in ablation_modes]

bars = ax.bar(ablation_labels, p3_vals, color=ablation_colors,
              edgecolor='white', linewidth=0.5, width=0.6)

# Value labels
for bar, val in zip(bars, p3_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Precision@3')
ax.set_title('Ablation Study: Contribution of Each Component')
ax.set_ylim(0, 0.65)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add connecting lines showing improvement
for i in range(len(ablation_modes) - 1):
    delta = p3_vals[0] - p3_vals[i+1]
    if delta > 0:
        ax.annotate('', xy=(0, p3_vals[0] - 0.005),
                   xytext=(i+1, p3_vals[i+1] + 0.005),
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.4, lw=1))

plt.tight_layout()
plt.savefig('fig5_ablation.png')
plt.close()
print("  Saved fig5_ablation.png")

# =====================================================================
# Figure 6: Average Persistence by Anomaly Type (Heatmap-style)
# =====================================================================
print("\n[9.5] Figure 6: Persistence heatmap by anomaly type...")

anomaly_types = sorted([t for t in df["anomaly_type"].unique() if t != "normal"])

# Build matrix: anomaly_type x service → average persistence
persist_matrix = np.zeros((len(anomaly_types), len(SERVICES)))
for i, atype in enumerate(anomaly_types):
    subset = df[df["anomaly_type"] == atype]
    for j, svc in enumerate(SERVICES):
        persist_matrix[i, j] = subset[f"persist_{svc}"].mean()

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(persist_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(range(len(SERVICES)))
ax.set_xticklabels(SERVICES, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(anomaly_types)))
ax.set_yticklabels([t.replace('-', ' ') for t in anomaly_types], fontsize=9)

# Add value annotations
for i in range(len(anomaly_types)):
    for j in range(len(SERVICES)):
        val = persist_matrix[i, j]
        color = 'white' if val > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

plt.colorbar(im, ax=ax, label='Average Persistence Score', shrink=0.8)
ax.set_title('Persistence Score by Anomaly Type and Service')

plt.tight_layout()
plt.savefig('fig6_persistence_heatmap.png')
plt.close()
print("  Saved fig6_persistence_heatmap.png")

# =====================================================================
# Figure 7: Risk Score Distribution — PersistRank vs AlertRank
# =====================================================================
print("\n[9.6] Figure 7: Risk score distributions...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# For a specific anomaly: memory-leak-cart
ml_df = df[df["label"] == "memory-leak-cart"].copy()

# PersistRank risk scores for target (cart) vs others
target_risk_pr = []
other_risk_pr = []
target_risk_ar = []
other_risk_ar = []

for _, row in ml_df.iterrows():
    for svc in SERVICES:
        a = (row[f"score_{svc}"] - s_min) / (s_max - s_min + 1e-8)
        c = CRITICALITY[svc] / MAX_CRIT
        p = row[f"persist_{svc}"]
        risk_pr = a * c * p
        risk_ar = a * c

        if svc == "cart":
            target_risk_pr.append(risk_pr)
            target_risk_ar.append(risk_ar)
        else:
            other_risk_pr.append(risk_pr)
            other_risk_ar.append(risk_ar)

ax1 = axes[0]
ax1.hist(other_risk_ar, bins=50, alpha=0.6, color='#9E9E9E', label='Other services', density=True)
ax1.hist(target_risk_ar, bins=50, alpha=0.8, color='#FF9800', label='Cart (target)', density=True)
ax1.set_xlabel('Risk Score')
ax1.set_ylabel('Density')
ax1.set_title('AlertRank-style (No Persistence)')
ax1.legend()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2 = axes[1]
ax2.hist(other_risk_pr, bins=50, alpha=0.6, color='#9E9E9E', label='Other services', density=True)
ax2.hist(target_risk_pr, bins=50, alpha=0.8, color='#2196F3', label='Cart (target)', density=True)
ax2.set_xlabel('Risk Score')
ax2.set_ylabel('Density')
ax2.set_title('PersistRank (With Persistence)')
ax2.legend()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle('Memory Leak (Cart): Risk Score Separation — Target vs Other Services', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('fig7_risk_distribution.png')
plt.close()
print("  Saved fig7_risk_distribution.png")

# =====================================================================
# Summary
# =====================================================================
total = time.time() - t0
print(f"\n{'=' * 60}")
print(f"All figures generated in {total:.1f}s")
print(f"{'=' * 60}")
print(f"  fig2_bar_p_at_k.png          - Main P@K comparison (Table 3 visual)")
print(f"  fig3_persistence_over_time.png - Memory leak vs CPU spike persistence")
print(f"  fig4_per_type_p3.png         - Per-anomaly-type PersistRank vs AlertRank")
print(f"  fig5_ablation.png            - Ablation study (component contributions)")
print(f"  fig6_persistence_heatmap.png - Persistence by anomaly type x service")
print(f"  fig7_risk_distribution.png   - Risk score separation (target vs others)")
print(f"\nPhase 9 complete!")
