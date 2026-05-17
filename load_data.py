"""
PersistRank — Step 1: Load RS-Anomic Dataset
Loads all 12 services × 19 metrics for normal + 14 anomaly scenarios.
Outputs a single DataFrame: ~114K rows × 228 features + label column.
"""
import pandas as pd
import numpy as np
import os

# === Configuration ===
BASE_PATH = "rs-anomic"
CADVISOR_NORMAL = os.path.join(BASE_PATH, "normal_data", "normal_data", "cAdvisor")
CADVISOR_ANOMALY = os.path.join(BASE_PATH, "anomaly_data", "anomaly_data", "cAdvisor")

# 12 services in RS-Anomic (from constants.py)
SERVICES = ['payment', 'shipping', 'redis', 'mongodb', 'dispatch',
            'rabbitmq', 'user', 'mysql', 'catalogue', 'ratings', 'web', 'cart']

# 19 cAdvisor metrics per service
METRIC_COLS = [
    'container_fs_usage_bytes',
    'container_memory_rss',
    'container_memory_usage_bytes',
    'container_memory_working_set_bytes',
    'container_cpu_system_seconds_total',
    'container_cpu_usage_seconds_total',
    'container_cpu_user_seconds_total',
    'container_network_receive_bytes_total',
    'container_network_receive_errors_total',
    'container_network_receive_packets_dropped_total',
    'container_network_receive_packets_total',
    'container_network_transmit_bytes_total',
    'container_network_transmit_errors_total',
    'container_network_transmit_packets_dropped_total',
    'container_network_transmit_packets_total',
    'container_fs_io_time_seconds_total',
    'container_memory_failures_total',
    'container_memory_failcnt',
    'container_fs_write_seconds_total'
]

# Cumulative columns that need .diff()
CUMULATIVE_COLS = [
    'container_cpu_system_seconds_total',
    'container_cpu_usage_seconds_total',
    'container_cpu_user_seconds_total',
    'container_network_receive_bytes_total',
    'container_network_receive_errors_total',
    'container_network_receive_packets_dropped_total',
    'container_network_receive_packets_total',
    'container_network_transmit_bytes_total',
    'container_network_transmit_errors_total',
    'container_network_transmit_packets_dropped_total',
    'container_network_transmit_packets_total',
    'container_fs_io_time_seconds_total',
    'container_memory_failures_total',
    'container_memory_failcnt',
    'container_fs_write_seconds_total'
]

# 14 anomaly scenarios (folder names)
ANOMALIES = [
    'rt-delay-catalogue',
    'packetloss-user',
    'low-bandwidth-user',
    'high-cpu-dispatch',
    'high-latency-user-2',
    'high-load-1500',
    'out-of-order-packets-user-2',
    'high-latency-user',
    'service-down-payment',
    'out-of-order-packets-user',
    'packetloss-user-2',
    'high-fileIO-payment',
    'memory-leak-cart',
    'low-bandwidth-user-2'
]


def load_service_csv(csv_path, service_name):
    """Load one service CSV file and prefix columns with service name."""
    try:
        df = pd.read_csv(csv_path)
        # Keep only metric columns that exist
        available_cols = [c for c in METRIC_COLS if c in df.columns]
        df = df[available_cols]

        # Handle cumulative columns: take difference
        for col in CUMULATIVE_COLS:
            if col in df.columns:
                df[col] = df[col].diff()
                df.loc[df[col] < 0, col] = 0

        # Drop first 3 rows (noisy after diff)
        df = df.iloc[3:].reset_index(drop=True)

        # Interpolate missing values
        df = df.interpolate(method='linear', limit_direction='both')
        df = df.fillna(0)

        # Prefix columns with service name
        df.columns = [f"{service_name}_{col}" for col in df.columns]
        return df

    except Exception as e:
        print(f"  Warning: Could not load {csv_path}: {e}")
        # Return zeros if file missing
        cols = [f"{service_name}_{c}" for c in METRIC_COLS]
        return pd.DataFrame(0, index=range(100), columns=cols)


def load_scenario(cadvisor_path, anomaly_name=""):
    """Load all 12 services for one scenario (normal or specific anomaly)."""
    dfs = []
    min_rows = float('inf')

    for svc in SERVICES:
        if anomaly_name:
            csv_path = os.path.join(cadvisor_path, anomaly_name, f"{svc}.csv")
        else:
            csv_path = os.path.join(cadvisor_path, f"{svc}.csv")

        df_svc = load_service_csv(csv_path, svc)
        dfs.append(df_svc)
        min_rows = min(min_rows, len(df_svc))

    # Align all services to same length
    dfs_aligned = [df.iloc[:min_rows].reset_index(drop=True) for df in dfs]

    # Concatenate horizontally: 12 services × 19 metrics = 228 columns
    combined = pd.concat(dfs_aligned, axis=1)
    return combined


def load_all_data():
    """Load normal + all 14 anomaly scenarios into one DataFrame with labels."""

    print("=" * 60)
    print("Loading RS-Anomic Dataset")
    print("=" * 60)

    # --- Load normal data ---
    print("\n[1/2] Loading normal data...")
    normal_df = load_scenario(CADVISOR_NORMAL)
    normal_df["label"] = "normal"
    normal_df["anomaly_type"] = "normal"
    normal_df["target_service"] = "none"
    print(f"  Normal rows: {len(normal_df)}, columns: {normal_df.shape[1]}")

    # --- Load anomaly data ---
    print("\n[2/2] Loading anomaly data...")
    anomaly_dfs = []

    for anom in ANOMALIES:
        print(f"  Loading: {anom}...", end=" ")
        anom_df = load_scenario(CADVISOR_ANOMALY, anom)

        # Parse anomaly name to get type and target service
        # e.g., "memory-leak-cart" → type="memory-leak", target="cart"
        # e.g., "high-cpu-dispatch" → type="high-cpu", target="dispatch"
        # e.g., "service-down-payment" → type="service-down", target="payment"
        parts = anom.split("-")

        # Extract target service (last meaningful part)
        if anom == "high-load-1500":
            anomaly_type = "high-load"
            target_service = "web"  # high load affects the whole system, web is entry
        elif anom.endswith("-2"):
            # e.g., "high-latency-user-2" → type="high-latency", target="user"
            target_service = parts[-2]
            anomaly_type = "-".join(parts[:-2])
        else:
            target_service = parts[-1]
            anomaly_type = "-".join(parts[:-1])

        anom_df["label"] = anom
        anom_df["anomaly_type"] = anomaly_type
        anom_df["target_service"] = target_service
        anomaly_dfs.append(anom_df)
        print(f"{len(anom_df)} rows")

    all_anomaly = pd.concat(anomaly_dfs, ignore_index=True)
    print(f"\n  Total anomaly rows: {len(all_anomaly)}")

    # --- Combine all ---
    full_df = pd.concat([normal_df, all_anomaly], ignore_index=True)

    # Get feature columns (everything except label columns)
    feature_cols = [c for c in full_df.columns
                    if c not in ["label", "anomaly_type", "target_service"]]

    print(f"\n{'=' * 60}")
    print(f"DATASET READY")
    print(f"  Total rows:     {len(full_df)}")
    print(f"  Feature cols:   {len(feature_cols)}")
    print(f"  Normal rows:    {len(full_df[full_df['label'] == 'normal'])}")
    print(f"  Anomaly rows:   {len(full_df[full_df['label'] != 'normal'])}")
    print(f"  Anomaly types:  {full_df['anomaly_type'].nunique() - 1}")
    print(f"  Target services:{full_df[full_df['target_service'] != 'none']['target_service'].unique()}")
    print(f"{'=' * 60}")

    return full_df, feature_cols


if __name__ == "__main__":
    df, feature_cols = load_all_data()

    print("\n\nLabel distribution:")
    print(df["label"].value_counts())

    print("\n\nAnomaly type distribution:")
    print(df["anomaly_type"].value_counts())

    print("\n\nTarget service distribution:")
    print(df["target_service"].value_counts())

    print(f"\n\nSample feature columns (first 10):")
    print(feature_cols[:10])

    # Save for quick loading later
    print("\n\nSaving to processed_data.csv...")
    df.to_csv("processed_data.csv", index=False)
    print("Done! File saved as processed_data.csv")
