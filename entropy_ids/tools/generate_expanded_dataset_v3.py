import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Project root on sys.path so we can import app modules ──────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR  = os.path.join(ROOT_DIR, "app")
sys.path.insert(0, APP_DIR)

MASTER_CSV   = os.path.join(ROOT_DIR, "data", "raw",       "master_features.csv")
OUTPUT_CSV   = os.path.join(ROOT_DIR, "data", "synthetic", "expanded_dataset_v3.csv")
MODEL_PATH   = os.path.join(ROOT_DIR, "models", "saved_models", "iso_v3.pkl")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

FEATURE_COLS = [
    "file_size", "global_entropy", "block_mean_entropy", "block_std_entropy",
    "block_entropy_range", "byte_skewness", "byte_kurtosis",
    "ascii_ratio", "digit_ratio", "uppercase_ratio", "lowercase_ratio",
    "symbol_ratio", "null_ratio", "non_printable_ratio",
    "byte_mean", "byte_std", "line_count", "avg_line_length",
    "empty_line_ratio", "max_line_length",
    "hist_bin_0", "hist_bin_1", "hist_bin_2", "hist_bin_3",
    "hist_bin_4", "hist_bin_5", "hist_bin_6", "hist_bin_7"
]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Compute real distribution stats from master dataset
# ══════════════════════════════════════════════════════════════════════════════
def compute_real_stats(master_df: pd.DataFrame) -> dict:
    b = master_df[master_df["label"] == 0]
    a = master_df[master_df["label"] == 1]
    return {
        "benign":  {"mean": b["global_entropy"].mean(), "std": b["global_entropy"].std()},
        "anomaly": {"mean": a["global_entropy"].mean(), "std": a["global_entropy"].std()},
        "benign_file_size_mean":  np.log(b["file_size"].mean()),
        "anomaly_file_size_mean": np.log(a["file_size"].mean()),
        "benign_digit_mean":   b["digit_ratio"].mean(),
        "anomaly_digit_mean":  a["digit_ratio"].mean(),
        "benign_symbol_mean":  b["symbol_ratio"].mean(),
        "anomaly_symbol_mean": a["symbol_ratio"].mean(),
        "benign_np_mean":  b["non_printable_ratio"].mean(),
        "anomaly_np_mean": a["non_printable_ratio"].mean(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Generate one benign row (correlated features)
# ══════════════════════════════════════════════════════════════════════════════
def make_benign_row(fid: str, stats: dict, ambiguous: bool = False) -> dict:
    row = {"file_id": fid, "label": 0}

    if ambiguous:
        # Suspicious-but-benign: higher entropy, high digit/symbol
        ent = np.clip(np.random.normal(5.8, 0.5), 5.5, 6.5)
        row["digit_ratio"]  = np.clip(np.random.uniform(0.15, 0.35), 0, 1)
        row["symbol_ratio"] = np.clip(np.random.uniform(0.12, 0.25), 0, 1)
    else:
        ent = np.clip(
            np.random.normal(stats["benign"]["mean"], stats["benign"]["std"] * 1.2),
            3.2, 5.8
        )
        row["digit_ratio"]  = np.clip(
            stats["benign_digit_mean"] + np.random.normal(0, 0.015), 0, 1
        )
        row["symbol_ratio"] = np.clip(
            stats["benign_symbol_mean"] + ent * 0.008 + np.random.normal(0, 0.01), 0, 1
        )

    row["global_entropy"]      = ent
    row["block_mean_entropy"]  = np.clip(ent - np.random.uniform(0.05, 0.35), 2.5, ent)
    row["block_std_entropy"]   = np.clip(np.random.normal(0.22, 0.12), 0.01, 1.2)
    row["block_entropy_range"] = row["block_std_entropy"] * np.random.uniform(1.8, 4.0)

    # STEP 3 — Correlated bytes stats
    row["byte_std"]  = np.clip(ent * np.random.uniform(6, 9), 20, 60)   # correlated
    row["byte_mean"] = np.clip(np.random.normal(73, 6), 55, 90)
    row["byte_skewness"] = np.clip(np.random.normal(-0.45, 0.2), -1.2, 0.3)
    row["byte_kurtosis"] = np.clip(np.random.normal(-1.55, 0.25), -2.0, -0.5)

    np_ratio = np.clip(stats["benign_np_mean"] + np.random.normal(0, 0.01), 0.0, 0.08)
    row["non_printable_ratio"] = np_ratio
    row["ascii_ratio"] = np.clip(1.0 - np_ratio + np.random.normal(0, 0.005), 0.85, 1.0)
    row["null_ratio"]  = 0.0
    row["uppercase_ratio"] = np.clip(np.random.uniform(0.002, 0.04), 0, 1)
    row["lowercase_ratio"] = np.clip(np.random.uniform(0.30, 0.65), 0, 1)

    file_size = int(np.exp(np.random.normal(stats["benign_file_size_mean"], 0.6)))
    row["file_size"]  = max(file_size, 200)
    line_count = max(1, int(np.random.lognormal(3.5, 0.8)))
    row["line_count"]       = line_count
    row["avg_line_length"]  = row["file_size"] / line_count
    row["empty_line_ratio"] = np.clip(np.random.beta(2, 8), 0, 0.4)
    row["max_line_length"]  = row["avg_line_length"] * np.random.uniform(1.5, 5.0)

    # Hist bins — normalized to sum ~1
    raw_bins = np.random.dirichlet(np.ones(8) * 2)
    for j in range(8):
        row[f"hist_bin_{j}"] = float(raw_bins[j])

    return row


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Generate one anomaly row (correlated features)
# ══════════════════════════════════════════════════════════════════════════════
def make_anomaly_row(fid: str, stats: dict, ambiguous: bool = False) -> dict:
    row = {"file_id": fid, "label": 1}

    if ambiguous:
        # Low-entropy anomaly: harder to detect
        ent = np.clip(np.random.normal(4.9, 0.3), 4.5, 5.3)
        np_ratio = np.clip(np.random.uniform(0.02, 0.06), 0, 1)
        row["digit_ratio"]  = np.clip(np.random.uniform(0.02, 0.08), 0, 1)
    else:
        ent = np.clip(
            np.random.normal(stats["anomaly"]["mean"] + 0.4, stats["anomaly"]["std"] * 1.3),
            4.8, 7.8
        )
        np_ratio = np.clip(
            stats["anomaly_np_mean"] + np.random.normal(0.03, 0.02), 0.0, 0.55
        )
        row["digit_ratio"]  = np.clip(
            stats["anomaly_digit_mean"] + ent * 0.015 + np.random.normal(0, 0.02), 0, 0.5
        )

    row["global_entropy"]      = ent
    row["block_mean_entropy"]  = np.clip(ent - np.random.uniform(0.05, 0.55), 3.5, ent)
    row["block_std_entropy"]   = np.clip(np.random.normal(0.40, 0.18), 0.02, 1.5)
    row["block_entropy_range"] = row["block_std_entropy"] * np.random.uniform(1.4, 3.8)

    # Correlated bytes
    row["byte_std"]  = np.clip(ent * np.random.uniform(6, 9), 28, 80)
    row["byte_mean"] = np.clip(np.random.normal(85, 15), 60, 130)
    row["byte_skewness"] = np.clip(np.random.normal(-0.1, 0.35), -0.9, 0.8)
    row["byte_kurtosis"] = np.clip(np.random.normal(-1.1, 0.4),  -1.9, 0.5)

    row["non_printable_ratio"] = np_ratio
    row["ascii_ratio"] = np.clip(1.0 - np_ratio - np.random.uniform(0, 0.15), 0.2, 1.0)
    row["null_ratio"]  = np.clip(np.random.exponential(0.02), 0, 0.15)
    row["uppercase_ratio"] = np.clip(np.random.uniform(0.06, 0.32), 0, 1)
    row["lowercase_ratio"] = np.clip(np.random.uniform(0.18, 0.55), 0, 1)
    row["symbol_ratio"] = np.clip(
        stats["anomaly_symbol_mean"] + ent * 0.010 + np.random.normal(0, 0.015), 0, 0.35
    )

    file_size = int(np.exp(np.random.normal(stats["anomaly_file_size_mean"], 0.7)))
    row["file_size"]  = max(file_size, 500)
    line_count = max(1, int(np.random.lognormal(2.5, 1.0)))
    row["line_count"]       = line_count
    row["avg_line_length"]  = row["file_size"] / line_count
    row["empty_line_ratio"] = np.clip(np.random.beta(1, 9), 0, 0.2)
    row["max_line_length"]  = row["avg_line_length"] * np.random.uniform(1.1, 3.0)

    raw_bins = np.random.dirichlet(np.ones(8) * 1.2)
    for j in range(8):
        row[f"hist_bin_{j}"] = float(raw_bins[j])

    return row


# ══════════════════════════════════════════════════════════════════════════════
# Build full dataset column-ordered
# ══════════════════════════════════════════════════════════════════════════════
def build_dataframe(rows: list, columns: list) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df[columns]


# ══════════════════════════════════════════════════════════════════════════════
# Diversity filter — vectorised cosine similarity on scaled space
# ══════════════════════════════════════════════════════════════════════════════
def diversity_filter(df: pd.DataFrame, threshold: float = 0.998) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[FEATURE_COLS].values)
    sim_matrix = cosine_similarity(scaled)

    keep = []
    for i in range(len(df)):
        if not keep:
            keep.append(i)
            continue
        max_sim = sim_matrix[i, keep].max()
        if max_sim <= threshold:
            keep.append(i)

    return df.iloc[keep].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Overlap estimator
# ══════════════════════════════════════════════════════════════════════════════
def compute_overlap(df: pd.DataFrame) -> float:
    b_ent = df[df["label"] == 0]["global_entropy"]
    a_ent = df[df["label"] == 1]["global_entropy"]
    b_in_a_range = ((b_ent >= a_ent.min()) & (b_ent <= a_ent.max())).sum()
    total_b = len(b_ent)
    return (b_in_a_range / total_b * 100) if total_b > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    np.random.seed(42)

    # Load master
    master_df = pd.read_csv(MASTER_CSV)
    ALL_COLS  = master_df.columns.tolist()
    stats     = compute_real_stats(master_df)

    print("=" * 60)
    print("  REAL DATA STATS (master_features.csv)")
    print("=" * 60)
    print(f"  Benign  entropy: mean={stats['benign']['mean']:.3f}  std={stats['benign']['std']:.3f}")
    print(f"  Anomaly entropy: mean={stats['anomaly']['mean']:.3f}  std={stats['anomaly']['std']:.3f}")
    print()

    TARGET = 260          # generous pool before cosine filter
    BENIGN_RATIO  = 0.63  # ~65 % benign
    AMBIGUOUS_PCT = 0.12  # 12% ambiguous from synthetic only

    n_benign  = int(TARGET * BENIGN_RATIO)
    n_anomaly = TARGET - n_benign
    n_amb_b   = int(n_benign  * AMBIGUOUS_PCT)
    n_amb_a   = int(n_anomaly * AMBIGUOUS_PCT)

    print(f"  Generating pool: {n_benign} benign ({n_amb_b} ambiguous) + "
          f"{n_anomaly} anomaly ({n_amb_a} ambiguous)")

    rows = []

    # Normal benign
    for i in range(n_benign - n_amb_b):
        rows.append(make_benign_row(f"synth_benign_{i}", stats, ambiguous=False))
    # Ambiguous benign
    for i in range(n_amb_b):
        rows.append(make_benign_row(f"synth_amb_benign_{i}", stats, ambiguous=True))
    # Normal anomaly
    for i in range(n_anomaly - n_amb_a):
        rows.append(make_anomaly_row(f"synth_anomaly_{i}", stats, ambiguous=False))
    # Ambiguous anomaly
    for i in range(n_amb_a):
        rows.append(make_anomaly_row(f"synth_amb_anomaly_{i}", stats, ambiguous=True))

    synth_df = build_dataframe(rows, ALL_COLS)

    # Combine with master (always keep originals)
    combined = pd.concat([master_df, synth_df], ignore_index=True)

    # Diversity filter on just the synthetic portion (original always kept)
    orig_part  = combined.iloc[: len(master_df)].copy()
    synth_part = combined.iloc[len(master_df):].copy()
    synth_filtered = diversity_filter(synth_part, threshold=0.998)

    final_df = pd.concat([orig_part, synth_filtered], ignore_index=True)

    # Trim to 200–300 range with correct class ratio
    if len(final_df) > 300:
        b_pool = final_df[final_df["label"] == 0]
        a_pool = final_df[final_df["label"] == 1]
        n_b = min(len(b_pool), 195)
        n_a = min(len(a_pool),  80)
        final_df = pd.concat([
            b_pool.sample(n_b, random_state=42),
            a_pool.sample(n_a, random_state=42)
        ]).reset_index(drop=True)

    final_df.to_csv(OUTPUT_CSV, index=False)

    # ─── Report ─────────────────────────────────────────────────────────────
    orig_b = (master_df["label"] == 0).sum()
    orig_a = (master_df["label"] == 1).sum()
    fin_b  = (final_df["label"] == 0).sum()
    fin_a  = (final_df["label"] == 1).sum()
    total  = len(final_df)

    ent_b_mean = final_df[final_df["label"] == 0]["global_entropy"].mean()
    ent_b_std  = final_df[final_df["label"] == 0]["global_entropy"].std()
    ent_a_mean = final_df[final_df["label"] == 1]["global_entropy"].mean()
    ent_a_std  = final_df[final_df["label"] == 1]["global_entropy"].std()
    overlap    = compute_overlap(final_df)

    n_amb_added_b = (final_df["file_id"].str.startswith("synth_amb_benign")).sum()
    n_amb_added_a = (final_df["file_id"].str.startswith("synth_amb_anomaly")).sum()

    print()
    print("=" * 60)
    print("  DATASET EXPANSION REPORT  (v3, Realistic)")
    print("=" * 60)
    print(f"  Original samples   : {len(master_df)}")
    print(f"  New benign added   : {fin_b - orig_b}  (ambiguous: {n_amb_added_b})")
    print(f"  New anomaly added  : {fin_a - orig_a}  (ambiguous: {n_amb_added_a})")
    print(f"  Final total        : {total}")
    print(f"  Benign ratio       : {fin_b/total*100:.1f}%")
    print(f"  Anomaly ratio      : {fin_a/total*100:.1f}%")
    print(f"  Mean entropy benign : {ent_b_mean:.3f}  (std={ent_b_std:.3f})")
    print(f"  Mean entropy anomaly: {ent_a_mean:.3f}  (std={ent_a_std:.3f})")
    print(f"  Entropy overlap est : ~{overlap:.1f}%")
    print()

    if ent_a_mean <= ent_b_mean:
        print("  [WARNING] Anomaly mean entropy is NOT > benign mean. Review generation params.")

    if not (15 <= overlap <= 30):
        print(f"  [WARNING] Overlap {overlap:.1f}% is outside target 15–25%. Review distributions.")

    # ─── STEP 3 — Pearson correlation report ────────────────────────────────
    synth_only = final_df[final_df["file_id"].str.startswith("synth")]
    key_cols   = ["global_entropy", "byte_std", "digit_ratio", "symbol_ratio",
                  "non_printable_ratio", "ascii_ratio"]
    if len(synth_only) > 5:
        corr = synth_only[key_cols].corr().round(3)
        print("  Pearson Correlation Matrix (synthetic, key features):")
        print(corr.to_string())
        print()

    print(f"  Saved to: {OUTPUT_CSV}")
    print("=" * 60)

    # ─── STEP 4 — Run evaluation pipeline ───────────────────────────────────
    print()
    print("  Running evaluation pipeline ...")
    run_evaluation(final_df, master_df, ALL_COLS)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Evaluation pipeline
# ══════════════════════════════════════════════════════════════════════════════
def run_evaluation(expanded_df: pd.DataFrame, master_df: pd.DataFrame, all_cols: list):
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
        from sklearn.preprocessing import StandardScaler
        import pickle

        X_exp  = expanded_df[FEATURE_COLS].values.astype(float)
        y_exp  = expanded_df["label"].values
        X_real = master_df[FEATURE_COLS].values.astype(float)
        y_real = master_df["label"].values

        scaler = StandardScaler()
        X_exp_sc  = scaler.fit_transform(X_exp)
        X_real_sc = scaler.transform(X_real)

        # ── 5-Fold CV on expanded dataset ─────────────────────────────────
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1 = []

        for fold, (tr_idx, te_idx) in enumerate(skf.split(X_exp_sc, y_exp)):
            X_tr, X_te = X_exp_sc[tr_idx], X_exp_sc[te_idx]
            y_tr, y_te = y_exp[tr_idx], y_exp[te_idx]

            iso = IsolationForest(n_estimators=300, contamination=0.35, random_state=42)
            iso.fit(X_tr)
            raw = iso.score_samples(X_te)
            pct = np.percentile(raw, 55)
            preds = (raw < pct).astype(int)
            f1 = f1_score(y_te, preds, zero_division=0)
            cv_f1.append(f1)

        mean_cv_f1 = float(np.mean(cv_f1))

        # ── Final model train on full expanded ─────────────────────────────
        iso_final = IsolationForest(n_estimators=300, contamination=0.35, random_state=42)
        iso_final.fit(X_exp_sc)

        # Save as iso_v3 (do NOT overwrite iso_v2)
        with open(MODEL_PATH, "wb") as f:
            import pickle
            pickle.dump({"model": iso_final, "scaler": scaler}, f)

        # ── Inference on master_features.csv ──────────────────────────────
        raw_real   = iso_final.score_samples(X_real_sc)
        pct_real   = np.percentile(raw_real, 55)
        preds_real = (raw_real < pct_real).astype(int)
        real_f1    = f1_score(y_real, preds_real, zero_division=0)

        gap = abs(mean_cv_f1 - real_f1)

        print()
        print("=" * 60)
        print("  EVALUATION RESULTS")
        print("=" * 60)
        print(f"  CV F1 (5-Fold, expanded) : {mean_cv_f1:.4f}")
        print(f"  Real F1 (master_features): {real_f1:.4f}")
        print(f"  Synthetic vs Real Gap    : {gap:.4f}")

        if gap > 0.15:
            print()
            print("  [WARNING] Gap > 0.15 — synthetic distribution may not match real data!")
            print("            Inspect per-feature distributions below:")
            diag = expanded_df[FEATURE_COLS].describe().T[["mean", "std"]]
            diag.columns = ["synth_mean", "synth_std"]
            diag_r = master_df[FEATURE_COLS].describe().T[["mean", "std"]]
            diag_r.columns = ["real_mean", "real_std"]
            print(pd.concat([diag, diag_r], axis=1).round(4).to_string())
        else:
            print("  [OK] Gap < 0.15 — synthetic dataset generalises well.")

        if real_f1 < 0.70:
            print(f"  [WARNING] Real F1 {real_f1:.4f} < 0.70 target. Investigate threshold / contamination.")
        else:
            print(f"  [OK] Real F1 {real_f1:.4f} meets ≥ 0.70 target.")

        print(f"\n  Model saved: {MODEL_PATH}")
        print("=" * 60)

    except Exception as e:
        print(f"  [ERROR] Evaluation failed: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
