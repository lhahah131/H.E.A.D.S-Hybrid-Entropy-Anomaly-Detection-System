import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ===============================
# HWCL CONFIGURATION (Editable)
# ===============================

# 1️⃣ STATISTICAL PROFILES
BENIGN_ENT_MEAN = 4.5467
BENIGN_ENT_STD  = 0.2660
ANOMALY_ENT_MEAN = 5.2537
ANOMALY_ENT_STD  = 0.4599

# 2️⃣ DYNAMIC BOUNDS CALCULATION (v2.2 Calibrated)
# ENT_NORMAL = benign_mean + 0.75 * benign_std (Dead-zone untuk benign)
# ENT_ANOM   = anomaly_mean - 0.25 * anomaly_std (Target spesifik anomali dengan sedikit ruang borderline)
ENT_NORMAL = BENIGN_ENT_MEAN + (0.75 * BENIGN_ENT_STD) # ~4.746
ENT_ANOM   = ANOMALY_ENT_MEAN - (0.25 * ANOMALY_ENT_STD) # ~5.138

NP_NORMAL  = 0.01
NP_ANOM    = 0.08 # Dilebarkan agar benign yang berisik (0.05-0.07) tidak mendapat penalti maksimal

ASCII_HIGH = 0.90
ASCII_LOW  = 0.70

# Weights (Total = 1.0)
W_ENT   = 0.60
W_NP    = 0.30
W_ASCII = 0.10


def apply_hwcl_confirmation(
    preds: np.ndarray,
    df: pd.DataFrame,
    raw_scores: np.ndarray = None, # Dibiarkan untuk backward compatibility dengan script audit
    cumulative_threshold: float = 0.60,
    calibrated_bounds: dict = None # <--- NEW Auto-Calibration Injection
) -> np.ndarray:
    """
    Hybrid Weighted Confirmation Layer (HWCL v2.3 - High Sensitivity)
    Menggunakan cumulative scoring dengan batas anomali yang dikalibrasi ulang.

    Parameters:
    - preds: hasil prediksi Isolation Forest (1 = anomaly, 0 = benign)
    - df: dataframe fitur
    - cumulative_threshold: batas konfirmasi anomaly

    Returns:
    - final_preds: hasil prediksi setelah HWCL
    """

    final_preds = preds.copy()
    mitigated_count = 0
    confirmed_count = 0
    hwcl_scores_list = []
    
    # Auto-Calibration Overrides
    ent_normal = ENT_NORMAL
    ent_anom = ENT_ANOM
    np_normal = NP_NORMAL
    np_anom = NP_ANOM
    
    if calibrated_bounds:
        ent_normal = calibrated_bounds.get("ENT_NORMAL", ENT_NORMAL)
        ent_anom = calibrated_bounds.get("ENT_ANOM", ENT_ANOM)
        np_normal = calibrated_bounds.get("NP_NORMAL", NP_NORMAL)
        np_anom = calibrated_bounds.get("NP_ANOM", NP_ANOM)

    for idx in range(len(final_preds)):
        # Hanya evaluasi anomaly hasil Isolation Forest
        if final_preds[idx] == 1:
            row = df.iloc[idx]

            # ===============================
            # 1️⃣ Entropy Score (0-1) [Calibrated Bounds]
            # ===============================
            entropy = row.get("global_entropy", 0.0)
            
            # Linear scaling dengan sensitivitas tinggi
            ent_score = np.clip(
                (entropy - ent_normal) / max((ent_anom - ent_normal), 1e-6),
                0, 1
            )
            weighted_ent = ent_score * W_ENT

            # ===============================
            # 2️⃣ Non-Printable Score (0-1) [Calibrated Bounds]
            # ===============================
            np_ratio = row.get("non_printable_ratio", 0.0)
            np_score = np.clip(
                (np_ratio - np_normal) / max((np_anom - np_normal), 1e-6),
                0, 1
            )
            weighted_np = np_score * W_NP

            # ===============================
            # 3️⃣ ASCII Score (Reversed Logic)
            # ===============================
            ascii_ratio = row.get("ascii_ratio", 1.0)
            ascii_score = np.clip(
                (ASCII_HIGH - ascii_ratio) / max((ASCII_HIGH - ASCII_LOW), 1e-6),
                0, 1
            )
            weighted_ascii = ascii_score * W_ASCII

            # ===============================
            # Cumulative Risk Score
            # ===============================
            # Batal menggunakan skor IF, 100% menggunakan 3 fitur fundamental
            hwcl_score = weighted_ent + weighted_np + weighted_ascii

            logger.debug(
                f"[HWCL] idx={idx} | "
                f"Ent={entropy:.3f} ({weighted_ent:.3f}) | "
                f"NP={np_ratio:.3f} ({weighted_np:.3f}) | "
                f"ASCII={ascii_ratio:.3f} ({weighted_ascii:.3f}) | "
                f"Total={hwcl_score:.3f}"
            )

            # ===============================
            # Decision Boundary
            # ===============================
            hwcl_scores_list.append(hwcl_score)
            if hwcl_score >= cumulative_threshold:
                confirmed_count += 1
            else:
                final_preds[idx] = 0
                mitigated_count += 1

    if hwcl_scores_list:
        p25, p50, p75, p90 = np.percentile(hwcl_scores_list, [25, 50, 75, 90])
        print(
            f"\n[HWCL DIAGNOSTICS] Anomaly Score Distribution (N={len(hwcl_scores_list)}):\n"
            f"  -> P25: {p25:.3f}\n"
            f"  -> Median(P50): {p50:.3f}\n"
            f"  -> P75: {p75:.3f}\n"
            f"  -> P90: {p90:.3f}\n"
        )
        logger.info(
            f"[HWCL] Anomaly Score Distribution (N={len(hwcl_scores_list)}): "
            f"P25={p25:.3f} | Median(P50)={p50:.3f} | P75={p75:.3f} | P90={p90:.3f}"
        )

    logger.info(
        f"[HWCL] Confirmed: {confirmed_count} | "
        f"Mitigated: {mitigated_count} | "
        f"Threshold: {cumulative_threshold}"
    )

    return final_preds