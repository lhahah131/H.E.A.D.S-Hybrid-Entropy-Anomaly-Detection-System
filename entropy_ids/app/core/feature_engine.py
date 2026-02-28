import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to raw dataframe.
    """
    df_feat = df.copy()
    existing_cols = df_feat.columns.tolist()

    if "text_length" not in existing_cols:
        df_feat["text_length"] = df_feat["file_size"] / (df_feat["line_count"].replace(0, 1))

    df_feat["entropy_x_nonprint"] = df_feat["global_entropy"] * df_feat["non_printable_ratio"]
    df_feat["entropy_div_ascii"] = df_feat["global_entropy"] / df_feat["ascii_ratio"].replace(0, 0.0001)
    df_feat["bytestd_div_bytemean"] = df_feat["byte_std"] / df_feat["byte_mean"].replace(0, 0.0001)

    return df_feat

def extract_features_and_labels(df: pd.DataFrame):
    """
    Extracts the updated strong features and ground truth labels.
    """
    feature_cols = [
        "global_entropy", "block_mean_entropy", "block_std_entropy",
        "non_printable_ratio", "ascii_ratio", 
        "byte_mean", "byte_std", "byte_skewness",
        "entropy_x_nonprint", "entropy_div_ascii", "bytestd_div_bytemean",
        "is_executable", "num_sections", "suspicious_api_count", 
        "has_high_entropy_section", "suspicious_string_count"
    ]
    
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[feature_cols].values

    if "label" in df.columns:
        y_true = df["label"].fillna(0).astype(int).values
    else:
        file_ids = df["file_id"].values if "file_id" in df.columns else np.zeros(len(df))
        y_true = np.where(pd.Series(file_ids).astype(str).str.startswith("benign_"), 0, 1).astype(int)

    logger.info(f"Extracted {len(feature_cols)} features for {len(X)} samples.")
    return X, y_true, df
