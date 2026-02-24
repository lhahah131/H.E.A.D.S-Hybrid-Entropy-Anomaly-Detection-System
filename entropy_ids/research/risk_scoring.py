import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest

# Adjusted for relocation to /research/
DATA_PATH = "../data/features/expanded_dataset_v3.csv"

# ===============
# LOAD DATA
# ===============
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    DATA_PATH = "../data/features/master_features.csv"
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

file_ids = df["file_id"]
X = df.select_dtypes(include=[np.number])
if "label" in X.columns:
    X = X.drop(columns=["label"])

# ====================
# SCALING
# ====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso = IsolationForest(
    n_estimators=100,
    contamination=0.12,
    random_state=42
)

iso.fit(X_scaled)
# Lower scores correspond to more anomalous instances
anomaly_scores = iso.decision_function(X_scaled)

# ML-based score component (0-70 range)
minmax = MinMaxScaler()
scaled_scores = minmax.fit_transform((-anomaly_scores).reshape(-1, 1)).flatten()
ml_scores = scaled_scores * 70

# ====================
# RULE-BASED SCORING
# ====================

def rule_score(row):
    score = 0
    # Weighted heuristics
    if row.get("non_printable_ratio", 0) > 0.05:
        score += 3
    if row.get("avg_line_length", 0) > 60:
        score += 3
    if row.get("global_entropy", 0) > 5.0:
        score += 4
    if row.get("block_std_entropy", 0) > 0.5:
        score += 4
    return score

rule_scores = df.apply(rule_score, axis=1)

# ====================
# FINAL RISK SCORES
# ====================

final_scores = ml_scores + rule_scores

# ======================
# EXPLAINABILITY LAYER
# ======================
def generate_explanation(row, feature_means, feature_stds):
    z_scores = {}
    for feature in feature_means.index:
        if feature in row.index and feature_stds[feature] > 0:
            z = abs((row[feature] - feature_means[feature]) / feature_stds[feature])
            z_scores[feature] = z   

    # Extract top 3 anomalous features by Z-score
    top_features = sorted(z_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    reasons = [f"{feat} (z={round(val,2)})" for feat, val in top_features]
    return ", ".join(reasons)

feature_means = X.mean()
feature_stds  = X.std()

explanation = df.apply(
    lambda row: generate_explanation(row, feature_means, feature_stds),
    axis=1
)

results = pd.DataFrame({
    "file_id": file_ids,
    "ml_score": ml_scores,
    "rule_score": rule_scores,
    "final_score": final_scores,
    "explanation": explanation,
})

def assign_percentile_risk(df):
    p90 = np.percentile(df["final_score"], 90)
    p70 = np.percentile(df["final_score"], 70)
    p40 = np.percentile(df["final_score"], 40)

    def risk_label(score):
        if score >= p90: return "CRITICAL"
        if score >= p70: return "HIGH"
        if score >= p40: return "MEDIUM"
        return "LOW"

    df["risk_level"] = df["final_score"].apply(risk_label)
    return df

results = assign_percentile_risk(results)
results["final_score"] = results["final_score"].round(1)
results = results.sort_values(by="final_score", ascending=False)

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)

print("\n" + "="*80)
print(" SECURITY RISK SCORING REPORT (RESEARCH)")
print("="*80 + "\n")
print(results[["file_id", "final_score", "risk_level", "explanation"]].to_string(index=False))
print("\n" + "="*80)
