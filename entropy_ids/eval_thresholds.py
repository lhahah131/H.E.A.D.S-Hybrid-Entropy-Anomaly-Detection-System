import sys
import os
import pandas as pd
import numpy as np

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from core.feature_engine import engineer_features, extract_features_and_labels
from core.model_engine import IsolationForestEngine
from core.classifier_engine import apply_benign_confirmation
from core.evaluation_engine import evaluate_metrics

def main():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'saved_models', 'iso_v2.pkl')
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'master_features.csv')

    print(f"Loading data from {data_path}")
    df_raw = pd.read_csv(data_path)
    df_feats = engineer_features(df_raw)
    X, y_true, df_processed = extract_features_and_labels(df_feats)

    print(f"Loading model from {model_path}")
    model = IsolationForestEngine()
    model.load_model(model_path)

    raw_scores = model.get_scores(X)

    percentiles = [50, 52, 55, 58, 60]
    results = []

    print("\nRunning percentile experiments...")
    for p in percentiles:
        threshold = np.percentile(raw_scores, p)
        
        # Classification
        preds = (raw_scores < threshold).astype(int)
        
        # Benign Confirmation Layer
        # Note: the function apply_benign_confirmation modifies the array in place and returns it, but uses a loop.
        final_preds = apply_benign_confirmation(preds, df_processed)
        
        # Calculate metrics
        metrics = evaluate_metrics(y_true, final_preds, -raw_scores)
        
        results.append({
            'Percentile': p,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
            'FP': metrics['fp'],
            'FN': metrics['fn']
        })

    print("\nPercentile | Precision | Recall | F1 | FP | FN")
    print("-" * 55)
    
    best_p = None
    best_fp = float('inf')
    best_f1 = -1
    
    for r in results:
        print(f"{r['Percentile']:10d} | {r['Precision']:9.4f} | {r['Recall']:6.4f} | {r['F1']:4.4f} | {r['FP']:2d} | {r['FN']:2d}")
        
        # Selection logic:
        # - Recall >= 0.75
        # - FP as small as possible
        # - F1 maximal
        if r['Recall'] >= 0.75:
            if r['FP'] < best_fp or (r['FP'] == best_fp and r['F1'] > best_f1):
                best_fp = r['FP']
                best_f1 = r['F1']
                best_p = r['Percentile']

    if best_p is not None:
        print(f"\nRecommended production percentile = {best_p}")
    else:
        print("\nNo percentile met the minimum requirements (Recall >= 0.75).")

if __name__ == '__main__':
    main()
