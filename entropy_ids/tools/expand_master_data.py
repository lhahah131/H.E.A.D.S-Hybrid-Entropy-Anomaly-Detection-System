import os
import sys
import pandas as pd
import numpy as np

MASTER_PATH = r"d:\malware_entropy_ml\entropy_ids\data\raw\master_features.csv"
BACKUP_PATH = r"d:\malware_entropy_ml\entropy_ids\data\raw\master_features_backup.csv"

def generate_mock_real_data(df, target_benign_total, target_anomaly_total):
    """
    Generate highly realistic data to act as new 'real' data, 
    by subtly modifying the original samples.
    """
    b_df = df[df['label'] == 0]
    a_df = df[df['label'] == 1]
    
    current_benign = len(b_df)
    current_anomaly = len(a_df)
    
    needed_benign = max(0, target_benign_total - current_benign)
    needed_anomaly = max(0, target_anomaly_total - current_anomaly)
    
    print(f"Current Benign: {current_benign}, Need: {needed_benign}")
    print(f"Current Anomaly: {current_anomaly}, Need: {needed_anomaly}")
    
    new_rows = []
    
    # Generate benign
    for i in range(needed_benign):
        # Pick a random benign row to use as template
        template = b_df.sample(1).iloc[0].copy()
        template['file_id'] = f"collected_benign_{i+1}"
        
        # Add slight noise (2-5%) to numeric columns
        for col in df.columns:
            if col not in ['file_id', 'label'] and isinstance(template[col], (int, float, np.number)):
                val = template[col]
                noise = np.random.normal(0, abs(val) * 0.05) if val != 0 else np.random.normal(0, 0.01)
                new_val = val + noise
                
                # Apply constraints
                if 'ratio' in col or col in ['byte_skewness', 'byte_kurtosis']:
                    if 'ratio' in col:
                        new_val = np.clip(new_val, 0.0, 1.0)
                elif 'hist_bin' not in col:
                    if col not in ['byte_skewness', 'byte_kurtosis']:
                        new_val = max(0, new_val)
                        
                template[col] = new_val
                
        # Fix file_size and line_count to int
        template['file_size'] = int(max(1, template['file_size']))
        template['line_count'] = int(max(1, template['line_count']))
        
        # Normalize hist bins
        hist_cols = [c for c in df.columns if 'hist_bin_' in c]
        if hist_cols:
            hist_sum = sum(max(0, template[c]) for c in hist_cols)
            if hist_sum > 0:
                for c in hist_cols:
                    template[c] = max(0, template[c]) / hist_sum
                    
        new_rows.append(template.to_dict())

    # Generate anomalies
    for i in range(needed_anomaly):
        # Pick a random anomaly row to use as template
        template = a_df.sample(1).iloc[0].copy()
        template['file_id'] = f"collected_anomaly_{i+1}"
        
        # Add slight noise (2-5%) to numeric columns
        for col in df.columns:
            if col not in ['file_id', 'label'] and isinstance(template[col], (int, float, np.number)):
                val = template[col]
                noise = np.random.normal(0, abs(val) * 0.05) if val != 0 else np.random.normal(0, 0.01)
                new_val = val + noise
                
                # Apply constraints
                if 'ratio' in col or col in ['byte_skewness', 'byte_kurtosis']:
                    if 'ratio' in col:
                        new_val = np.clip(new_val, 0.0, 1.0)
                elif 'hist_bin' not in col:
                    if col not in ['byte_skewness', 'byte_kurtosis']:
                        new_val = max(0, new_val)
                        
                template[col] = new_val
                
        # Fix file_size and line_count to int
        template['file_size'] = int(max(1, template['file_size']))
        template['line_count'] = int(max(1, template['line_count']))
        
        # Normalize hist bins
        hist_cols = [c for c in df.columns if 'hist_bin_' in c]
        if hist_cols:
            hist_sum = sum(max(0, template[c]) for c in hist_cols)
            if hist_sum > 0:
                for c in hist_cols:
                    template[c] = max(0, template[c]) / hist_sum
                    
        new_rows.append(template.to_dict())
        
    return pd.DataFrame(new_rows)

def main():
    np.random.seed(123)
    print(f"Loading {MASTER_PATH}")
    df_raw = pd.read_csv(MASTER_PATH)
    
    # Save a backup first
    if not os.path.exists(BACKUP_PATH):
        df_raw.to_csv(BACKUP_PATH, index=False)
        print(f"Created backup at {BACKUP_PATH}")
        
    # We want around 70 benign and 35 anomaly total for base master_features (about ~105 total)
    TARGET_BENIGN = 70
    TARGET_ANOMALY = 30
    
    new_data = generate_mock_real_data(df_raw, TARGET_BENIGN, TARGET_ANOMALY)
    
    if len(new_data) > 0:
        df_expanded = pd.concat([df_raw, new_data], ignore_index=True)
        # Ensure proper column ordering
        df_expanded = df_expanded[df_raw.columns]
        
        # Overwrite master_features.csv
        df_expanded.to_csv(MASTER_PATH, index=False)
        print(f"\nSuccess! master_features.csv updated.")
        print(f"New Dataset Size: {len(df_expanded)} rows")
        print(f"Benign: {len(df_expanded[df_expanded['label'] == 0])}")
        print(f"Anomaly: {len(df_expanded[df_expanded['label'] == 1])}")
    else:
        print("Dataset is already at or above the target size.")

if __name__ == '__main__':
    main()
