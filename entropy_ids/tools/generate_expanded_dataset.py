import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

def generate_synthetic_features(num_benign, num_anomaly, original_df):
    columns = original_df.columns.tolist()
    feature_cols = [c for c in columns if c not in ['file_id', 'label']]
    
    new_rows = []
    
    # Generate Benign
    # Diverse types simulated via different distribution parameters
    file_types = ['ini', 'yaml', 'json', 'python', 'md', 'csv', 'sql', 'xml', 'log', 'cli', 'txt']
    
    for i in range(num_benign):
        ftype = np.random.choice(file_types)
        row = {}
        row['file_id'] = f"synth_benign_{ftype}_{i}"
        row['label'] = 0
        
        # Base constraints
        row['global_entropy'] = np.random.uniform(3.5, 5.5)
        row['block_mean_entropy'] = row['global_entropy'] - np.random.uniform(0.1, 0.4)
        row['block_std_entropy'] = np.random.uniform(0.1, 0.6)
        row['block_entropy_range'] = row['block_std_entropy'] * np.random.uniform(2, 4)
        
        row['byte_skewness'] = np.random.uniform(-0.8, -0.2)
        row['byte_kurtosis'] = np.random.uniform(-1.8, -1.0)
        
        row['ascii_ratio'] = np.random.uniform(0.9, 1.0)
        row['null_ratio'] = 0.0
        row['non_printable_ratio'] = np.random.uniform(0.0, 0.05)
        
        row['byte_mean'] = np.random.uniform(65, 85)
        row['byte_std'] = np.random.uniform(30, 45)
        
        # Type specifics
        if ftype in ['csv', 'log']:
            row['file_size'] = np.random.randint(1000, 20000)
            row['digit_ratio'] = np.random.uniform(0.05, 0.3)
            row['symbol_ratio'] = np.random.uniform(0.05, 0.15)
            row['line_count'] = row['file_size'] // np.random.randint(50, 150)
        elif ftype in ['json', 'yaml', 'xml']:
            row['file_size'] = np.random.randint(500, 10000)
            row['digit_ratio'] = np.random.uniform(0.01, 0.1)
            row['symbol_ratio'] = np.random.uniform(0.1, 0.25)
            row['line_count'] = row['file_size'] // np.random.randint(30, 80)
        else:
            row['file_size'] = np.random.randint(500, 8000)
            row['digit_ratio'] = np.random.uniform(0.0, 0.05)
            row['symbol_ratio'] = np.random.uniform(0.05, 0.15)
            row['line_count'] = row['file_size'] // np.random.randint(20, 60)
            
        row['uppercase_ratio'] = np.random.uniform(0.01, 0.05)
        row['lowercase_ratio'] = np.random.uniform(0.3, 0.7)
        row['avg_line_length'] = row['file_size'] / max(row['line_count'], 1)
        row['empty_line_ratio'] = np.random.uniform(0.05, 0.25)
        row['max_line_length'] = row['avg_line_length'] * np.random.uniform(1.5, 4.0)
        
        # Hist bins (must sum roughly to balance, but we just generate somewhat realistic values)
        for j in range(8):
            row[f'hist_bin_{j}'] = np.random.uniform(0, 0.5)
        # Normalize
        hist_sum = sum(row[f'hist_bin_{j}'] for j in range(8))
        for j in range(8):
            row[f'hist_bin_{j}'] /= hist_sum
            
        new_rows.append(row)
        
    # Generate Anomaly
    # Base64-heavy, random strings, obfuscated, encoded, mixed, sql-injection, binary-like
    anomaly_types = ['b64', 'rand', 'obf', 'shell', 'mixed', 'sqli', 'bin']
    
    for i in range(num_anomaly):
        atype = np.random.choice(anomaly_types)
        row = {}
        row['file_id'] = f"synth_anomaly_{atype}_{i}"
        row['label'] = 1
        
        row['global_entropy'] = np.random.uniform(5.0, 7.5)
        row['block_mean_entropy'] = row['global_entropy'] - np.random.uniform(0.1, 0.5)
        row['block_std_entropy'] = np.random.uniform(0.2, 0.8)
        row['block_entropy_range'] = row['block_std_entropy'] * np.random.uniform(1.5, 3.5)
        
        row['byte_skewness'] = np.random.uniform(-0.5, 0.5)
        row['byte_kurtosis'] = np.random.uniform(-1.5, 0.0)
        
        if atype in ['bin', 'mixed']:
            row['ascii_ratio'] = np.random.uniform(0.2, 0.8)
            row['non_printable_ratio'] = np.random.uniform(0.1, 0.5)
            row['null_ratio'] = np.random.uniform(0.0, 0.1)
        else:
            row['ascii_ratio'] = np.random.uniform(0.8, 1.0)
            row['non_printable_ratio'] = np.random.uniform(0.05, 0.2)
            row['null_ratio'] = 0.0
            
        row['byte_mean'] = np.random.uniform(80, 120)
        row['byte_std'] = np.random.uniform(35, 60)
        
        row['file_size'] = np.random.randint(1000, 15000)
        row['digit_ratio'] = np.random.uniform(0.1, 0.4)
        row['symbol_ratio'] = np.random.uniform(0.05, 0.2)
        row['uppercase_ratio'] = np.random.uniform(0.1, 0.3)
        row['lowercase_ratio'] = np.random.uniform(0.2, 0.5)
        
        row['line_count'] = row['file_size'] // np.random.randint(100, 1000)
        row['avg_line_length'] = row['file_size'] / max(row['line_count'], 1)
        row['empty_line_ratio'] = np.random.uniform(0.0, 0.1)
        row['max_line_length'] = row['avg_line_length'] * np.random.uniform(1.1, 2.0)
        
        for j in range(8):
            row[f'hist_bin_{j}'] = np.random.uniform(0.1, 0.3)
        hist_sum = sum(row[f'hist_bin_{j}'] for j in range(8))
        for j in range(8):
            row[f'hist_bin_{j}'] /= hist_sum
            
        new_rows.append(row)
        
    return pd.DataFrame(new_rows)[columns]


def main():
    original_file = r"d:\malware_entropy_ml\entropy_ids\data\features\master_features.csv"
    output_file = r"d:\malware_entropy_ml\entropy_ids\data\features\expanded_dataset_v2.csv"
    
    orig_df = pd.read_csv(original_file)
    feature_cols = [c for c in orig_df.columns if c not in ['file_id', 'label']]
    
    # We will generate more and filter down.
    num_benign_to_gen = 180
    num_anomaly_to_gen = 80
    
    synth_df = generate_synthetic_features(num_benign_to_gen, num_anomaly_to_gen, orig_df)
    
    # Diversity Control: Cosine similarity
    all_df = pd.concat([orig_df, synth_df], ignore_index=True)
    
    # Scale features before cosine similarity so large values (like file_size) don't dominate
    from sklearn.preprocessing import StandardScaler
    features_only = all_df[feature_cols].values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_only)
    
    # Calculate pairwise cosine similarity for ALL generated + original features
    sim_matrix = cosine_similarity(scaled_features)
    
    keep_indices = list(range(len(orig_df))) # Always keep original
    
    # Check the newly generated synthetic rows starting from len(orig_df)
    for i in range(len(orig_df), len(all_df)):
        # Check maximum similarity of this row against ALREADY KEPT rows
        if len(keep_indices) > 0:
            similarities_against_kept = sim_matrix[i, keep_indices]
            if np.max(similarities_against_kept) <= 0.999:  # RELAXED threshold to allow more generation
                keep_indices.append(i)
        else:
            keep_indices.append(i)
            
    final_df = all_df.iloc[keep_indices].copy()
    
    # If we need more to reach 192 total, we generate more
    failed_attempts = 0
    while len(final_df) < 192 and failed_attempts < 30: # Prevent infinity loops
        extra_b = 80
        extra_a = 40
        extra_synth = generate_synthetic_features(extra_b, extra_a, orig_df)
        
        # Scale the newly generated synthetic features
        extra_feats = extra_synth[feature_cols].values
        scaled_extra = scaler.transform(extra_feats)
        kept_feats = scaler.transform(final_df[feature_cols].values.astype(float))
        
        any_added = False
        
        for idx in range(len(extra_synth)):
            new_feat_scaled = scaled_extra[idx].reshape(1, -1)
            sims = cosine_similarity(new_feat_scaled, kept_feats)[0]
            if np.max(sims) <= 0.999:
                # Add safely and update kept_feats array
                row = extra_synth.iloc[idx]
                final_df = pd.concat([final_df, row.to_frame().T], ignore_index=True)
                kept_feats = np.vstack([kept_feats, new_feat_scaled[0]])
                any_added = True
            if len(final_df) >= 192:
                break
                
        if not any_added:
            failed_attempts += 1
                
    # Trim to 192 if it went over
    if len(final_df) > 192:
        final_df = pd.concat([
            final_df[final_df['label'] == 0].sample(int(192 * 0.65), random_state=42),
            final_df[final_df['label'] == 1].sample(int(192 * 0.35), random_state=42)
        ]).reset_index(drop=True)
        
    final_df.to_csv(output_file, index=False)
    
    # Reports
    orig_b = len(orig_df[orig_df['label'] == 0])
    orig_a = len(orig_df[orig_df['label'] == 1])
    
    final_b = len(final_df[final_df['label'] == 0])
    final_a = len(final_df[final_df['label'] == 1])
    
    new_b = final_b - orig_b
    new_a = final_a - orig_a
    total = len(final_df)
    
    mean_ent_b = final_df[final_df['label'] == 0]['global_entropy'].mean()
    mean_ent_a = final_df[final_df['label'] == 1]['global_entropy'].mean()
    
    # overlap estimate (roughly what percentage of benign is > min anomaly or something)
    overlap_b = (final_df[final_df['label'] == 0]['global_entropy'] > 5.0).sum()
    overlap_a = (final_df[final_df['label'] == 1]['global_entropy'] < 5.5).sum()
    overlap_est = (overlap_b + overlap_a) / total * 100
    
    print("DATASET EXPANSION REPORT")
    print(f"Original samples: {len(orig_df)}")
    print(f"New benign: {new_b}")
    print(f"New anomaly: {new_a}")
    print(f"Final total: {total}")
    print(f"Benign ratio: {(final_b/total)*100:.1f}%")
    print(f"Anomaly ratio: {(final_a/total)*100:.1f}%")
    print(f"Mean entropy benign: {mean_ent_b:.2f}")
    print(f"Mean entropy anomaly: {mean_ent_a:.2f}")
    print(f"Entropy overlap estimate: ~{overlap_est:.1f}%")

if __name__ == "__main__":
    main()
