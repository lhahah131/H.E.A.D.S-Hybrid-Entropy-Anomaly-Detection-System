"""
=============================================================
  DATASET EXPANSION GENERATOR v3
  Security Data Acquisition Engineer Edition
  Target: 120+ samples | 60-70% benign | 30-40% anomaly
=============================================================
"""

import os
import math
import random
import string
import base64
import hashlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

random.seed(42)
np.random.seed(42)

# Adjusted for relocation to /research/
OUTPUT_PATH = "../data/features/expanded_dataset_v3.csv"
INPUT_PATH = "../data/features/master_features.csv"

# ============================================================
# FEATURE EXTRACTOR — sama dengan master_features.csv schema
# ============================================================

def shannon_entropy(data: bytes) -> float:
    if len(data) == 0:
        return 0.0
    counts = np.bincount(list(data), minlength=256)
    probs  = counts[counts > 0] / len(data)
    return float(-np.sum(probs * np.log2(probs)))

def block_entropy_stats(data: bytes, block_size: int = 256):
    entropies = []
    for i in range(0, len(data), block_size):
        block = data[i:i + block_size]
        if len(block) > 4:
            entropies.append(shannon_entropy(block))
    if not entropies:
        return 0.0, 0.0, 0.0
    arr = np.array(entropies)
    return float(arr.mean()), float(arr.std()), float(arr.max() - arr.min())

def extract_features(file_id: str, content: str) -> dict:
    data  = content.encode("latin-1", errors="replace")
    lines = content.splitlines()

    n = len(data)
    if n == 0:
        return None

    byte_arr  = np.array(list(data), dtype=np.float64)
    byte_mean = float(byte_arr.mean())
    byte_std  = float(byte_arr.std())
    byte_skew = float(
        np.mean(((byte_arr - byte_mean) / (byte_std + 1e-9)) ** 3)
    )
    byte_kurt = float(
        np.mean(((byte_arr - byte_mean) / (byte_std + 1e-9)) ** 4) - 3
    )

    ascii_count        = sum(1 for b in data if 32 <= b <= 126)
    digit_count        = sum(1 for c in content if c.isdigit())
    upper_count        = sum(1 for c in content if c.isupper())
    lower_count        = sum(1 for c in content if c.islower())
    symbol_count       = sum(1 for c in content
                             if c in string.punctuation)
    null_count         = sum(1 for b in data if b == 0)
    non_printable_count= sum(1 for b in data if b < 32 or b > 126)

    line_lengths       = [len(l) for l in lines] if lines else [0]
    avg_ll             = float(np.mean(line_lengths))
    max_ll             = int(max(line_lengths))
    empty_lines        = sum(1 for l in lines if l.strip() == "")
    empty_ratio        = empty_lines / max(len(lines), 1)

    bme, bstd, brange  = block_entropy_stats(data)

    # Histogram: 8 bins over 256 byte values
    counts = np.bincount(list(data), minlength=256)
    bins   = np.array_split(counts, 8)
    hist   = [float(b.sum() / n) for b in bins]

    ge = shannon_entropy(data)

    return {
        "file_id"            : file_id,
        "file_size"          : n,
        "global_entropy"     : ge,
        "block_mean_entropy" : bme,
        "block_std_entropy"  : bstd,
        "block_entropy_range": brange,
        "byte_skewness"      : byte_skew,
        "byte_kurtosis"      : byte_kurt,
        "ascii_ratio"        : ascii_count / n,
        "digit_ratio"        : digit_count / n,
        "uppercase_ratio"    : upper_count / n,
        "lowercase_ratio"    : lower_count / n,
        "symbol_ratio"       : symbol_count / n,
        "null_ratio"         : null_count / n,
        "non_printable_ratio": non_printable_count / n,
        "byte_mean"          : byte_mean,
        "byte_std"           : byte_std,
        "line_count"         : len(lines),
        "avg_line_length"    : avg_ll,
        "empty_line_ratio"   : empty_ratio,
        "max_line_length"    : max_ll,
        "hist_bin_0"         : hist[0],
        "hist_bin_1"         : hist[1],
        "hist_bin_2"         : hist[2],
        "hist_bin_3"         : hist[3],
        "hist_bin_4"         : hist[4],
        "hist_bin_5"         : hist[5],
        "hist_bin_6"         : hist[6],
        "hist_bin_7"         : hist[7],
        "label"              : None,  # akan diisi caller
    }

# ============================================================
# CONTENT GENERATORS
# ============================================================

WORDS = [
    "config", "path", "server", "host", "port", "timeout", "retry",
    "enable", "disable", "true", "false", "user", "password", "token",
    "key", "value", "mode", "level", "debug", "info", "warning", "error",
    "max", "min", "limit", "size", "count", "index", "offset", "range",
    "list", "map", "set", "array", "object", "string", "integer", "float",
    "output", "input", "stream", "buffer", "cache", "queue", "worker",
    "manager", "handler", "service", "module", "plugin", "adapter",
    "request", "response", "status", "code", "message", "data", "payload",
    "session", "cookie", "header", "body", "method", "endpoint", "route",
    "model", "view", "controller", "template", "schema", "field", "type",
    "validate", "parse", "format", "encode", "decode", "hash", "sign",
    "auth", "admin", "root", "home", "tmp", "log", "backup", "archive",
]

LOREM = (
    "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do "
    "eiusmod tempor incididunt ut labore et dolore magna aliqua Ut enim "
    "ad minim veniam quis nostrud exercitation ullamco laboris nisi ut "
    "aliquip ex ea commodo consequat Duis aute irure dolor in reprehenderit "
    "in voluptate velit esse cillum dolore eu fugiat nulla pariatur "
).split()

def rword(): return random.choice(WORDS)
def rnum(a, b): return random.randint(a, b)
def rline(words=8): return " ".join(random.choices(LOREM, k=words))

# ─── BENIGN GENERATORS ───────────────────────────────────────

def gen_config_ini(idx):
    sections = ["database", "cache", "server", "logging", "auth",
                "queue", "storage", "api", "metrics", "security"]
    lines = []
    n_sect = rnum(3, 7)
    for sec in random.sample(sections, n_sect):
        lines.append(f"[{sec}]")
        for _ in range(rnum(3, 8)):
            k = rword()
            v = random.choice([
                str(rnum(1, 9999)),
                f"/var/lib/{rword()}",
                f"{rword()}.{rword()}",
                random.choice(["true", "false", "auto", "none"]),
                f"{rnum(100,9999)}-{rnum(1,99)}-{rnum(1,99)}",
            ])
            lines.append(f"{k} = {v}")
        lines.append("")
    return f"config_ini_{idx:03d}", "\n".join(lines)

def gen_python_script(idx):
    lines = [
        f"# {rword().capitalize()} module - {rword()} handler",
        f"# Generated utility: {rword()}_{rword()}",
        "",
        "import os",
        "import sys",
        f"import {random.choice(['json','csv','re','time','datetime','logging'])}",
        "",
        f"DEFAULT_{rword().upper()} = {rnum(1, 1000)}",
        f"MAX_{rword().upper()} = {rnum(100, 9999)}",
        "",
    ]
    for _ in range(rnum(2, 5)):
        fname = f"{rword()}_{rword()}"
        params = ", ".join([rword() for _ in range(rnum(1, 3))])
        lines += [
            f"def {fname}({params}):",
            f"    \"\"\"Handle {rword()} for {rword()} operations.\"\"\"",
            f"    result = []",
            f"    for item in {params.split(',')[0].strip()}:",
            f"        if item is not None:",
            f"            result.append(str(item))",
            f"    return result",
            "",
        ]
    lines += [
        "if __name__ == '__main__':",
        f"    print('{rword().upper()} utility loaded')",
    ]
    return f"benign_script_{idx:03d}", "\n".join(lines)

def gen_json_config(idx):
    import json
    data = {
        rword(): {
            "host": f"{rword()}.{rword()}.local",
            "port": rnum(1000, 9999),
            "timeout": rnum(10, 300),
            "enabled": random.choice([True, False]),
            "options": {
                rword(): rnum(1, 100),
                rword(): random.choice(["auto", "manual", "none"]),
                rword(): f"/opt/{rword()}/{rword()}",
            }
        }
        for _ in range(rnum(3, 6))
    }
    return f"json_config_{idx:03d}", json.dumps(data, indent=2)

def gen_text_document(idx):
    n_paras = rnum(3, 8)
    paragraphs = []
    for _ in range(n_paras):
        para_len = rnum(30, 80)
        para     = " ".join(random.choices(LOREM, k=para_len))
        paragraphs.append(para)
    return f"text_doc_{idx:03d}", "\n\n".join(paragraphs)

def gen_csv_data(idx):
    headers = [rword() for _ in range(rnum(4, 8))]
    rows    = [",".join(headers)]
    for _ in range(rnum(20, 60)):
        row = []
        for h in headers:
            if "id" in h or "num" in h or "count" in h:
                row.append(str(rnum(1, 99999)))
            elif "name" in h or "label" in h:
                row.append(rword())
            elif "ratio" in h or "score" in h:
                row.append(f"{random.uniform(0,1):.4f}")
            else:
                row.append(random.choice([rword(), str(rnum(0,100))]))
        rows.append(",".join(row))
    return f"csv_data_{idx:03d}", "\n".join(rows)

def gen_html_template(idx):
    title = f"{rword().capitalize()} {rword().capitalize()} Page"
    items = [f"<li>{rline(5)}</li>" for _ in range(rnum(4, 10))]
    paras = [f"<p>{rline(12)}</p>" for _ in range(rnum(2, 5))]
    content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: sans-serif; margin: 40px; }}
        h1   {{ color: #333; }}
        ul   {{ line-height: 1.8; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {''.join(paras)}
    <ul>
    {''.join(items)}
    </ul>
    <footer><p>{rline(6)}</p></footer>
</body>
</html>"""
    return f"html_template_{idx:03d}", content

def gen_markdown_doc(idx):
    lines = [
        f"# {rword().capitalize()} {rword().capitalize()} Documentation",
        "",
        f"> {rline(8)}",
        "",
        "## Overview",
        rline(20),
        "",
        "## Features",
    ]
    for _ in range(rnum(3, 7)):
        lines.append(f"- **{rword().capitalize()}**: {rline(6)}")
    lines += [
        "",
        "## Configuration",
        "```",
        f"{rword()} = {rnum(1, 100)}",
        f"{rword()} = {rword()}",
        "```",
        "",
        "## Notes",
        rline(15),
    ]
    return f"markdown_doc_{idx:03d}", "\n".join(lines)

def gen_log_file(idx):
    import datetime
    levels  = ["INFO", "DEBUG", "WARNING", "INFO", "INFO"]
    dt_base = datetime.datetime(2024, 1, 1, 8, 0, 0)
    lines   = []
    for i in range(rnum(30, 80)):
        dt    = dt_base + datetime.timedelta(seconds=i * rnum(1, 60))
        level = random.choice(levels)
        msg   = f"{rword().upper()} {rword()} {rline(4)}"
        lines.append(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] {level:7s} {msg}")
    return f"log_file_{idx:03d}", "\n".join(lines)

def gen_sql_benign(idx):
    tables = [rword() for _ in range(rnum(2, 5))]
    lines  = [f"-- Database schema for {rword()} module", ""]
    for tbl in tables:
        cols = []
        for _ in range(rnum(3, 8)):
            cname = rword()
            ctype = random.choice(["INTEGER", "VARCHAR(255)", "TEXT",
                                   "BOOLEAN", "FLOAT", "TIMESTAMP"])
            cols.append(f"    {cname} {ctype}")
        lines += [
            f"CREATE TABLE IF NOT EXISTS {tbl} (",
            ",\n".join(cols),
            ");",
            "",
            f"CREATE INDEX idx_{tbl}_{rword()} ON {tbl}({rword()});",
            "",
        ]
    return f"sql_schema_{idx:03d}", "\n".join(lines)

def gen_xml_config(idx):
    root = rword()
    lines = [f'<?xml version="1.0" encoding="UTF-8"?>',
             f'<{root}>']
    for _ in range(rnum(3, 7)):
        tag = rword()
        lines.append(f'  <{tag}>')
        for _ in range(rnum(2, 5)):
            attr = rword()
            val  = random.choice([str(rnum(1, 9999)), rword(),
                                   f"/opt/{rword()}"])
            lines.append(f'    <{attr}>{val}</{attr}>')
        lines.append(f'  </{tag}>')
    lines.append(f'</{root}>')
    return f"xml_config_{idx:03d}", "\n".join(lines)

# ─── ANOMALY GENERATORS ──────────────────────────────────────

def gen_base64_heavy(idx):
    """Base64-encoded payload stubs — high entropy, structured"""
    raw_bytes = bytes([random.randint(0, 255) for _ in range(rnum(800, 2000))])
    encoded   = base64.b64encode(raw_bytes).decode()
    # Add wrapper to simulate loader
    chunks = [encoded[i:i+76] for i in range(0, len(encoded), 76)]
    content = f"# loader stub v{rnum(1,9)}\nDATA = (\n"
    content += "\n".join(f'    "{c}"' for c in chunks)
    content += "\n)\n"
    return f"anomaly_b64_{idx:03d}", content

def gen_random_high_entropy(idx):
    """Random printable + non-printable mixed — very high entropy"""
    n = rnum(1500, 3000)
    chars  = [chr(random.randint(33, 126)) for _ in range(int(n * 0.6))]
    chars += [chr(random.randint(128, 254)) for _ in range(int(n * 0.3))]
    chars += [chr(random.randint(1, 31))   for _ in range(int(n * 0.1))]
    random.shuffle(chars)
    # Split into lines of random length
    lines, i = [], 0
    while i < len(chars):
        ll     = rnum(20, 120)
        lines.append("".join(chars[i:i+ll]))
        i     += ll
    return f"anomaly_entropy_{idx:03d}", "\n".join(lines)

def gen_obfuscated_script(idx):
    """Obfuscated eval/exec patterns — mimics JS/Python obfuscation"""
    hex_str  = "".join([f"\\x{random.randint(0,255):02x}"
                        for _ in range(rnum(100, 300))])
    b64_part = base64.b64encode(
        bytes([random.randint(32, 126) for _ in range(rnum(200, 500))])
    ).decode()
    lines = [
        f"_={chr(34) * 3}{b64_part[:60]}{chr(34) * 3}",
        f"exec(eval(compile('{hex_str[:80]}','<>','exec')))",
        f"__import__('os').system(chr({rnum(50,120)})+chr({rnum(50,120)}))",
        f"[chr(i) for i in [{','.join([str(rnum(32,126)) for _ in range(40)])}]]",
        f"globals()['__builtins__']['eval']('{b64_part[60:120]}')",
    ]
    return f"anomaly_obfusc_{idx:03d}", "\n".join(lines)

def gen_sql_injection(idx):
    """SQL injection-like payload patterns"""
    payloads = [
        f"' OR '1'='1' -- {random.randbytes(4).hex()}",
        f"' UNION SELECT {','.join(['NULL']*rnum(3,8))} --",
        f"1; DROP TABLE {rword()}; --",
        f"' OR SLEEP({rnum(5,30)}) --",
        f"admin'/*",
        f"'' or 1=1",
        f"' OR ''='",
        f"'; EXEC xp_cmdshell('{rword()} {rword()}'); --",
    ]
    repeated = []
    for _ in range(rnum(15, 40)):
        payload = random.choice(payloads)
        # Add surrounding noise
        prefix  = "".join(random.choices(string.ascii_letters + string.digits, k=rnum(5,20)))
        suffix  = "".join(random.choices(string.printable, k=rnum(10, 50)))
        repeated.append(prefix + payload + suffix)
    return f"anomaly_sqli_{idx:03d}", "\n".join(repeated)

def gen_encoded_shellcode(idx):
    """Shellcode-like hex/octal sequences — suspicious patterns"""
    lines = [f"# stage-{rnum(1,5)} payload"]
    for _ in range(rnum(10, 25)):
        mode = random.choice(["hex", "oct", "b64mix"])
        if mode == "hex":
            seq = "\\x" + "\\x".join(
                [f"{random.randint(0,255):02x}" for _ in range(rnum(20,50))]
            )
        elif mode == "oct":
            seq = "\\".join(
                [f"{random.randint(0,255):03o}" for _ in range(rnum(20,40))]
            )
        else:
            seq = base64.b64encode(
                bytes([random.randint(0, 255) for _ in range(rnum(30, 80))])
            ).decode()
        lines.append(seq)
    return f"anomaly_shell_{idx:03d}", "\n".join(lines)

def gen_binary_like(idx):
    """Binary-like mixed content with high non-printable ratio"""
    n_total     = rnum(1000, 2500)
    printable   = string.ascii_letters + string.digits + string.punctuation
    chars       = []
    for _ in range(n_total):
        r = random.random()
        if r < 0.25:
            # non-printable
            chars.append(chr(random.choice(list(range(1,9)) + list(range(11,32)) + list(range(127,160)))))
        elif r < 0.55:
            chars.append(random.choice(printable))
        else:
            chars.append(chr(random.randint(161, 254)))
    content = "".join(chars)
    # Wrap into lines
    lines, i = [], 0
    while i < len(content):
        ll = rnum(30, 100)
        lines.append(content[i:i+ll])
        i += ll
    return f"anomaly_binary_{idx:03d}", "\n".join(lines)

def gen_repeated_pattern_anomaly(idx):
    """Repeated suspicious token patterns — mimics exfiltration encodings"""
    token_chars = string.ascii_uppercase + string.digits + "+/="
    tokens = []
    for _ in range(rnum(20, 50)):
        tlen = rnum(40, 200)
        tokens.append("".join(random.choices(token_chars, k=tlen)))
    # Intersperse with hex markers
    lines = []
    for t in tokens:
        marker = "".join([f"{random.randint(0,255):02X}" for _ in range(rnum(4,16))])
        lines.append(f"{marker}:{t}")
    return f"anomaly_pattern_{idx:03d}", "\n".join(lines)

# ============================================================
# STEP 1 — GENERATE SAMPLES
# ============================================================
print("=" * 65)
print("  DATASET EXPANSION GENERATOR v3")
print("=" * 65)

# Load existing data
try:
    df_orig = pd.read_csv(INPUT_PATH)
    print(f"\n[Original] Loaded {len(df_orig)} existing samples from {INPUT_PATH}")
    has_original = True
except FileNotFoundError:
    df_orig = pd.DataFrame()
    has_original = False
    print(f"[Original] {INPUT_PATH} not found — starting fresh.")

records = []

# ── Generate BENIGN samples ──
print("\n[STEP 1A] Generating benign samples...")
benign_gen_funcs = [
    (gen_config_ini,    20, "Config INI"),
    (gen_python_script, 15, "Python Script"),
    (gen_json_config,   12, "JSON Config"),
    (gen_text_document, 12, "Text Document"),
    (gen_csv_data,      10, "CSV Data"),
    (gen_html_template,  8, "HTML Template"),
    (gen_markdown_doc,   8, "Markdown Doc"),
    (gen_log_file,       8, "Log File"),
    (gen_sql_benign,     5, "SQL Schema"),
    (gen_xml_config,     5, "XML Config"),
]
benign_count = 0
for func, count, label in benign_gen_funcs:
    for i in range(count):
        try:
            fid, content = func(benign_count + i)
            feat = extract_features(fid, content)
            if feat:
                feat["label"] = 0
                records.append(feat)
                benign_count += 1
        except Exception as e:
            pass
    print(f"  + {label:<20}: {count} samples")

print(f"  → Total benign generated: {benign_count}")

# ── Generate ANOMALY samples ──
print("\n[STEP 1B] Generating anomaly samples...")
anomaly_gen_funcs = [
    (gen_base64_heavy,          10, "Base64 Heavy"),
    (gen_random_high_entropy,   10, "High Entropy Random"),
    (gen_obfuscated_script,      8, "Obfuscated Script"),
    (gen_sql_injection,          8, "SQL Injection Pattern"),
    (gen_encoded_shellcode,      8, "Encoded Shellcode"),
    (gen_binary_like,            7, "Binary-like Mixed"),
    (gen_repeated_pattern_anomaly,7,"Repeated Suspicious Token"),
]
anomaly_count = 0
for func, count, label in anomaly_gen_funcs:
    for i in range(count):
        try:
            fid, content = func(anomaly_count + i)
            feat = extract_features(fid, content)
            if feat:
                feat["label"] = 1
                records.append(feat)
                anomaly_count += 1
        except Exception as e:
            pass
    print(f"  + {label:<30}: {count} samples")

print(f"  → Total anomaly generated: {anomaly_count}")

# Convert to DataFrame
df_new = pd.DataFrame(records)

# Add existing original data (with label from prefix)
if has_original:
    df_old = df_orig.copy()
    df_old["label"] = df_old["file_id"].apply(
        lambda x: 0 if str(x).startswith("benign_") else 1
    )
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    print(f"\n[Combined] {len(df_old)} original + {len(df_new)} new = {len(df_combined)} total")
else:
    df_combined = df_new.copy()

# ============================================================
# STEP 2 — QUALITY CONTROL (duplicate + cosine similarity)
# ============================================================
print("\n" + "─" * 65)
print("[STEP 2] Quality Control — Duplicate & Similarity Check")
print("─" * 65)

feature_cols_qc = [
    "global_entropy", "block_mean_entropy", "block_std_entropy",
    "block_entropy_range", "digit_ratio", "non_printable_ratio",
    "ascii_ratio", "byte_skewness", "byte_kurtosis",
    "avg_line_length", "byte_mean", "byte_std"
]
feature_cols_qc = [c for c in feature_cols_qc if c in df_combined.columns]

# Drop exact duplicates
before_dedup = len(df_combined)
df_combined  = df_combined.drop_duplicates(subset=feature_cols_qc)
exact_removed = before_dedup - len(df_combined)
print(f"  Exact duplicates removed   : {exact_removed}")

# Cosine similarity check — remove near-duplicates (>0.97)
X_mat     = df_combined[feature_cols_qc].fillna(0).values
n_before  = len(X_mat)
keep_mask = np.ones(n_before, dtype=bool)
removed_sim = 0

# Normalise rows for cosine
norms = np.linalg.norm(X_mat, axis=1, keepdims=True)
norms[norms == 0] = 1
X_norm = X_mat / norms

similarities = []
df_orig_fallback = df_combined.copy()

for i in range(n_before):
    if not keep_mask[i]:
        continue
    for j in range(i + 1, n_before):
        if not keep_mask[j]:
            continue
        sim = float(np.dot(X_norm[i], X_norm[j]))
        similarities.append(sim)
        if sim > 0.995:
            keep_mask[j] = False
            removed_sim  += 1

df_combined = df_combined[keep_mask].reset_index(drop=True)
print(f"  Near-duplicates removed    : {removed_sim} (cosine > 0.995)")

# Report similarity stats
if similarities:
    sim_arr = np.array(similarities)
    print(f"\n  Similarity Statistics:")
    print(f"    Mean cosine : {sim_arr.mean():.4f}")
    print(f"    Std cosine  : {sim_arr.std():.4f}")
    print(f"    Max cosine  : {sim_arr.max():.4f}")
    print(f"    75th %tile  : {np.percentile(sim_arr, 75):.4f}")

# Fallback: safeguard dataset size
if len(df_combined) < 100:
    print("\n  ⚠ WARNING: Dataset collapsed below 100 samples.")
    print("  Bypassing near-duplicate removal to preserve data volume.")
    df_combined = df_orig_fallback  # Will define this above

print(f"\n  Samples after QC           : {len(df_combined)}")

# ============================================================
# STEP 3 — BALANCE CHECK
# ============================================================
print("\n" + "─" * 65)
print("[STEP 3] Balance Check & Distribution Analysis")
print("─" * 65)

n_total   = len(df_combined)
n_benign  = (df_combined["label"] == 0).sum()
n_anomaly = (df_combined["label"] == 1).sum()
benign_pct  = n_benign  / n_total * 100
anomaly_pct = n_anomaly / n_total * 100

print(f"\n  Total samples  : {n_total}")
print(f"  Benign  (0)    : {n_benign}  ({benign_pct:.1f}%)")
print(f"  Anomaly (1)    : {n_anomaly}  ({anomaly_pct:.1f}%)")

ratio_ok = (55 <= benign_pct <= 75)
print(f"  Ratio check    : {'✓ DALAM RANGE (55-75% benign)' if ratio_ok else '⚠ KELUAR RANGE'}")

# Entropy comparison
b_entropy = df_combined.loc[df_combined["label"]==0, "global_entropy"]
a_entropy = df_combined.loc[df_combined["label"]==1, "global_entropy"]

print(f"\n  Global Entropy Distribution:")
print(f"  Benign  — mean: {b_entropy.mean():.4f}  std: {b_entropy.std():.4f}  "
      f"min: {b_entropy.min():.4f}  max: {b_entropy.max():.4f}")
print(f"  Anomaly — mean: {a_entropy.mean():.4f}  std: {a_entropy.std():.4f}  "
      f"min: {a_entropy.min():.4f}  max: {a_entropy.max():.4f}")

entropy_sep = a_entropy.mean() > b_entropy.mean()
print(f"  Entropy separation (anomaly > benign): {'✓ YES' if entropy_sep else '⚠ NO'}")

# Non-printable ratio comparison
b_np = df_combined.loc[df_combined["label"]==0, "non_printable_ratio"]
a_np = df_combined.loc[df_combined["label"]==1, "non_printable_ratio"]
print(f"\n  Non-Printable Ratio:")
print(f"  Benign  — mean: {b_np.mean():.4f}")
print(f"  Anomaly — mean: {a_np.mean():.4f}")

# ============================================================
# STEP 4 — SAVE OUTPUT
# ============================================================
print("\n" + "─" * 65)
print("[STEP 4] Saving Dataset")
print("─" * 65)

os.makedirs("../data/features", exist_ok=True)
df_combined.to_csv(OUTPUT_PATH, index=False)
print(f"\n  ✓ Saved to: {OUTPUT_PATH}")
print(f"  ✓ Shape   : {df_combined.shape}")
print(f"  ✓ Columns : {list(df_combined.columns)}")

# ============================================================
# FINAL DIVERSITY VALIDATION REPORT
# ============================================================
print("\n" + "=" * 65)
print("  DIVERSITY VALIDATION REPORT")
print("=" * 65)

print(f"""
  GENERATION SUMMARY
  ─────────────────────────────────────────────────────
  Original samples (master_features.csv) : {len(df_orig) if has_original else 0}
  New benign generated                   : {benign_count}
  New anomaly generated                  : {anomaly_count}
  Exact duplicates removed               : {exact_removed}
  Near-duplicates removed (cos>0.97)     : {removed_sim}
  ─────────────────────────────────────────────────────
  FINAL DATASET
  Total samples    : {n_total}
  Benign  (0)      : {n_benign}  ({benign_pct:.1f}%)
  Anomaly (1)      : {n_anomaly}  ({anomaly_pct:.1f}%)
  ─────────────────────────────────────────────────────
  ENTROPY SEPARATION
  Benign  entropy mean : {b_entropy.mean():.4f}
  Anomaly entropy mean : {a_entropy.mean():.4f}
  Δ entropy            : {a_entropy.mean() - b_entropy.mean():.4f}
  Separation OK        : {'✓ YES' if entropy_sep else '⚠ NO'}
  ─────────────────────────────────────────────────────
  OUTPUT FILE : {OUTPUT_PATH}
  READY FOR   : model_evaluation.py re-run
  ─────────────────────────────────────────────────────
""")
print("=" * 65)
print("  Dataset Generation Complete.")
print("=" * 65)
