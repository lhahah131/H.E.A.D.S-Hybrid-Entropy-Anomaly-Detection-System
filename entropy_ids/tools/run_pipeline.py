import subprocess
import sys
import os

# ── Lokasi project ──────────────────────────────────────────────────────────
ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_PY   = os.path.join(ROOT_DIR, "app", "main.py")
PYTHON    = sys.executable  # Python yang sedang aktif di environment ini

TRAIN_DATA    = "master_features.csv"
REAL_DATA     = "master_features.csv"
MODE          = "adaptive"
DIVIDER       = "=" * 60


def run_step(label: str, cmd: list) -> int:
    """Jalankan satu perintah dan tampilkan output-nya secara real-time."""
    print()
    print(DIVIDER)
    print(f"  {label}")
    print(DIVIDER)
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    print()
    if result.returncode != 0:
        print(f"  [ERROR] Step gagal dengan exit code {result.returncode}")
    else:
        print(f"  [OK] Step selesai.")
    return result.returncode


def main():
    print()
    print(DIVIDER)
    print("  HYBRID ANOMALY DETECTION — FULL PIPELINE")
    print(DIVIDER)
    print(f"  Python     : {PYTHON}")
    print(f"  Main script: {MAIN_PY}")
    print(f"  Train data : {TRAIN_DATA}")
    print(f"  Real data  : {REAL_DATA}")
    print(f"  Mode       : {MODE}")
    print()

    steps = [
        (
            "STEP 1 — TRAINING (master_features.csv)",
            [PYTHON, MAIN_PY, "--action", "train",
             "--mode", MODE, "--data", TRAIN_DATA]
        ),
        (
            "STEP 2 — CROSS-VALIDATION 5-Fold (master_features.csv)",
            [PYTHON, MAIN_PY, "--action", "cv",
             "--mode", MODE, "--data", TRAIN_DATA]
        ),
        (
            "STEP 3 — INFERENCE pada data asli (master_features.csv)",
            [PYTHON, MAIN_PY, "--action", "inference",
             "--mode", MODE, "--data", REAL_DATA]
        ),
    ]

    results = {}
    for label, cmd in steps:
        code = run_step(label, cmd)
        results[label] = "✅ SUKSES" if code == 0 else "❌ GAGAL"

    # ── Ringkasan Akhir ─────────────────────────────────────────────────────
    print()
    print(DIVIDER)
    print("  RINGKASAN PIPELINE")
    print(DIVIDER)
    for label, status in results.items():
        print(f"  {status}  {label}")
    print(DIVIDER)
    print()


if __name__ == "__main__":
    main()
