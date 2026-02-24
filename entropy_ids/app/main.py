import argparse
import logging
import os
import sys

# Ensure current directory (app/) is in sys.path for sibling imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import LOG_DIR, MODEL_DIR, DATA_DIR
from controller.hybrid_controller import HybridController
from core.model_engine import IsolationForestEngine
from core.threshold_engine import AdaptiveThreshold, StrictThreshold

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, "system.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Hybrid Anomaly Detection System")
    parser.add_argument("--action", type=str, choices=["train", "inference", "cv"], default="train",
                        help="Action to perform: 'train', 'inference', or 'cv'")
    parser.add_argument("--mode", type=str, choices=["adaptive", "strict"], default="adaptive",
                        help="Threshold mode: 'adaptive' (Percentile) or 'strict' (ROC)")
    parser.add_argument("--data", type=str, default="expanded_dataset_v3.csv",
                        help="Dataset filename located in parent data/features/ dir")
    parser.add_argument("--cv", type=str, choices=["true", "false"], default="false",
                        help="Enable cross-validation if action is 'cv' or boolean flag")
    
    args = parser.parse_args()
    
    # Path resolution: search through all known data subdirectories
    base_data_dir = os.path.dirname(DATA_DIR)  # points to data/
    possible_paths = [
        os.path.join(DATA_DIR, args.data),                          # data/features/
        os.path.join(base_data_dir, "synthetic", args.data),        # data/synthetic/
        os.path.join(base_data_dir, "raw", args.data),              # data/raw/
        os.path.join(base_data_dir, "final_datashet", args.data),   # data/final_datashet/
    ]
    
    data_path = None
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            print(f"[DATA] Located '{args.data}' at: {os.path.abspath(p)}")
            break
            
    if not data_path:
        print(f"Dataset '{args.data}' not found. Searched in:")
        for p in possible_paths:
            print(f"  - {os.path.abspath(p)}")
        print(f"Error: Could not locate '{args.data}'. Please place the file in one of the above directories.")
        return

    print(f"\n[ENGINE START] Booting Hybrid Anomaly System | Action: {args.action.upper()} | Mode: {args.mode.upper()}")
    
    # 1. Dependency Injection: Select Threshold Engine
    if args.mode == "adaptive":
        thresh_engine = AdaptiveThreshold()
    else:
        thresh_engine = StrictThreshold()
        
    # 2. Dependency Injection: Instantiate Model Engine
    model_engine = IsolationForestEngine()
    
    # 3. Inject dependencies into Controller
    controller = HybridController(
        data_path=data_path,
        model=model_engine,
        threshold_engine=thresh_engine
    )

    model_save_path = os.path.join(MODEL_DIR, "production", "iso_v1_production.pkl")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # 4. Run pipeline
    execution_action = "cv" if args.cv.lower() == "true" else args.action
    metrics, threshold = controller.run(action=execution_action, mode=args.mode, model_save_path=model_save_path)
    
    print("\n" + "="*50)
    print(" SYSTEM EXECUTION COMPLETED ")
    print("="*50)
    print(f"  Action Performed : {execution_action.upper()}")
    print(f"  Mode Used        : {args.mode.upper()}")
    print(f"  Threshold Set    : {threshold:.4f}")
    print("-" * 50)
    print(f"  Precision      : {metrics['precision']:.4f}")
    print(f"  Recall         : {metrics['recall']:.4f}")
    print(f"  F1 Score       : {metrics['f1']:.4f}")
    print(f"  ROC AUC        : {metrics['roc_auc']:.4f}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")
    print("="*50)
    print(f"Logs and predictions saved in: {os.path.abspath(LOG_DIR)}")

if __name__ == "__main__":
    main()
