import argparse
import logging
import os
from config.settings import LOG_DIR
from controller.hybrid_controller import HybridController
from models.model_engine import IsolationForestEngine
from thresholds.threshold_engine import AdaptiveThreshold, StrictThreshold

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
    parser.add_argument("--action", type=str, choices=["train", "inference"], default="train",
                        help="Action to perform: 'train' (train and save model) or 'inference' (load and predict)")
    parser.add_argument("--mode", type=str, choices=["adaptive", "strict"], default="adaptive",
                        help="Threshold mode: 'adaptive' (Percentile) or 'strict' (ROC)")
    parser.add_argument("--data", type=str, default="expanded_dataset_v3.csv",
                        help="Dataset filename located in parent features folder or datasets dir")
    
    args = parser.parse_args()
    
    # Path resolution to fallback paths since raw datasets logic isn't fully migrated
    possible_paths = [
        os.path.join("data", "datasets", args.data),
        os.path.join("..", "entropy-malware-framework", "features", args.data),
        os.path.join("..", "features", args.data),
        os.path.join("d:\\malware_entropy_ml\\entropy-malware-framework\\features", args.data)
    ]
    
    data_path = None
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            break
            
    if not data_path:
        print(f"Dataset '{args.data}' not found. Please ensure it is inside data/datasets/ or the old features/ folder.")
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

    model_save_path = os.path.join("models", "saved_models", "iso_v2.pkl")
    
    # 4. Run pipeline
    metrics, threshold = controller.run(action=args.action, mode=args.mode, model_save_path=model_save_path)
    
    print("\n" + "="*50)
    print(" SYSTEM EXECUTION COMPLETED ")
    print("="*50)
    print(f"  Action Performed : {args.action.upper()}")
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
