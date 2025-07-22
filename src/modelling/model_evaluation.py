import os
import logging
from typing import Any, Dict, Tuple
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report

# Configure logging to both console and file
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "model_evaluation.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode='a')
    ]
)

def load_model(model_path: str) -> Any:
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logging.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        raise

def load_test_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        df = pd.read_csv(file_path)
        X = df.drop(columns=['label']).values
        y = df['label'].values
        logging.info(f"Loaded test data from {file_path} with shape {df.shape}")
        return X, y
    except Exception as e:
        logging.error(f"Failed to load test data from {file_path}: {e}")
        raise

def evaluate_model(model: Any, X_test: Any, y_test: Any) -> Tuple[Dict[str, float], Dict[str, Any]]:
    try:
        y_pred = model.predict(X_test)
        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }
        classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
        logging.info("Model evaluation completed.")
        return metrics_dict, classification_report_dict
    except Exception as e:
        logging.error(f"Model evaluation failed: {e}")
        raise

def save_json(data: Dict, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        logging.info(f"Saved JSON to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON to {file_path}: {e}")
        raise

def main() -> None:
    try:
        model = load_model("models/random_forest_model.pkl")
        X_test, y_test = load_test_data("data/interim/test_bow.csv")
        metrics_dict, classification_report_dict = evaluate_model(model, X_test, y_test)
        save_json(metrics_dict, "reports/metrics.json")
        save_json(classification_report_dict, "reports/classification_report.json")
        logging.info("Model evaluation and report saving completed successfully.")
    except Exception as e:
        logging.critical(f"Model evaluation pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
