import os
import logging
from typing import Tuple
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

# Configure logging to both console and file
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "modelling.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode='a')
    ]
)

def load_params(params_path: str = "params.yaml") -> dict:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Loaded parameters from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Failed to load parameters from {params_path}: {e}")
        raise

def load_train_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded training data from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load training data from {file_path}: {e}")
        raise

def get_features_and_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    try:
        X = df.drop(columns=['label']).values
        y = df['label'].values
        return X, y
    except Exception as e:
        logging.error(f"Failed to extract features and labels: {e}")
        raise

def train_model(X: np.ndarray, y: np.ndarray, n_estimators: int, max_depth: int) -> RandomForestClassifier:
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X, y)
        logging.info(f"Trained RandomForest model with n_estimators={n_estimators}, max_depth={max_depth}")
        return model
    except Exception as e:
        logging.error(f"Failed to train model: {e}")
        raise

def save_model(model: RandomForestClassifier, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Saved model to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save model to {file_path}: {e}")
        raise

def main() -> None:
    try:
        params = load_params("params.yaml")
        n_estimators = params["model_building"]["n_estimators"]
        max_depth = params["model_building"]["max_depth"]

        train_data = load_train_data("data/interim/train_bow.csv")
        X_train, y_train = get_features_and_labels(train_data)

        model = train_model(X_train, y_train, n_estimators, max_depth)
        save_model(model, "models/random_forest_model.pkl")

        logging.info("Model training and saving completed successfully.")
    except Exception as e:
        logging.critical(f"Model training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
