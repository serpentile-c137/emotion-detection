import os
import logging
from typing import Tuple
import pandas as pd
import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml

# Configure logging to both console and file
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "features.log")
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

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path).dropna(subset=['content'])
        logging.info(f"Loaded data from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        raise

def extract_features_and_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    try:
        X = df['content'].values
        y = df['sentiment'].values
        return X, y
    except Exception as e:
        logging.error(f"Failed to extract features and labels: {e}")
        raise

def vectorize_data(X_train: np.ndarray, X_test: np.ndarray, max_features: int) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        logging.info(f"Vectorized data with max_features={max_features}")
        return X_train_tfidf, X_test_tfidf, vectorizer
    except Exception as e:
        logging.error(f"Vectorization failed: {e}")
        raise

def save_features(X: np.ndarray, y: np.ndarray, file_path: str) -> None:
    try:
        df = pd.DataFrame(X.toarray())
        df['label'] = y
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info(f"Saved features to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save features to {file_path}: {e}")
        raise

def main() -> None:
    try:
        params = load_params("params.yaml")
        max_features = params["feature_engg"]["max_features"]

        train_data = load_data("data/processed/train.csv")
        test_data = load_data("data/processed/test.csv")

        X_train, y_train = extract_features_and_labels(train_data)
        X_test, y_test = extract_features_and_labels(test_data)

        X_train_bow, X_test_bow, _ = vectorize_data(X_train, X_test, max_features)

        save_features(X_train_bow, y_train, "data/interim/train_bow.csv")
        save_features(X_test_bow, y_test, "data/interim/test_bow.csv")

        logging.info("Feature engineering completed successfully.")
    except Exception as e:
        logging.critical(f"Feature engineering failed: {e}")
        raise

if __name__ == "__main__":
    main()
