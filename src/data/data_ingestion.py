import os
import logging
from typing import Tuple
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import yaml

# Configure logging to both console and file
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "data_ingestion.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode='a')
    ]
)

pd.set_option('future.no_silent_downcasting', True)

def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Loaded parameters from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Failed to load parameters from {params_path}: {e}")
        raise

def load_dataset(url: str) -> DataFrame:
    """Load dataset from a given URL."""
    try:
        df = pd.read_csv(url)
        logging.info(f"Loaded dataset from {url} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load dataset from {url}: {e}")
        raise

def preprocess_dataset(df: DataFrame) -> DataFrame:
    """Preprocess the dataset: drop columns, filter, and encode labels."""
    try:
        df = df.drop(columns=['tweet_id'])
        df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        df['sentiment'] = df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        logging.info(f"Preprocessed dataset. New shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

def split_dataset(df: DataFrame, test_size: float, random_state: int = 42) -> Tuple[DataFrame, DataFrame]:
    """Split the dataset into train and test sets."""
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        logging.info(f"Split data: train shape {train_data.shape}, test shape {test_data.shape}")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error during train-test split: {e}")
        raise

def save_datasets(train_data: DataFrame, test_data: DataFrame, output_dir: str = "data/raw") -> None:
    """Save train and test datasets to CSV files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "train.csv")
        test_path = os.path.join(output_dir, "test.csv")
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logging.info(f"Saved train data to {train_path} and test data to {test_path}")
    except Exception as e:
        logging.error(f"Failed to save datasets: {e}")
        raise

def main() -> None:
    """Main function to orchestrate data ingestion."""
    try:
        params = load_params("params.yaml")
        test_size = params["data_ingestion"]["test_size"]
        url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
        df = load_dataset(url)
        final_df = preprocess_dataset(df)
        train_data, test_data = split_dataset(final_df, test_size)
        save_datasets(train_data, test_data)
        logging.info("Data ingestion completed successfully.")
    except Exception as e:
        logging.critical(f"Data ingestion failed: {e}")
        raise

if __name__ == "__main__":
    main()
