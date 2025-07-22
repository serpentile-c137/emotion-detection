import os
import re
import logging
from typing import Any, Optional
import numpy as np
import pandas as pd
import nltk
from pandas import DataFrame
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def download_nltk_resources() -> None:
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        logging.info("Downloaded required NLTK resources.")
    except Exception as e:
        logging.error(f"Failed to download NLTK resources: {e}")
        raise

def lemmatization(text: str) -> str:
    try:
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        lemmatized = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized)
    except Exception as e:
        logging.error(f"Lemmatization error: {e}")
        return text

def remove_stop_words(text: str) -> str:
    try:
        stop_words = set(stopwords.words("english"))
        filtered = [word for word in str(text).split() if word not in stop_words]
        return " ".join(filtered)
    except Exception as e:
        logging.error(f"Stop word removal error: {e}")
        return text

def removing_numbers(text: str) -> str:
    try:
        return ''.join([char for char in text if not char.isdigit()])
    except Exception as e:
        logging.error(f"Number removal error: {e}")
        return text

def lower_case(text: str) -> str:
    try:
        return " ".join([word.lower() for word in text.split()])
    except Exception as e:
        logging.error(f"Lowercase conversion error: {e}")
        return text

def removing_punctuations(text: str) -> str:
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()
    except Exception as e:
        logging.error(f"Punctuation removal error: {e}")
        return text

def removing_urls(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logging.error(f"URL removal error: {e}")
        return text

def remove_small_sentences(df: DataFrame) -> None:
    try:
        for i in range(len(df)):
            if len(str(df.text.iloc[i]).split()) < 3:
                df.text.iloc[i] = np.nan
        logging.info("Removed small sentences from DataFrame.")
    except Exception as e:
        logging.error(f"Error removing small sentences: {e}")

def normalize_text(df: DataFrame) -> DataFrame:
    try:
        df = df.copy()
        df.content = df.content.apply(lower_case)
        df.content = df.content.apply(remove_stop_words)
        df.content = df.content.apply(removing_numbers)
        df.content = df.content.apply(removing_punctuations)
        df.content = df.content.apply(removing_urls)
        df.content = df.content.apply(lemmatization)
        logging.info("Normalized DataFrame text.")
        return df
    except Exception as e:
        logging.error(f"Error normalizing DataFrame: {e}")
        return df

def normalized_sentence(sentence: str) -> str:
    try:
        sentence = lower_case(sentence)
        sentence = remove_stop_words(sentence)
        sentence = removing_numbers(sentence)
        sentence = removing_punctuations(sentence)
        sentence = removing_urls(sentence)
        sentence = lemmatization(sentence)
        return sentence
    except Exception as e:
        logging.error(f"Error normalizing sentence: {e}")
        return sentence

def load_data(file_path: str) -> Optional[DataFrame]:
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        return None

def save_data(df: DataFrame, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info(f"Saved data to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save data to {file_path}: {e}")

def main() -> None:
    download_nltk_resources()
    train_data = load_data("data/raw/train.csv")
    test_data = load_data("data/raw/test.csv")
    if train_data is not None:
        train_data = normalize_text(train_data)
        save_data(train_data, "data/processed/train.csv")
    if test_data is not None:
        test_data = normalize_text(test_data)
        save_data(test_data, "data/processed/test.csv")
    logging.info("Data preprocessing completed.")

if __name__ == "__main__":
    main()
