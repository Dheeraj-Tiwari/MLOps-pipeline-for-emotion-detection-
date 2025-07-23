import os
import yaml
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logging.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}.")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    try:
        data = pd.read_csv(file_path).dropna(subset=['content'])
        logging.info(f"Data loaded successfully from {file_path}.")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}.")
        raise
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV file: {e}")
        raise

def extract_features_and_labels(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Extract features and labels from the dataset.

    Args:
        data (pd.DataFrame): Input DataFrame.

    Returns:
        Tuple[pd.Series, pd.Series]: Features (X) and labels (y).
    """
    try:
        X = data['content'].values
        y = data['sentiment'].values
        logging.info("Features and labels extracted successfully.")
        return X, y
    except KeyError as e:
        logging.error(f"Missing expected column in DataFrame: {e}")
        raise

def apply_tfidf_vectorizer(X_train: pd.Series, X_test: pd.Series, max_features: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply TF-IDF Vectorizer to the training and test data.

    Args:
        X_train (pd.Series): Training data features.
        X_test (pd.Series): Test data features.
        max_features (int): Maximum number of features for TfidfVectorizer.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Transformed training and test data.
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        logging.info("TF-IDF Vectorizer applied successfully.")
        return X_train_tfidf, X_test_tfidf
    except Exception as e:
        logging.error(f"Error during TF-IDF transformation: {e}")
        raise

def save_data_with_labels(data: pd.DataFrame, labels: pd.Series, file_path: str) -> None:
    """
    Save TF-IDF transformed data along with labels to a CSV file.

    Args:
        data (pd.DataFrame): TF-IDF transformed data.
        labels (pd.Series): Labels corresponding to the data.
        file_path (str): Path to save the output CSV file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        tfidf_data = pd.DataFrame(data.toarray())
        tfidf_data['sentiment'] = labels  # Add the sentiment column
        tfidf_data.to_csv(file_path, index=False)
        logging.info(f"Data saved successfully to {file_path}.")
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {e}")
        raise

def main():
    """
    Main function to execute the feature engineering pipeline.
    """
    try:
        # Load configuration
        config = load_config("params.yaml")
        max_features = config['feature_engineering']['max_features']

        # Load processed train and test data
        train_data = load_data("data/processed/train.csv")
        test_data = load_data("data/processed/test.csv")

        # Extract features and labels
        X_train, y_train = extract_features_and_labels(train_data)
        X_test, y_test = extract_features_and_labels(test_data)

        # Apply TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Save transformed data with labels
        save_data_with_labels(X_train_tfidf, y_train, "data/interim/train_tfidf.csv")
        save_data_with_labels(X_test_tfidf, y_test, "data/interim/test_tfidf.csv")
        logging.info("Feature engineering pipeline executed successfully.")
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    main()