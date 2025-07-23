import os
import json
import pickle
import pandas as pd
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_model(model_path: str) -> object:
    """
    Load a trained model from a file.

    Args:
        model_path (str): Path to the model file.

    Returns:
        object: Loaded model.
    """
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        logging.info(f"Model loaded successfully from {model_path}.")
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found at {model_path}.")
        raise

def load_vectorizer(vectorizer_path: str) -> object:
    """
    Load a saved TfidfVectorizer from a file.

    Args:
        vectorizer_path (str): Path to the vectorizer file.

    Returns:
        object: Loaded TfidfVectorizer.
    """
    try:
        with open(vectorizer_path, "rb") as file:
            vectorizer = pickle.load(file)
        logging.info(f"Vectorizer loaded successfully from {vectorizer_path}.")
        return vectorizer
    except FileNotFoundError:
        logging.error(f"Vectorizer file not found at {vectorizer_path}.")
        raise

def load_test_data(file_path: str) -> pd.DataFrame:
    """
    Load test data from a CSV file.

    Args:
        file_path (str): Path to the test data file.

    Returns:
        pd.DataFrame: Loaded test data.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Test data loaded successfully from {file_path}.")
        return data
    except FileNotFoundError:
        logging.error(f"Test data file not found at {file_path}.")
        raise

def evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate the model on test data and compute metrics.

    Args:
        model (object): Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics.
    """
    try:
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }
        logging.info("Model evaluation completed successfully.")
        return metrics
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

def save_metrics(metrics: Dict[str, float], file_path: str) -> None:
    """
    Save evaluation metrics to a JSON file.

    Args:
        metrics (Dict[str, float]): Dictionary containing evaluation metrics.
        file_path (str): Path to save the metrics file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)
        logging.info(f"Metrics saved successfully to {file_path}.")
    except Exception as e:
        logging.error(f"Error saving metrics to {file_path}: {e}")
        raise

def main():
    """
    Main function to execute the model evaluation pipeline.
    """
    try:
        # Load model and vectorizer
        model = load_model("models/random_forest_model.pkl")
        vectorizer = load_vectorizer("models/tfidf_vectorizer.pkl")

        # Load test data
        test_data = load_test_data("data/interim/test_tfidf.csv")
        X_test_raw = test_data.drop(columns=['sentiment']).values  # Drop the sentiment column to get TF-IDF features
        y_test = test_data['sentiment'].values  # Extract labels

        # Transform test data using the loaded vectorizer
        # Ensure X_test_raw contains valid text data
        X_test = vectorizer.transform([" ".join(map(str, row)) for row in X_test_raw])

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # Save metrics
        save_metrics(metrics, "reports/metrics.json")
        logging.info("Model evaluation pipeline executed successfully.")
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    main()