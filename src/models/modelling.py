import pandas as pd
import numpy as np
import pickle
import yaml
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load configuration
with open("params.yaml", "r") as file:
    config = yaml.safe_load(file)

n_estimators = config['modelling']['n_estimators']
max_depth = config['modelling']['max_depth']
max_features = config['feature_engineering']['max_features']

# Load training data
train_data = pd.read_csv("data/processed/train.csv")

# Extract features and labels
x_train_raw = train_data['content'].fillna("").values  # Assuming 'content' column contains text data
y_train = train_data['sentiment'].values    # Assuming 'sentiment' column contains labels

# Apply TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=max_features)
x_train = vectorizer.fit_transform(x_train_raw)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(x_train, y_train)

# Save the trained model
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/random_forest_model.pkl", "wb"))

# Save the TF-IDF vectorizer for future use
pickle.dump(vectorizer, open("models/tfidf_vectorizer.pkl", "wb"))