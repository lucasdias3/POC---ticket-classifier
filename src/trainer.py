from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import pandas as pd
import joblib
import json

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return pd.DataFrame(data)

def train_model(data):
    # Accept list of dicts or DataFrame
    if isinstance(data, list):
        data = pd.DataFrame(data)
    elif not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except Exception as e:
            raise TypeError("train_model espera list[dict] ou pd.DataFrame") from e

    # Ensure expected columns exist
    if 'preprocessed_text' not in data.columns or 'category' not in data.columns:
        raise KeyError("Dados devem conter colunas 'preprocessed_text' e 'category'.")

    X = data['preprocessed_text']
    y = data['category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=1),
        LogisticRegression(solver="lbfgs", max_iter=2000, class_weight="balanced")
    )
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def save_model(model, filename):
    joblib.dump(model, filename)

if __name__ == "__main__":
    data = load_data('../data/knowledge_base.json')
    model, X_test, y_test = train_model(data)
    save_model(model, 'ticket_classifier_model.joblib')