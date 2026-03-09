from typing import List, Dict
import json
import pandas as pd

# usar preprocess_text implementado em src.preprocessing
try:
    from src.preprocessing import preprocess_text
except Exception:
    # fallback se executar fora do package
    from preprocessing import preprocess_text

def load_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def preprocess_text_wrapper(text: str) -> str:
    # wrapper para manter compatibilidade de chamadas; desativa stemming por padrão
    return preprocess_text(text, do_stem=False, remove_stopwords=True)

def transform_data(tickets: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(tickets)
    if 'text' in df.columns:
        df['text'] = df['text'].astype(str).map(preprocess_text_wrapper)
    return df

def create_pipeline(file_path: str) -> pd.DataFrame:
    raw = load_data(file_path)
    return transform_data(raw)