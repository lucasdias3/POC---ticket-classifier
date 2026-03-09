import joblib
import json
import pandas as pd
from typing import Tuple, Dict, Any
from pathlib import Path
import time
from functools import lru_cache
import os

# model filename candidates (aceita nome antigo e nome atual)
_MODEL_CANDIDATES = [
    Path("ticket_classifier_model.joblib"),
    Path("ticket_classifier_lr.joblib"),
    Path("ticket_classifier_nb.joblib")
]

# preferência pelo data file correto
DATA_DEFAULT_PATH = Path("data/knowledge_base.json")

# determina caminho do modelo disponível
def _default_model_path() -> Path:
    for p in _MODEL_CANDIDATES:
        if p.exists():
            return p
    return _MODEL_CANDIDATES[0]  # fallback para primeiro nome (erro se não existir)

MODEL_DEFAULT_PATH = _default_model_path()

def _model_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0

def _cached_predict(text: str, model_mtime: float = None):
    """
    Predict without caching: carrega o modelo a cada chamada.
    Mantive o parâmetro model_mtime para compatibilidade com chamadas existentes,
    mas ele é ignorado aqui.
    Retorna (category, priority).
    """
    # carregar modelo (sem cache)
    model_path = MODEL_DEFAULT_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em '{model_path}'")
    model = joblib.load(model_path)

    # preprocessar texto
    try:
        try:
            from src.preprocessing import preprocess_text
        except Exception:
            from preprocessing import preprocess_text
        tx = preprocess_text(text, do_stem=False, remove_stopwords=True)
    except Exception:
        tx = str(text).lower()

    # predizer categoria
    cat = str(model.predict([tx])[0])

    # mapping priority (recarrega KB a cada chamada)
    priority = "medium"
    try:
        with open(DATA_DEFAULT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if "priority" in df.columns and "category" in df.columns:
            grp = df.groupby(["category", "priority"]).size().reset_index(name="count")
            best = grp.loc[grp.groupby("category")["count"].idxmax()][["category", "priority"]]
            mapping = dict(zip(best["category"], best["priority"]))
            priority = mapping.get(cat, priority)
    except Exception:
        # se falhar, manter prioridade padrão
        pass

    return cat, priority

def classify_ticket(model_or_ticket, ticket=None) -> Tuple[str, str]:
    """
    Uso flexível:
      - classify_ticket(ticket_dict) -> carrega/usa modelo salvo
      - classify_ticket(model, ticket_dict) -> usa o modelo fornecido

    Retorna (categoria:str, prioridade:str)
    """
    if ticket is None:
        ticket = model_or_ticket
        models_mtime = _model_mtime(MODEL_DEFAULT_PATH)
        category, priority = _cached_predict(str(ticket.get("text", "")), models_mtime)
        return str(category), str(priority)
    else:
        model = model_or_ticket
        if not isinstance(ticket, dict) or "text" not in ticket:
            raise ValueError("ticket deve ser um dict com chave 'text'")
        text = str(ticket["text"])
        try:
            try:
                from src.preprocessing import preprocess_text
            except Exception:
                from preprocessing import preprocess_text
            tx = preprocess_text(text, do_stem=False, remove_stopwords=True)
        except Exception:
            tx = text.lower()
        if hasattr(model, "predict"):
            cat = model.predict([tx])[0]
            # mapear prioridade
            with open(DATA_DEFAULT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            mapping = {}
            if "category" in df.columns and "priority" in df.columns:
                for c, grp in df.groupby("category"):
                    modes = grp["priority"].mode()
                    mapping[str(c)] = str(modes.iloc[0]) if not modes.empty else "medium"
            return str(cat), mapping.get(str(cat), "medium")
        raise ValueError("Modelo fornecido não é suportado; forneça um objeto com método predict.")

# para compatibilidade com uso direto via classe (se necessário)
class TicketInferencer:
    def __init__(self, model_path: str = None, data_path: str = None):
        self.model_path = Path(model_path) if model_path else MODEL_DEFAULT_PATH
        self.data_path = Path(data_path) if data_path else DATA_DEFAULT_PATH
        # tentar carregar o modelo; se não existir ou ocorrer erro, não lançar na importação
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
            else:
                # não treinamos automaticamente no import — deixar None e carregar treinado depois
                self.model = None
        except Exception as e:
            # avisar e continuar (o método de predição deve carregar o modelo quando necessário)
            print("Aviso: falha ao carregar modelo em inferencer:", e)
            self.model = None

    def classify_ticket(self, ticket: Dict[str, Any]) -> Tuple[str, str]:
        return classify_ticket(self.model, ticket)

if __name__ == "__main__":
    # exemplo rápido
    model = _load_model_or_train()
    with open(DATA_DEFAULT_PATH, "r", encoding="utf-8") as f:
        tickets = json.load(f)
    inf = TicketInferencer()
    print(inf.classify_ticket(tickets[0]))