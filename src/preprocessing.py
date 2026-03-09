from functools import lru_cache
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import json
import re
import unicodedata
from typing import Iterable

# pequeno conjunto de stopwords em português (adicione/remova conforme necessário)
PT_STOPWORDS = {
    "de","da","do","dos","das","e","o","a","os","as","um","uma","uns","umas",
    "que","para","por","com","sem","na","no","nas","nos","em","ao","aos",
    "meu","minha","meus","minhas","seu","sua","seus","suas","eu","voce","você",
    "não","nao","isso","este","esta","isso","isso","como","já","ja","mais","ou",
    "tambem","também","entre","sobre","às","as"
}

_logger = logging.getLogger(__name__)

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

_non_word_re = re.compile(r"[^\w\s]", flags=re.UNICODE)
_digit_re = re.compile(r"\d+")

def _remove_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def _tokenize(text: str) -> Iterable[str]:
    return text.split()

@lru_cache(maxsize=8192)
def preprocess_text(text: str,
                    remove_stopwords: bool = True,
                    keep_numbers: bool = True,
                    lowercase: bool = True) -> str:
    """
    Gera um preprocessed_text a partir de text:
      - normaliza acentos
      - lowercase (opcional)
      - remove pontuação mantendo letras, dígitos e espaços
      - opcionalmente remove stopwords PT (lista interna)
      - colapsa espaços e retorna string limpa

    Uso:
      from src.preprocessing import preprocess_text
      pre = preprocess_text("Meu login NÃO funciona!!!", remove_stopwords=True)

    Parâmetros:
      text: texto de entrada
      remove_stopwords: True para remover palavras comuns em PT
      keep_numbers: True para preservar dígitos; False para remover
      lowercase: aplicar lower()
    """
    t = (text or "")
    if lowercase:
        t = t.lower()

    # remover acentos
    t = _remove_accents(t)

    # normalizar números (opcional)
    if not keep_numbers:
        t = re.sub(r"\d+", " ", t)

    # remover pontuação, manter letras a-z, dígitos e espaços
    t = re.sub(r"[^a-z0-9\s]", " ", t)

    # colapsar espaços
    t = re.sub(r"\s+", " ", t).strip()

    if remove_stopwords and t:
        tokens = [tok for tok in _tokenize(t) if tok not in PT_STOPWORDS]
        t = " ".join(tokens)

    return t

def prepare_data(data):
    texts = [preprocess_text(ticket.get("text", "")) for ticket in data]
    categories = [ticket.get("category") for ticket in data]
    priorities = [ticket.get("priority") for ticket in data]
    return texts, categories, priorities

if __name__ == "__main__":
    print("module preprocessing")