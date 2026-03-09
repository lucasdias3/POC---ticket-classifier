import json
import joblib
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results_metrics"
RESULTS_DIR.mkdir(exist_ok=True)

# candidate model filenames (search order)
MODEL_CANDIDATES = [
    "ticket_classifier_lr.joblib",
    "ticket_classifier_model.joblib",
    "ticket_classifier_lr_combined.joblib",
    "ticket_classifier_text_lr.joblib",
    "ticket_classifier_lr_combined.joblib",
    "ticket_classifier_text_lr.joblib",
]

TESTSET_PATH = PROJECT_ROOT / "data" / "testset_kgb.json"

def _preprocess(text: str) -> str:
    if text is None:
        return ""
    s = str(text).lower().strip()
    # simple accent removal + keep alnum and spaces
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    s = " ".join(s.split())
    return s

def find_model():
    for fname in MODEL_CANDIDATES:
        p = PROJECT_ROOT / fname
        if p.exists():
            return p
    # try common files in root and model outputs
    for p in PROJECT_ROOT.glob("**/*.joblib"):
        if p.is_file():
            return p
    raise FileNotFoundError("Modelo .joblib não encontrado. Rode scripts/train_and_save.py antes.")

def load_testset(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def prepare_inputs(df: pd.DataFrame):
    # prefer preprocessed_text, fallback to preprocessing text
    texts = []
    for _, r in df.iterrows():
        if r.get("preprocessed_text"):
            texts.append(str(r["preprocessed_text"]))
        else:
            texts.append(_preprocess(r.get("text", "")))
    categories = df["category"].astype(str).tolist()
    priorities = df["priority"].astype(str).tolist() if "priority" in df.columns else [None]*len(df)
    return texts, categories, priorities, df

def safe_predict(model, texts, priorities):
    # try predicting with plain text list (TF-IDF pipeline)
    try:
        preds = model.predict(texts)
        return list(preds)
    except Exception:
        pass
    # try DataFrame with preprocessed_text + priority (for ColumnTransformer pipelines)
    try:
        Xdf = pd.DataFrame({"preprocessed_text": texts, "priority": priorities})
        preds = model.predict(Xdf)
        return list(preds)
    except Exception as e:
        raise RuntimeError(f"Falha ao usar o modelo para predizer: {e}")

def save_metrics(metrics: dict):
    # JSON (machine-readable)
    mf = RESULTS_DIR / "evaluation_metrics.json"
    with mf.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # CSV summary (compact)
    summary = {
        "accuracy": metrics.get("accuracy"),
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "f1_macro": metrics.get("f1_macro")
    }
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "evaluation_metrics_summary.csv", index=False)

    # Human-readable one-line-per-key file
    txt_file = RESULTS_DIR / "evaluation_metrics.txt"
    with txt_file.open("w", encoding="utf-8") as f:
        for k, v in metrics.items():
            # convert to string and escape newlines so each key:value stays in one line
            v_str = str(v) if v is not None else ""
            v_str_escaped = v_str.replace("\r\n", "\\n").replace("\n", "\\n")
            f.write(f"{k}: {v_str_escaped}\n")

def plot_and_save(metrics, y_true, y_pred, priorities, categories_list):
    # scalar metrics bar
    labels = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    vals = [metrics[k] for k in labels]
    plt.figure(figsize=(6,4))
    plt.bar(labels, vals, color="tab:blue")
    plt.ylim(0,1)
    plt.title("Classification summary")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "metrics_summary.png", dpi=150)
    plt.close()

    # per-category F1
    per_prec, per_rec, per_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=categories_list, zero_division=0)
    x = np.arange(len(categories_list))
    plt.figure(figsize=(max(6,len(categories_list)*0.8),4))
    plt.bar(x-0.2, per_prec, width=0.2, label="precision")
    plt.bar(x, per_rec, width=0.2, label="recall")
    plt.bar(x+0.2, per_f1, width=0.2, label="f1")
    plt.xticks(x, categories_list, rotation=45, ha="right")
    plt.legend()
    plt.title("Per-category precision/recall/f1")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "per_category_prf.png", dpi=150)
    plt.close()

    # distribution of categories and priorities in testset
    import collections
    cat_counts = collections.Counter(y_true)
    pr_counts = collections.Counter(priorities)

    # category distribution
    cats = list(cat_counts.keys())
    vals = [cat_counts[c] for c in cats]
    plt.figure(figsize=(max(6,len(cats)*0.6),4))
    plt.bar(cats, vals, color="tab:green")
    plt.title("Category distribution (testset)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "distribution_category.png", dpi=150)
    plt.close()

    # priority distribution (percent)
    pr_keys = list(pr_counts.keys())
    pr_vals = [pr_counts[k] for k in pr_keys]
    total = sum(pr_vals) if pr_vals else 1
    pr_pct = [v/total*100 for v in pr_vals]
    plt.figure(figsize=(6,4))
    plt.bar(pr_keys, pr_pct, color="tab:orange")
    plt.ylabel("Percentage (%)")
    plt.title("Priority distribution (%) (testset)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "distribution_priority_pct.png", dpi=150)
    plt.close()

def main():
    model_path = find_model()
    print("Usando modelo:", model_path)
    model = joblib.load(model_path)

    if not TESTSET_PATH.exists():
        raise FileNotFoundError("Arquivo de teste não encontrado: data/testset_kgb.json")
    df = load_testset(TESTSET_PATH)
    texts, y_true, priorities, raw_df = prepare_inputs(df)

    y_pred = safe_predict(model, texts, priorities)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "classification_report": report
    }

    save_metrics(metrics)

    # categories order for per-category plot: all unique categories seen in testset
    categories_list = sorted(list(set(y_true)))
    plot_and_save(metrics, y_true, y_pred, priorities, categories_list)

    print("Avaliação concluída. Métricas e gráficos salvos em:", RESULTS_DIR)

if __name__ == "__main__":
    main()
# filepath: c:\Users\lucleite\OneDrive - Capgemini\Vivo\scripts-nlp-main\case_vivo\ticket-classifier-poc\tests\test_evaluate_model.py
import json
import joblib
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results_metrics"
RESULTS_DIR.mkdir(exist_ok=True)

# candidate model filenames (search order)
MODEL_CANDIDATES = [
    "ticket_classifier_lr.joblib",
    "ticket_classifier_model.joblib",
    "ticket_classifier_lr_combined.joblib",
    "ticket_classifier_text_lr.joblib",
    "ticket_classifier_lr_combined.joblib",
    "ticket_classifier_text_lr.joblib",
]

TESTSET_PATH = PROJECT_ROOT / "data" / "testset_kgb.json"

def _preprocess(text: str) -> str:
    if text is None:
        return ""
    s = str(text).lower().strip()
    # simple accent removal + keep alnum and spaces
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    s = " ".join(s.split())
    return s

def find_model():
    for fname in MODEL_CANDIDATES:
        p = PROJECT_ROOT / fname
        if p.exists():
            return p
    # try common files in root and model outputs
    for p in PROJECT_ROOT.glob("**/*.joblib"):
        if p.is_file():
            return p
    raise FileNotFoundError("Modelo .joblib não encontrado. Rode scripts/train_and_save.py antes.")

def load_testset(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def prepare_inputs(df: pd.DataFrame):
    # prefer preprocessed_text, fallback to preprocessing text
    texts = []
    for _, r in df.iterrows():
        if r.get("preprocessed_text"):
            texts.append(str(r["preprocessed_text"]))
        else:
            texts.append(_preprocess(r.get("text", "")))
    categories = df["category"].astype(str).tolist()
    priorities = df["priority"].astype(str).tolist() if "priority" in df.columns else [None]*len(df)
    return texts, categories, priorities, df

def safe_predict(model, texts, priorities):
    # try predicting with plain text list (TF-IDF pipeline)
    try:
        preds = model.predict(texts)
        return list(preds)
    except Exception:
        pass
    # try DataFrame with preprocessed_text + priority (for ColumnTransformer pipelines)
    try:
        Xdf = pd.DataFrame({"preprocessed_text": texts, "priority": priorities})
        preds = model.predict(Xdf)
        return list(preds)
    except Exception as e:
        raise RuntimeError(f"Falha ao usar o modelo para predizer: {e}")

def save_metrics(metrics: dict):
    # JSON (machine-readable)
    mf = RESULTS_DIR / "evaluation_metrics.json"
    with mf.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # CSV summary (compact)
    summary = {
        "accuracy": metrics.get("accuracy"),
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "f1_macro": metrics.get("f1_macro")
    }
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "evaluation_metrics_summary.csv", index=False)

    # Human-readable one-line-per-key file
    txt_file = RESULTS_DIR / "evaluation_metrics.txt"
    with txt_file.open("w", encoding="utf-8") as f:
        for k, v in metrics.items():
            # convert to string and escape newlines so each key:value stays in one line
            v_str = str(v) if v is not None else ""
            v_str_escaped = v_str.replace("\r\n", "\\n").replace("\n", "\\n")
            f.write(f"{k}: {v_str_escaped}\n")

def plot_and_save(metrics, y_true, y_pred, priorities, categories_list):
    # scalar metrics bar
    labels = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    vals = [metrics[k] for k in labels]
    plt.figure(figsize=(6,4))
    plt.bar(labels, vals, color="tab:blue")
    plt.ylim(0,1)
    plt.title("Classification summary")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "metrics_summary.png", dpi=150)
    plt.close()

    # per-category F1
    per_prec, per_rec, per_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=categories_list, zero_division=0)
    x = np.arange(len(categories_list))
    plt.figure(figsize=(max(6,len(categories_list)*0.8),4))
    plt.bar(x-0.2, per_prec, width=0.2, label="precision")
    plt.bar(x, per_rec, width=0.2, label="recall")
    plt.bar(x+0.2, per_f1, width=0.2, label="f1")
    plt.xticks(x, categories_list, rotation=45, ha="right")
    plt.legend()
    plt.title("Per-category precision/recall/f1")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "per_category_prf.png", dpi=150)
    plt.close()

    # distribution of categories and priorities in testset
    import collections
    cat_counts = collections.Counter(y_true)
    pr_counts = collections.Counter(priorities)

    # category distribution
    cats = list(cat_counts.keys())
    vals = [cat_counts[c] for c in cats]
    plt.figure(figsize=(max(6,len(cats)*0.6),4))
    plt.bar(cats, vals, color="tab:green")
    plt.title("Category distribution (testset)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "distribution_category.png", dpi=150)
    plt.close()

    # priority distribution (percent)
    pr_keys = list(pr_counts.keys())
    pr_vals = [pr_counts[k] for k in pr_keys]
    total = sum(pr_vals) if pr_vals else 1
    pr_pct = [v/total*100 for v in pr_vals]
    plt.figure(figsize=(6,4))
    plt.bar(pr_keys, pr_pct, color="tab:orange")
    plt.ylabel("Percentage (%)")
    plt.title("Priority distribution (%) (testset)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "distribution_priority_pct.png", dpi=150)
    plt.close()

def main():
    model_path = find_model()
    print("Usando modelo:", model_path)
    model = joblib.load(model_path)

    if not TESTSET_PATH.exists():
        raise FileNotFoundError("Arquivo de teste não encontrado: data/testset_kgb.json")
    df = load_testset(TESTSET_PATH)
    texts, y_true, priorities, raw_df = prepare_inputs(df)

    y_pred = safe_predict(model, texts, priorities)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "classification_report": report
    }

    save_metrics(metrics)

    # categories order for per-category plot: all unique categories seen in testset
    categories_list = sorted(list(set(y_true)))
    plot_and_save(metrics, y_true, y_pred, priorities, categories_list)

    print("Avaliação concluída. Métricas e gráficos salvos em:", RESULTS_DIR)

if __name__ == "__main__":
    main()