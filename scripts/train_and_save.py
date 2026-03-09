import traceback
import joblib
from pathlib import Path
import sys
import importlib

# garantir que o root do projeto esteja no sys.path para permitir "from src import ..."
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Preferência por 'knowledge_base.json', fallback para 'knowledge_database.json'
PREFERRED = Path("data/knowledge_base.json")
FALLBACK = Path("data/knowledge_database.json")
DATA_PATH = PREFERRED if PREFERRED.exists() else (FALLBACK if FALLBACK.exists() else PREFERRED)
MODEL_OUT_DIR = Path(".")

# carregar módulo trainer de forma robusta
try:
    trainer_mod = importlib.import_module("src.trainer")
except Exception:
    try:
        trainer_mod = importlib.import_module("trainer")
    except Exception as e:
        raise ImportError("Não foi possível importar 'src.trainer' nem 'trainer'") from e

# extrair funções existentes
load_data = getattr(trainer_mod, "load_data", None)
train_and_save_both = getattr(trainer_mod, "train_and_save_both", None)
train_text_only = getattr(trainer_mod, "train_text_only", None)
train_model = getattr(trainer_mod, "train_model", None)

if load_data is None or train_model is None:
    raise ImportError("Funções mínimas 'load_data' e 'train_model' devem existir em trainer.py")

try:
    print("Carregando dados de:", DATA_PATH)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {DATA_PATH.resolve()}")

    df = load_data(str(DATA_PATH))
    print("Dados carregados, registros:", len(df) if hasattr(df, "__len__") else "unknown")

    # chama a função de treino disponível (prioridade: train_and_save_both > train_text_only > train_model)
    if train_and_save_both:
        print("Usando train_and_save_both (treino combinado tabular+texto)...")
        res = train_and_save_both(df, out_dir=str(MODEL_OUT_DIR), prefix="ticket_classifier")
        print("Modelo salvo em:", res.get("lr_model", res))
    elif train_text_only:
        print("Usando train_text_only (TF-IDF sobre texto)...")
        res = train_text_only(df, out_dir=str(MODEL_OUT_DIR), prefix="ticket_classifier_text")
        print("Modelo salvo em:", res.get("model", res))
    else:
        print("Usando train_model (pipeline básico). Treinando, salvando e calculando métricas...")
        model, X_test, y_test = train_model(df)
        # garantir listas
        X_test = list(X_test)
        y_test = list(y_test)

        # calcular previsões e métricas
        try:
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
            import pandas as pd
        except Exception:
            raise RuntimeError("Instale sklearn e pandas para calcular/salvar métricas")

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
        report = classification_report(y_test, preds, zero_division=0)
        metrics = {
            "accuracy": float(acc),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "classification_report": report
        }

        # salvar modelo
        out_file = MODEL_OUT_DIR / "ticket_classifier_model.joblib"
        joblib.dump(model, out_file)

        # salvar métricas CSV e relatório em pasta 'ticket_classifier_model_results' no root do projeto
        results_dir = PROJECT_ROOT / "ticket_classifier_model_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_file = results_dir / "ticket_classifier_metrics.csv"
        pd.DataFrame([{
            "model": "pipeline_logistic_regression",
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"]
        }]).to_csv(csv_file, index=False)

        report_file = results_dir / "ticket_classifier_report.txt"
        report_file.write_text(metrics["classification_report"], encoding="utf-8")

        print("Modelo salvo em:", str(out_file))
        print("Métricas salvas em:", str(csv_file))
        print("Relatório salvo em:", str(report_file))

except Exception:
    print("Falha ao treinar/salvar os modelos. Traceback abaixo:")
    traceback.print_exc()