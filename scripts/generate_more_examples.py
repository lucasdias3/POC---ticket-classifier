import json
import random
import unicodedata
from pathlib import Path

random.seed(42)

BASE = Path(__file__).resolve().parent.parent
IN_KB = BASE / "data" / "knowledge_base.json"
OUT_KB = BASE / "data" / "knowledge_base_augmented.json"

# targets: (category, priority) -> desired total count
TARGETS = {
    ("account_management", "low"): 100,
    ("billing", "medium"): 100,
    ("billing", "high"): 100,
    ("cancellation", "high"): 100,
    ("cancellation", "medium"): 100,
    ("technical_issue", "medium"): 100,
    ("technical_issue", "high"): 100,
}

products = ["internet", "mobile", "tv", "voip", "cloud", "portal", "fatura", "plano"]

templates = {
    "account_management": [
        "Como altero o titular da conta do {product}?",
        "Preciso adicionar um novo usuário ao {product}.",
        "Como configuro permissões administrativas no {product}?",
        "Quero atualizar os dados cadastrais da minha conta.",
        "Como transfiro serviços do {product} para outra conta?"
    ],
    "billing": [
        "Verifiquei cobrança incorreta na fatura do {product}.",
        "Fui cobrado a mais referente ao {product}, preciso de ajuste.",
        "Quero entender lançamentos na minha fatura do {product}.",
        "Solicito estorno de uma cobrança do {product}.",
        "Há um débito indevido relacionado ao {product} na minha fatura."
    ],
    "cancellation": [
        "Quero cancelar meu plano do {product} imediatamente.",
        "Como faço para encerrar o serviço de {product}?",
        "Solicito cancelamento do {product} por insatisfação.",
        "Preciso confirmar o cancelamento do {product}.",
        "Quero cancelar e saber sobre possíveis multas do {product}."
    ],
    "technical_issue": [
        "O {product} está fora do ar desde ontem.",
        "Estou com erro ao acessar o {product}.",
        "O {product} apresenta alta latência e falhas constantes.",
        "Não consigo fazer login no {product}, aparece erro.",
        "Perda de conexão recorrente no {product}."
    ],
    "complaint": [
        "Não estou satisfeito com o atendimento recebido.",
        "Quero registrar uma reclamação formal sobre o {product}.",
        "Problema recorrente não foi resolvido pelo suporte.",
        "Atendimento ineficiente sobre o {product}.",
        "Insatisfeito com a qualidade do {product}."
    ]
}

def _remove_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def _preprocess(text: str) -> str:
    """
    Normaliza texto para 'preprocessed_text':
    - lower
    - remove acentos
    - remove pontuação (mantém letras, dígitos e espaços)
    - colapsa espaços
    """
    if text is None:
        return ""
    t = str(text).lower().strip()
    t = _remove_accents(t)
    # keep alnum and spaces
    t = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in t)
    t = " ".join(t.split())
    return t

def load_existing(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

def count_pairs(records):
    cnt = {}
    for r in records:
        cat = r.get("category")
        pr = r.get("priority")
        if cat is None or pr is None:
            continue
        cnt[(cat, pr)] = cnt.get((cat, pr), 0) + 1
    return cnt

def make_text(template, product):
    text = template.format(product=product)
    # small random variations
    if random.random() < 0.15:
        text += " por favor"
    if random.random() < 0.12:
        text = "Por favor, " + text
    if random.random() < 0.08:
        text += " preciso urgente"
    return text

def generate(records):
    # ensure every record has 'preprocessed_text' derived from 'text'
    for r in records:
        if "preprocessed_text" not in r or not r.get("preprocessed_text"):
            r["preprocessed_text"] = _preprocess(r.get("text", ""))

    existing_texts = { (r.get("text") or "").strip().lower() for r in records }
    counts = count_pairs(records)

    for (cat, pr), target in TARGETS.items():
        current = counts.get((cat, pr), 0)
        tries = 0
        while current < target and tries < target * 10:
            tpl = random.choice(templates.get(cat, [""])) 
            product = random.choice(products)
            text = make_text(tpl, product)
            key = text.strip().lower()
            # avoid exact duplicates
            if key in existing_texts:
                tries += 1
                continue
            rec = {
                "text": text,
                "preprocessed_text": _preprocess(text),
                "category": cat,
                "priority": pr
            }
            records.append(rec)
            existing_texts.add(key)
            current += 1
            tries = 0
        # update counts
        counts[(cat, pr)] = current

    return records

def main():
    existing = load_existing(IN_KB)
    print(f"Loaded {len(existing)} existing records from {IN_KB.name}")
    out = list(existing)  # copy
    out = generate(out)
    # garantir ordem das chaves: text, preprocessed_text, category, priority
    ordered_keys = ["text", "preprocessed_text", "category", "priority"]
    formatted = []
    for r in out:
        # recria o dict na ordem desejada (mantém apenas essas chaves)
        ordered = {k: r.get(k) for k in ordered_keys if k in r}
        formatted.append(ordered)

    OUT_KB.parent.mkdir(parents=True, exist_ok=True)
    with OUT_KB.open("w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)
    final_counts = count_pairs(out)
    print(f"Wrote {len(out)} records to {OUT_KB.name}")
    for k, v in sorted(final_counts.items()):
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()