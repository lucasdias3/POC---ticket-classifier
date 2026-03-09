import json
import random
import re
import math
from pathlib import Path
from collections import defaultdict

random.seed(42)

BASE = Path(__file__).resolve().parent.parent
KB_PATH = BASE / "data" / "knowledge_base.json"
OUT_PATH = BASE / "data" / "testset_kgb.json"

def normalize_text(t: str) -> str:
    t = (t or "").lower()
    # remove punctuation (keep unicode letters and digits and spaces)
    t = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE)
    # collapse spaces
    t = re.sub(r"\s+", " ", t).strip()
    return t

def main():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    if total == 0:
        raise SystemExit("knowledge_base.json vazio")

    # half of total (arredondando para cima para garantir pelo menos metade)
    n = math.ceil(total / 2)

    # amostra aleatória sem reposição (reprodutível pelo seed acima)
    selected = random.sample(data, n)

    # garantir compatibilidade: usar preprocessed_text se existir, caso contrário normalizar 'text'
    out = []
    for item in selected:
        it = dict(item)  # copy
        if "preprocessed_text" in it and it.get("preprocessed_text"):
            it["text"] = it["preprocessed_text"]
        else:
            it["text"] = normalize_text(it.get("text", ""))

        out.append(it)

    # write pretty json, one object per block (indent=2)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"wrote {len(out)} examples (half of {total}) to {OUT_PATH}")

if __name__ == "__main__":
    main()