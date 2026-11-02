"""Print web/metrics.json from the saved eval results (only files that exist)."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
ROWS = [
    ("heldout_rules.json", "Held-out (independent author) · rules only"),
    ("heldout_rules_ner.json", "Held-out · rules + local LLM-NER (qwen2.5 7B)"),
    ("dev_rules.json", "Dev set (used for tuning) · rules only"),
]
out = []
for fname, label in ROWS:
    p = R / fname
    if not p.exists():
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    out.append({"label": label, "notes": d["notes"], "gold": d["gold_entities"], "recall": d["protection_recall"],
                "precision": d["precision"], "leak_free": d["leak_free_notes"], "source": f"eval/results/{fname}"})
print(json.dumps({"rows": out}, ensure_ascii=False, indent=2))
