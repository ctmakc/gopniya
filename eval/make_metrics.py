"""Print web/metrics.json from the saved eval results (only files that exist)."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
ROWS = [
    ("blind2_rules.json", "Blind set 2 (unseen) · rules v2"),
    ("blind2_rules_ner.json", "Blind set 2 (unseen) · rules v2 + local LLM-NER"),
    ("heldout_rules_v1_frozen.json", "Held-out set 1 (unseen at the time) · rules v1"),
    ("heldout_rules.json", "Held-out set 1 after error analysis · rules v2"),
    ("heldout_rules_ner.json", "Held-out set 1 · rules v2 + local LLM-NER"),
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
