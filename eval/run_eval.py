"""Measure detection quality on a JSONL set with gold spans.

Metrics (character-level, whitespace and punctuation ignored):

- protection recall: share of gold entities whose every significant character ends up
  inside some redacted span (of any type). This is the privacy number: an entity counts
  as protected only when nothing of it leaks.
- partial: gold entities with some, but not all, characters covered (a partial leak).
- precision: share of predicted spans that overlap at least one gold entity. A span that
  overlaps nothing is a false alarm (over-redaction, which costs answer quality).
- note-level leak-free rate: share of notes where every gold entity is fully protected.

Usage:
    python eval/run_eval.py data/heldout.jsonl [--ner ollama:qwen2.5:7b-instruct] [--out eval/results/x.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gopniya.agent import RedactionAgent  # noqa: E402

_SIG = lambda ch: not (ch.isspace() or ch in ",.;:-()/")  # noqa: E731


def evaluate(rows: list[dict], agent: RedactionAgent) -> dict:
    per_type = defaultdict(lambda: {"gold": 0, "protected": 0, "partial": 0})
    pred_total = pred_tp = 0
    fp_by_type: dict[str, int] = defaultdict(int)
    leak_free = 0
    leaks: list[dict] = []
    false_alarms: list[dict] = []
    t0 = time.perf_counter()
    for row in rows:
        text = row["text"]
        _, spans, _, _ = asyncio.run(agent.deidentify(text))
        covered = [False] * len(text)
        for s in spans:
            for i in range(s.start, s.end):
                covered[i] = True
        note_ok = True
        for g in row["entities"]:
            idx = [i for i in range(g["start"], g["end"]) if _SIG(text[i])]
            if not idx:
                continue
            hit = sum(covered[i] for i in idx)
            st = per_type[g["type"]]
            st["gold"] += 1
            if hit == len(idx):
                st["protected"] += 1
            else:
                note_ok = False
                if hit:
                    st["partial"] += 1
                leaks.append({"id": row["id"], "type": g["type"], "text": g["text"], "covered": f"{hit}/{len(idx)}"})
        leak_free += note_ok
        for s in spans:
            pred_total += 1
            if any(s.start < g["end"] and g["start"] < s.end for g in row["entities"]):
                pred_tp += 1
            else:
                fp_by_type[s.type] += 1
                false_alarms.append({"id": row["id"], "type": s.type, "text": text[s.start:s.end]})
    elapsed = time.perf_counter() - t0
    gold = sum(v["gold"] for v in per_type.values())
    prot = sum(v["protected"] for v in per_type.values())
    return {
        "notes": len(rows),
        "gold_entities": gold,
        "protection_recall": round(prot / gold, 4) if gold else None,
        "precision": round(pred_tp / pred_total, 4) if pred_total else None,
        "predicted_spans": pred_total,
        "false_alarms": pred_total - pred_tp,
        "leak_free_notes": round(leak_free / len(rows), 4) if rows else None,
        "seconds_total": round(elapsed, 2),
        "ms_per_note": round(elapsed * 1000 / max(1, len(rows)), 2),
        "per_type": {
            k: {**v, "recall": round(v["protected"] / v["gold"], 4) if v["gold"] else None}
            for k, v in sorted(per_type.items())
        },
        "false_alarms_by_type": dict(sorted(fp_by_type.items())),
        "leaks": leaks,
        "false_alarm_examples": false_alarms[:200],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--ner", default=None, help="ollama:<model> for the optional local NER pass")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    rows = [json.loads(line) for line in Path(a.path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if a.lang:
        rows = [r for r in rows if r["lang"] == a.lang]
    if a.limit:
        rows = rows[: a.limit]
    ner = None
    if a.ner:
        from gopniya.llm import ollama_ner
        ner = ollama_ner(a.ner.split(":", 1)[1])
    res = evaluate(rows, RedactionAgent(llm=None, ner=ner))
    res["config"] = {"set": a.path, "ner": a.ner, "lang": a.lang}
    summary = {k: v for k, v in res.items() if k not in ("leaks", "false_alarm_examples")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
