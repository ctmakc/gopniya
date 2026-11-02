"""Command line.

    python -m gopniya redact note.txt                 # print the redacted note, nothing leaves the machine
    python -m gopniya run note.txt --task summary_hi --llm endpoint:https://gopniya.pages.dev/api/llm
    python -m gopniya run note.txt --llm ollama:qwen2.5:7b-instruct --ner ollama:qwen2.5:7b-instruct   # fully offline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from urllib.parse import urlparse

from . import llm as backends
from .agent import TASKS, RedactionAgent


def _llm(spec: str | None):
    if not spec:
        return None
    kind, _, arg = spec.partition(":")
    if kind == "endpoint":
        return backends.http_endpoint(arg)
    if kind == "ollama":
        return backends.ollama_chat(arg or "qwen2.5:7b-instruct")
    if kind == "gemini":
        return backends.gemini_cli()
    if kind == "workers-ai":
        return backends.workers_ai(os.environ["CF_ACCOUNT_ID"], os.environ["CF_API_TOKEN"], arg or backends.DEFAULT_CLOUD_MODEL)
    raise SystemExit(f"unknown --llm {spec}")


def _ner(spec: str | None):
    if not spec:
        return None, True
    kind, _, model = spec.partition(":")
    if kind != "ollama":
        raise SystemExit("--ner supports ollama:<model> only (it sees raw text, so it must be local)")
    host = urlparse(backends.OLLAMA_URL).hostname or ""
    return backends.ollama_ner(model or "qwen2.5:7b-instruct"), host in {"127.0.0.1", "localhost", "::1"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gopniya")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("redact", "run"):
        p = sub.add_parser(name)
        p.add_argument("file", help="path to a note, or - for stdin")
        p.add_argument("--ner", default=None, help="ollama:<model> local NER pass")
        p.add_argument("--json", action="store_true")
        if name == "run":
            p.add_argument("--task", default="summary_en", help="one of: " + ", ".join(TASKS) + ", or free text")
            p.add_argument("--llm", required=True, help="endpoint:<url> | ollama:<model> | gemini | workers-ai[:model]")
    a = ap.parse_args(argv)
    text = sys.stdin.read() if a.file == "-" else open(a.file, encoding="utf-8").read()
    ner, local = _ner(a.ner)
    agent = RedactionAgent(llm=_llm(getattr(a, "llm", None)), ner=ner, ner_is_local=local)
    res = asyncio.run(agent.run(text, getattr(a, "task", "summary_en")))
    if a.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return 0 if res.status == "ok" else 2
    for s in res.steps:
        print(f"[{s.name:8}] {s.detail}", file=sys.stderr)
    if a.cmd == "redact" or res.status != "ok":
        print(res.redacted)
    if a.cmd == "run" and res.status == "ok":
        print("\n--- sent to the cloud model ---\n" + res.redacted, file=sys.stderr)
        print("\n--- model answer (placeholders) ---\n" + res.answer_redacted, file=sys.stderr)
        print(res.answer)
    return 0 if res.status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
