"""LLM backends. Standard library only (urllib + subprocess).

Cloud backends receive redacted text only; the agent enforces that. The NER backend sees
raw text and therefore must be local: ``ollama_ner`` talks to an Ollama server on
localhost by default.

- ``http_endpoint(url)``: our deployed demo endpoint (Cloudflare Pages Function in front
  of Workers AI). No key on the client side.
- ``workers_ai(account_id, token, model)``: Cloudflare Workers AI REST API directly.
- ``ollama_chat(model)``: a local model as the answering LLM (fully offline mode).
- ``gemini_cli()``: the ``gemini`` command-line tool, if installed and logged in.
- ``ollama_ner(model)``: local LLM-NER pass for names and addresses the rules miss.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import urllib.error
import urllib.request

from .detect import Span

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")


def _post(url: str, payload: dict, headers: dict | None = None, timeout: float = 120) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "gopniya-cli/0.1", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed URLs
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:  # surface the server's JSON error message
        try:
            return json.loads(exc.read().decode())
        except ValueError:
            raise exc from None


def http_endpoint(url: str):
    async def call(messages: list[dict]) -> str:
        data = await asyncio.to_thread(_post, url, {"messages": messages})
        if "error" in data:
            raise RuntimeError(data["error"])
        return data["answer"]
    return call


DEFAULT_CLOUD_MODEL = "@cf/google/gemma-4-26b-a4b-it"


def _workers_ai_text(result: dict) -> str:
    if "response" in result:
        return result["response"]
    return result["choices"][0]["message"]["content"]


def workers_ai(account_id: str, token: str, model: str = DEFAULT_CLOUD_MODEL):
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"

    async def call(messages: list[dict]) -> str:
        data = await asyncio.to_thread(
            _post, url,
            {"messages": messages, "max_tokens": 900, "temperature": 0.2,
             "chat_template_kwargs": {"enable_thinking": False}},
            {"Authorization": f"Bearer {token}"},
        )
        return _workers_ai_text(data["result"])
    return call


def ollama_chat(model: str = "qwen2.5:7b-instruct"):
    async def call(messages: list[dict]) -> str:
        data = await asyncio.to_thread(
            _post, f"{OLLAMA_URL}/api/chat",
            {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0.2}},
        )
        return data["message"]["content"]
    return call


def gemini_cli(binary: str = "gemini"):
    async def call(messages: list[dict]) -> str:
        prompt = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)
        proc = await asyncio.to_thread(
            subprocess.run, [binary, "-p", prompt], capture_output=True, text=True, timeout=180
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip()[:300])
        return proc.stdout.strip()
    return call


NER_PROMPT = (
    "You find personal information in Indian clinical notes (English, Hindi, or mixed). "
    "Return ONLY JSON: {\"entities\": [{\"type\": \"PERSON\"|\"ADDRESS\", \"text\": \"exact substring\"}]}. "
    "PERSON = names of real people (patient, relatives, doctors, nurses, ASHA/ANM workers), without titles "
    "like Mr., Smt., Dr., श्री, श्रीमती. ADDRESS = a person's home address (house, street, village, tehsil, "
    "district, state, PIN). Do NOT include hospital or clinic names, medicine names, diseases, or job titles. "
    "Copy each text exactly as it appears. If nothing is found return {\"entities\": []}."
)


def _anchor(text: str, items: list[dict]) -> list[Span]:
    spans: list[Span] = []
    for it in items:
        val, typ = it.get("text"), str(it.get("type", "")).upper()
        if not isinstance(val, str) or len(val.strip()) < 2 or typ not in {"PERSON", "ADDRESS"}:
            continue
        val = val.strip()
        for m in re.finditer(re.escape(val), text):
            a, b = m.start(), m.end()
            # whole-word only
            if (a > 0 and text[a - 1].isalnum()) or (b < len(text) and text[b].isalnum()):
                continue
            spans.append(Span(a, b, typ, val, 0.65, "local_ner"))
    return spans


def ollama_ner(model: str = "qwen2.5:7b-instruct"):
    async def ner(text: str) -> list[Span]:
        data = await asyncio.to_thread(
            _post, f"{OLLAMA_URL}/api/chat",
            {
                "model": model, "stream": False, "format": "json",
                "options": {"temperature": 0, "num_ctx": 8192},
                "messages": [{"role": "system", "content": NER_PROMPT}, {"role": "user", "content": text[:12000]}],
            },
            None, 300,
        )
        try:
            items = json.loads(data["message"]["content"]).get("entities", [])
        except (ValueError, KeyError, AttributeError):
            return []
        return _anchor(text, [i for i in items if isinstance(i, dict)])
    return ner
