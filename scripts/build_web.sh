#!/usr/bin/env bash
# Assemble web/ for Cloudflare Pages: copy the browser-safe Python modules and metrics.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/web/py/gopniya"
for f in __init__.py validators.py detect.py vault.py guard.py agent.py; do
  cp "$ROOT/gopniya/$f" "$ROOT/web/py/gopniya/$f"
done
python3 "$ROOT/eval/make_metrics.py" > "$ROOT/web/metrics.json"
echo "web/ ready"
