#!/usr/bin/env bash
# Record the captioned demo from the live site and encode it as H.264 MP4.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
URL="${1:-https://gopniya.pages.dev/}"
RAW=$(python3 "$ROOT/scripts/record_demo.py" "$URL" "$ROOT/video" | tail -1)
FF=$(command -v ffmpeg || ls ~/.cache/ms-playwright/ffmpeg-*/ffmpeg-linux | head -1)
"$FF" -y -loglevel error -i "$RAW" -c:v libx264 -preset slow -crf 24 -pix_fmt yuv420p -movflags +faststart -an "$ROOT/video/gopniya-demo.mp4"
echo "$ROOT/video/gopniya-demo.mp4"
