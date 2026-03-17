#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

DEVICE_ID="${1:-${DEVICE_ID:-demo-usbserial-six-axis}}"
SCRIPT_PATH="${DEMO_SCRIPT:-}"
DEVICE_CONFIG="${DEMO_DEVICE_CONFIG:-data/demo/devices.json}"
CATALOG_PATH="${DEMO_CATALOG:-data/demo/catalog_en.tsv}"
AUDIO_DIR="${DEMO_AUDIO_DIR:-data/demo/backchannel_en}"
PORT_ARG="${DEMO_PORT:-}"

if [ ! -f "$DEVICE_CONFIG" ]; then
  echo "デバイス設定が見つかりません: $DEVICE_CONFIG" >&2
  exit 1
fi

if [ ! -f "$CATALOG_PATH" ]; then
  echo "カタログが見つかりません: $CATALOG_PATH" >&2
  exit 1
fi

if [ ! -d "$AUDIO_DIR" ]; then
  echo "英語 backchannel 音声ディレクトリが見つかりません: $AUDIO_DIR" >&2
  exit 1
fi

if [ -n "$SCRIPT_PATH" ] && [ ! -f "$SCRIPT_PATH" ]; then
  echo "台本が見つかりません: $SCRIPT_PATH" >&2
  exit 1
fi

if [ -n "$SCRIPT_PATH" ] && [ -z "${OPENAI_API_KEY:-}" ] && [ ! -f ".env" ]; then
  echo "OPENAI_API_KEY がありません。.env か環境変数で設定してください。" >&2
  exit 1
fi

if [ "${SKIP_SYNC:-0}" != "1" ]; then
  uv sync
fi

CMD=(
  uv run python app/cli/demo.py
  --device-id "$DEVICE_ID"
  --device-config "$DEVICE_CONFIG"
  --catalog "$CATALOG_PATH"
  --audio-dir "$AUDIO_DIR"
)

if [ -n "$SCRIPT_PATH" ]; then
  CMD+=(--script "$SCRIPT_PATH")
fi

if [ -n "$PORT_ARG" ]; then
  CMD+=(--port "$PORT_ARG")
fi

echo "起動: device_id=$DEVICE_ID mode=$([ -n "$SCRIPT_PATH" ] && echo script || echo sensor-only)"
"${CMD[@]}"
