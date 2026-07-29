#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

uv run scripts/convert_model.py \
  --checkpoint models/checkpoints/mindsurf-pretrain-80m-seed20260511/final_model.pt \
  --output-dir models/transformers/mindsurf-pretrain-80m-qwen3 \
  --tokenizer models/tokenizers/minimind_tokenizer \
  --target auto
