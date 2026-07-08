#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${AUDEX_MODEL_ROOT:-/models/audex}"
REPO_ID="${AUDEX_REPO_ID:-nvidia/Nemotron-Labs-Audex-30B-A3B}"
PORT="${AUDEX_VLLM_PORT:-9999}"

python3 - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    "${REPO_ID}",
    local_dir="${MODEL_ROOT}",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "checkpoint_folder_full/**",
        "inference_scripts_vllm/audioqa_scripts/**",
    ],
)
PY

pip install -e "${MODEL_ROOT}/inference_scripts_vllm/audioqa_scripts" \
    --no-deps \
    --no-build-isolation

exec bash "${MODEL_ROOT}/inference_scripts_vllm/audioqa_scripts/serve_audioqa_vllm.sh" \
    "${MODEL_ROOT}/checkpoint_folder_full" \
    "${PORT}"
