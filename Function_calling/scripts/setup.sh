#!/bin/bash
# First-time setup for the FunctionGemma physical-AI demo on the Coral Dev Board.
#
# Idempotent: re-run is safe. Skips work that's already done.
#
# Steps:
#   1. Activate the shared venv at <repo>/.venv (one directory above Function_calling/).
#      The venv must already exist — this script will not create it.
#   2. Install requirements.txt from PyPI.
#   3. Download the FunctionGemma GGUF from HuggingFace into models/.
#   4. Install torq_runtime wheel (--no-deps), extract portaudio_libs.tgz into /,
#      and download Moonshine artifacts from HuggingFace into
#      models/Synaptics/moonshine-tiny-bf16-torq/.
#
# Usage (from anywhere):
#   bash Function_calling/scripts/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_DIR="$(cd "${FC_DIR}/.." && pwd)"

# requirements.txt references wheelhouse/...whl as a relative path, which pip
# resolves against cwd (not against the requirements file). Anchor cwd to
# Function_calling/ so the script is invokable from any directory.
cd "${FC_DIR}"

VENV_DIR="${PARENT_DIR}/.venv"
MODELS_DIR="${FC_DIR}/../models"
MOONSHINE_DIR="${MODELS_DIR}/Synaptics/moonshine-tiny-bf16-torq"
WHEELS_DIR="${FC_DIR}/../wheelhouse"
LIBRARY_DIR="${FC_DIR}/../library"
REQS="${FC_DIR}/requirements.txt"

GGUF_FILENAME="functiongemma-physical-ai-v9-Q5_K_M.gguf"
HF_BASE="https://huggingface.co/BrinqAI/functiongemma-270m-physical-ai/resolve/main"
GGUF_URL="${HF_BASE}/${GGUF_FILENAME}"
MOONSHINE_HF_BASE="https://huggingface.co/Synaptics/moonshine-tiny-bf16-torq/resolve/main"
MOONSHINE_FILES=(encoder.vmfb decoder.vmfb decoder_with_past.vmfb decoder_token_embeddings.npy tokenizer.json preprocessor.onnx)

TORQ_WHEEL="${WHEELS_DIR}/torq_runtime-2.0.0a1-cp312-cp312-manylinux_2_28_aarch64.whl"
PORTAUDIO_TGZ="${LIBRARY_DIR}/portaudio_libs.tgz"

log()  { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup]\033[0m %s\n' "$*" >&2; }

# --- helpers -----------------------------------------------------------------
download() {
    # download URL DEST  — uses wget if available, falls back to curl
    local url="$1" dest="$2"
    if command -v wget >/dev/null 2>&1; then
        wget -O "${dest}" "${url}"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -o "${dest}" "${url}"
    else
        warn "no wget or curl — fetch manually:"
        warn "  ${url} -> ${dest}"
        return 1
    fi
}

# --- 1. venv -----------------------------------------------------------------
if [ ! -d "${VENV_DIR}" ]; then
    echo "shared venv not found at ${VENV_DIR}" >&2
    echo "create it one directory up before running this script" >&2
    exit 1
fi
log "activating shared venv at ${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python3 -m pip install --upgrade pip >/dev/null

# --- 2. requirements ---------------------------------------------------------
log "installing from PyPI"
pip install -r "${REQS}"

# --- 3. GGUF model -----------------------------------------------------------
mkdir -p "${MODELS_DIR}"
if [ -f "${MODELS_DIR}/${GGUF_FILENAME}" ]; then
    log "GGUF model already present at ${MODELS_DIR}/${GGUF_FILENAME} (skip)"
else
    log "downloading ${GGUF_FILENAME} from HuggingFace (~248 MB)"
    download "${GGUF_URL}" "${MODELS_DIR}/${GGUF_FILENAME}"
fi

# --- 4. voice ----------------------------------------------------------------
if [ -f "${TORQ_WHEEL}" ]; then
    log "installing torq_runtime wheel (--no-deps per upstream guidance)"
    pip install --no-deps "${TORQ_WHEEL}"
else
    warn "torq_runtime wheel missing at ${TORQ_WHEEL}; voice will fall back to CPU"
fi

if [ -f "${PORTAUDIO_TGZ}" ]; then
    log "extracting portaudio libs into / (needs root)"
    tar -xzf "${PORTAUDIO_TGZ}" -C /
else
    warn "portaudio_libs.tgz missing at ${PORTAUDIO_TGZ}; mic stream will fail unless libportaudio.so.2 is installed system-wide"
fi

log "downloading Moonshine artifacts from HuggingFace into ${MOONSHINE_DIR}"
mkdir -p "${MOONSHINE_DIR}"
for f in "${MOONSHINE_FILES[@]}"; do
    if [ -f "${MOONSHINE_DIR}/${f}" ]; then
        log "  ${f} already present (skip)"
    else
        log "  fetching ${f}"
        download "${MOONSHINE_HF_BASE}/${f}" "${MOONSHINE_DIR}/${f}"
    fi
done

log "done. activate the shared venv and run the demo:"
log "  source ${VENV_DIR}/bin/activate"
log "  cd ${FC_DIR} && python3 demo.py"
