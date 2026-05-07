#!/bin/bash
# First-time setup for the FunctionGemma physical-AI demo on the Coral Dev Board.
#
# Idempotent: re-run is safe. Skips work that's already done. Self-contained:
# every dep, wheel, library, and model lives under Function_calling/.
#
# Steps:
#   1. Create the venv at Function_calling/.venv (--system-site-packages so the
#      OOBE image's PyQt5/numpy/etc. carry through).
#   2. Install requirements.txt from PyPI (or --offline, from wheels/).
#   3. Download the v7 GGUF from HuggingFace into models/.
#   4. With --voice: install torq_runtime wheel (--no-deps), extract
#      library/portaudio_libs.tgz into /, and download the four Moonshine
#      VMFB artifacts from HuggingFace into models/moonshine/.
#
# Usage (from anywhere):
#   bash Function_calling/scripts/setup.sh                  # online install, no voice
#   bash Function_calling/scripts/setup.sh --offline        # use Function_calling/wheels/
#   bash Function_calling/scripts/setup.sh --voice          # online + voice + Moonshine
#   bash Function_calling/scripts/setup.sh --offline --voice

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# requirements.txt references wheels/...whl as a relative path, which pip
# resolves against cwd (not against the requirements file). Anchor cwd to
# Function_calling/ so the script is invokable from any directory.
cd "${FC_DIR}"

VENV_DIR="${FC_DIR}/.venv"
MODELS_DIR="${FC_DIR}/models"
MOONSHINE_DIR="${MODELS_DIR}/moonshine"
WHEELS_DIR="${FC_DIR}/wheels"
LIBRARY_DIR="${FC_DIR}/library"
REQS="${FC_DIR}/requirements.txt"

GGUF_FILENAME="functiongemma-physical-ai-v7-Q5_K_M.gguf"
HF_BASE="https://huggingface.co/BrinqAI/functiongemma-270m-physical-ai/resolve/main"
GGUF_URL="${HF_BASE}/${GGUF_FILENAME}"
MOONSHINE_FILES=(encoder.onnx decoder.vmfb decoder_with_past.vmfb decoder_token_embeddings.npy)

TORQ_WHEEL="${WHEELS_DIR}/torq_runtime-1.5.0-cp312-cp312-manylinux_2_28_aarch64.whl"
PORTAUDIO_TGZ="${LIBRARY_DIR}/portaudio_libs.tgz"

OFFLINE=0
VOICE=0
for arg in "$@"; do
    case "${arg}" in
        --offline) OFFLINE=1 ;;
        --voice)   VOICE=1 ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            echo "unknown arg: ${arg}" >&2
            exit 2
            ;;
    esac
done

log() { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
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
    log "creating venv at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}" --system-site-packages
else
    log "venv exists at ${VENV_DIR} (skip)"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python3 -m pip install --upgrade pip >/dev/null

# --- 2. requirements ---------------------------------------------------------
PIP_ARGS=()
if [ "${OFFLINE}" -eq 1 ]; then
    log "installing from wheels/ (offline)"
    PIP_ARGS+=(--no-index --find-links "${WHEELS_DIR}")
else
    log "installing from PyPI (online; wheels/ used as a local fallback)"
    PIP_ARGS+=(--find-links "${WHEELS_DIR}")
fi
pip install "${PIP_ARGS[@]}" -r "${REQS}"

# --- 3. GGUF model -----------------------------------------------------------
mkdir -p "${MODELS_DIR}"
if [ -f "${MODELS_DIR}/${GGUF_FILENAME}" ]; then
    log "GGUF model already present at ${MODELS_DIR}/${GGUF_FILENAME} (skip)"
else
    log "downloading ${GGUF_FILENAME} from HuggingFace (~248 MB)"
    download "${GGUF_URL}" "${MODELS_DIR}/${GGUF_FILENAME}"
fi

# --- 4. voice (optional) -----------------------------------------------------
if [ "${VOICE}" -eq 1 ]; then
    if [ -f "${TORQ_WHEEL}" ]; then
        log "installing torq_runtime wheel (--no-deps per upstream guidance)"
        pip install --no-deps "${TORQ_WHEEL}"
    else
        warn "torq_runtime wheel missing at ${TORQ_WHEEL}; voice will fall back to CPU"
    fi

    if [ -f "${PORTAUDIO_TGZ}" ]; then
        log "extracting portaudio libs into / (needs sudo or root)"
        if [ "$(id -u)" -eq 0 ]; then
            tar -xzf "${PORTAUDIO_TGZ}" -C /
        else
            sudo tar -xzf "${PORTAUDIO_TGZ}" -C /
        fi
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
            download "${HF_BASE}/moonshine/${f}" "${MOONSHINE_DIR}/${f}"
        fi
    done
fi

log "done. activate the venv and run the demo:"
log "  source ${VENV_DIR}/bin/activate"
log "  cd ${FC_DIR} && python3 demo.py"
