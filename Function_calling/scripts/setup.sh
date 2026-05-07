#!/bin/bash
# First-time setup for the FunctionGemma physical-AI demo on the Coral Dev Board.
#
# Idempotent: re-run is safe. Skips work that's already done.
#
# Steps:
#   1. Create the venv at <repo_root>/.venv (--system-site-packages so the
#      OOBE image's PyQt5/numpy/etc. carry through).
#   2. Install requirements.txt from PyPI (or --offline, from wheelhouse/).
#   3. Download the v7 GGUF from HuggingFace into models/.
#   4. With --voice: install voice deps (sounddevice + silero-vad-notorch +
#      torq_runtime wheel) and extract library/portaudio_libs.tgz.
#
# Usage:
#   bash scripts/setup.sh                  # online install, no voice
#   bash scripts/setup.sh --offline        # use wheelhouse/, no voice
#   bash scripts/setup.sh --voice          # online + voice deps
#   bash scripts/setup.sh --offline --voice
#
# Run from anywhere — paths are resolved relative to the script location.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${FC_DIR}/.." && pwd)"

# requirements.txt references wheels under ./wheelhouse/, which pip resolves
# against cwd (not against the requirements file). Run everything from the
# repo root so the script is invokable from anywhere.
cd "${REPO_ROOT}"

VENV_DIR="${REPO_ROOT}/.venv"
MODELS_DIR="${REPO_ROOT}/models"
WHEELHOUSE="${REPO_ROOT}/wheelhouse"
LIBRARY_DIR="${REPO_ROOT}/library"
REQS="${REPO_ROOT}/requirements.txt"
VOICE_REQS="${REPO_ROOT}/speech_to_text/requirements.txt"

MODEL_FILENAME="functiongemma-physical-ai-v7-Q5_K_M.gguf"
MODEL_URL="https://huggingface.co/BrinqAI/functiongemma-270m-physical-ai/resolve/main/${MODEL_FILENAME}"
TORQ_WHEEL="${WHEELHOUSE}/torq_runtime-1.5.0-cp312-cp312-manylinux_2_28_aarch64.whl"
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
    log "installing from wheelhouse (offline)"
    PIP_ARGS+=(--no-index --find-links "${WHEELHOUSE}")
else
    log "installing from PyPI (online; wheelhouse used as a local fallback)"
    PIP_ARGS+=(--find-links "${WHEELHOUSE}")
fi
pip install "${PIP_ARGS[@]}" -r "${REQS}"

# --- 3. model ----------------------------------------------------------------
mkdir -p "${MODELS_DIR}"
if [ -f "${MODELS_DIR}/${MODEL_FILENAME}" ]; then
    log "model already present at ${MODELS_DIR}/${MODEL_FILENAME} (skip)"
else
    log "downloading ${MODEL_FILENAME} from HuggingFace (~248 MB)"
    if command -v wget >/dev/null 2>&1; then
        wget -O "${MODELS_DIR}/${MODEL_FILENAME}" "${MODEL_URL}"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -o "${MODELS_DIR}/${MODEL_FILENAME}" "${MODEL_URL}"
    else
        warn "no wget or curl available — fetch the model manually:"
        warn "  ${MODEL_URL}"
        warn "  -> ${MODELS_DIR}/${MODEL_FILENAME}"
    fi
fi

# --- 4. voice (optional) -----------------------------------------------------
if [ "${VOICE}" -eq 1 ]; then
    log "installing voice deps"
    if [ -f "${VOICE_REQS}" ]; then
        pip install "${PIP_ARGS[@]}" -r "${VOICE_REQS}"
    else
        warn "voice requirements not found at ${VOICE_REQS}; skipping pip install"
    fi
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
fi

log "done. activate the venv and run the demo:"
log "  source ${VENV_DIR}/bin/activate"
log "  cd ${FC_DIR} && python3 demo.py"
