#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# start_gpu_ollama.sh - start an OLLAMA server that USES THE GPU.
#
# WHY THIS EXISTS (measured 2026-11-11):
#
#   The machine has an Intel Arc A770. Ollama's Vulkan runner IS installed
#   (/usr/local/lib/ollama/vulkan/libggml-vulkan.so) and the driver exposes
#   the device. But the LONG-RUNNING systemd server (port 11434) -- the one
#   the engine talks to -- discovered NO GPU at ITS startup and has been
#   running the model on the CPU ever since.
#
#   MEASURED (the same prompt, the same model):
#     the CPU server (port 11434):  101.9 s / 94.6 s wall,  93.8 s eval
#     a GPU server  (port 11500):     9.6 s /  5.0 s wall,   4.69 s eval
#   -- about TWENTY TIMES faster, and the engine's llm_probe answers in
#   0.65 s against it.
#
#   The cause is NOT a missing driver, a missing runner, or a permissions
#   problem -- all of those were checked and are fine. The cause is that the
#   service's STARTUP DISCOVERY found only `id=cpu`:
#
#     Nov 06 13:21:38  msg="inference compute" id=cpu library=cpu ...
#
#   and a FRESH process started since then finds the A770. So the fix is to
#   RESTART the service (`sudo systemctl restart ollama`) -- which needs root
#   -- or to run this script, which needs none.
#
# USAGE:
#   ./start_gpu_ollama.sh [PORT]        # default 11500
#   # then point the engine at it:
#   ./build_delirium/delirium_workspace --host 127.0.0.1 --port 11500 ...
#
# WHY IT SETS THE VARIABLES EXPLICITLY:
#   OLLAMA_MODELS -- the systemd service's models live under the SERVICE
#     user's home (/usr/share/ollama/.ollama/models), NOT under the invoking
#     user's (~/.ollama/models, which is empty). Without this the server
#     starts and every model is "not found".
#   OLLAMA_VULKAN  -- makes the Vulkan runner a candidate at all.
# ---------------------------------------------------------------------------
set -u

PORT="${1:-11500}"
MODELS="${OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}"
LOG="/tmp/ollama_gpu_${PORT}.log"

if [ ! -d "$MODELS" ]; then
    echo "start_gpu_ollama: the models dir '$MODELS' does not exist." >&2
    echo "  The systemd service uses /usr/share/ollama/.ollama/models." >&2
    exit 1
fi
if [ ! -f /usr/local/lib/ollama/vulkan/libggml-vulkan.so ]; then
    echo "start_gpu_ollama: the Vulkan runner is not installed." >&2
    echo "  Without it Ollama falls back to the CPU." >&2
    exit 1
fi

# Is a GPU-backed server already up on this port?
if curl -s --max-time 5 "http://127.0.0.1:${PORT}/api/version" >/dev/null 2>&1; then
    echo "start_gpu_ollama: something is already serving on ${PORT}."
    exit 0
fi

echo "start_gpu_ollama: starting on port ${PORT}"
echo "  models: ${MODELS}"
echo "  log:    ${LOG}"

OLLAMA_MODELS="$MODELS" \
OLLAMA_VULKAN=true \
OLLAMA_HOST="127.0.0.1:${PORT}" \
    nohup ollama serve > "$LOG" 2>&1 &

# Wait for it to come up, and PROVE it found the GPU.
for i in $(seq 1 40); do
    sleep 1
    if curl -s --max-time 3 "http://127.0.0.1:${PORT}/api/version" >/dev/null 2>&1; then
        if grep -q 'library=Vulkan' "$LOG" 2>/dev/null; then
            echo "  READY: the Vulkan device was discovered."
            grep -m1 'inference compute' "$LOG" | sed 's/^/  /'
            echo "  point the engine at it with: --host 127.0.0.1 --port ${PORT}"
            exit 0
        fi
        echo "  the server is up but NO Vulkan device was found." >&2
        echo "  (check $LOG)" >&2
        exit 2
    fi
done

echo "  the server did not come up within 40 s; check $LOG" >&2
exit 3
