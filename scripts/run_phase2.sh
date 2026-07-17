#!/bin/bash
# Phase 2 automated runner
# Starts LabVLA service in background, waits for ready, runs MuJoCo client, cleans up.

set -e

PROJECT_DIR=~/projects/labvla-mujoco
CONDA_BASE=/home/josan/miniforge3
ENV_NAME=labvla-cu124
LOG_FILE=$PROJECT_DIR/phase2_run.log
SERVICE_LOG=$PROJECT_DIR/phase2_service.log

cd "$PROJECT_DIR"

# Activate conda env (needed so `python` resolves to the right interpreter)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"

echo "[Phase2] Starting LabVLA inference service (background)..." | tee -a "$LOG_FILE"

# Start service in background, redirect output
PYTHONPATH="$PROJECT_DIR/LabVLA" \
  nohup "$PYTHON" LabVLA/deployment/serve_labvla.py \
    --pretrained_path "$PROJECT_DIR/LabVLA-5B-Base" \
    --vlm_path Qwen/Qwen3-VL-4B-Instruct \
    --device cuda --port 8000 \
  > "$SERVICE_LOG" 2>&1 &
SERVICE_PID=$!
echo "[Phase2] Service PID: $SERVICE_PID" | tee -a "$LOG_FILE"

# Poll for service readiness (port 8000).
# Primary check: HTTP /health endpoint.
# Fallback: "Listening on port" string in service log.
echo "[Phase2] Waiting for service to be ready (may take 2-4 min)..." | tee -a "$LOG_FILE"
MAX_WAIT=360
WAITED=0
READY=0
while [ $WAITED -lt $MAX_WAIT ]; do
    # Python-based port probe (avoids curl dependency)
    if "$PYTHON" -c "
import socket, sys
s = socket.socket()
s.settimeout(2)
try:
    s.connect(('127.0.0.1', 8000))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        echo "[Phase2] Service port is open! (after ${WAITED}s)" | tee -a "$LOG_FILE"
        READY=1
        break
    fi
    # Fallback: look for readiness string in log
    if grep -q -E "Listening on port|WebSocket server started|ws.*8000|ready" "$SERVICE_LOG" 2>/dev/null; then
        echo "[Phase2] Service log indicates readiness (after ${WAITED}s)" | tee -a "$LOG_FILE"
        READY=1
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $READY -eq 0 ]; then
    echo "[Phase2] ERROR: Service did not become ready within ${MAX_WAIT}s" | tee -a "$LOG_FILE"
    echo "[Phase2] Last 30 lines of service log:" | tee -a "$LOG_FILE"
    tail -30 "$SERVICE_LOG" | tee -a "$LOG_FILE"
    kill $SERVICE_PID 2>/dev/null || true
    exit 1
fi

# Extra grace period for WebSocket handler to bind fully
sleep 3

# Run MuJoCo client (5 frames)
echo "" | tee -a "$LOG_FILE"
echo "[Phase2] Running MuJoCo closed-loop client (5 frames)..." | tee -a "$LOG_FILE"
PYTHONPATH="$PROJECT_DIR/LabVLA" \
  "$PYTHON" scripts/mujoco_client.py \
    --host 127.0.0.1 --port 8000 \
    --prompt "pick up the beaker" \
    --num_steps 5 \
  2>&1 | tee -a "$LOG_FILE"

CLIENT_EXIT=${PIPESTATUS[0]}

# Cleanup: stop the inference service
echo "" | tee -a "$LOG_FILE"
echo "[Phase2] Cleaning up service (PID $SERVICE_PID)..." | tee -a "$LOG_FILE"
kill $SERVICE_PID 2>/dev/null || true
wait $SERVICE_PID 2>/dev/null || true

if [ $CLIENT_EXIT -eq 0 ]; then
    echo "[Phase2] ✅ Closed-loop validation complete!" | tee -a "$LOG_FILE"
else
    echo "[Phase2] ❌ Client exited with code $CLIENT_EXIT" | tee -a "$LOG_FILE"
    exit $CLIENT_EXIT
fi
