#!/bin/bash
# Phase 2 + Tactile + Heatmap automated runner (T006)
# Starts LabVLA service in background, waits for ready, runs the HEATMAP
# MuJoCo client (mujoco_client_heatmap.py) which pops a live 12x12 pressure
# window alongside the closed loop, cleans up.
#
# Requires a display (WSLg on WS). Pass EXTRA_ARGS="--no-viz" for headless.

set -e

PROJECT_DIR=~/projects/labvla-mujoco
CONDA_BASE=/home/josan/miniforge3
ENV_NAME=labvla-cu124
LOG_FILE=$PROJECT_DIR/phase2_heatmap_run.log
SERVICE_LOG=$PROJECT_DIR/phase2_heatmap_service.log
OBJECT_TYPE=${OBJECT_TYPE:-beaker}
EXTRA_ARGS=${EXTRA_ARGS:-}

cd "$PROJECT_DIR"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"

echo "[Phase2-Heatmap] Starting LabVLA inference service (background)..." | tee -a "$LOG_FILE"

PYTHONPATH="$PROJECT_DIR/LabVLA" \
  nohup "$PYTHON" LabVLA/deployment/serve_labvla.py \
    --pretrained_path "$PROJECT_DIR/LabVLA-5B-Base" \
    --vlm_path Qwen/Qwen3-VL-4B-Instruct \
    --device cuda --port 8000 \
  > "$SERVICE_LOG" 2>&1 &
SERVICE_PID=$!
echo "[Phase2-Heatmap] Service PID: $SERVICE_PID" | tee -a "$LOG_FILE"

echo "[Phase2-Heatmap] Waiting for service to be ready (may take 4-5 min)..." | tee -a "$LOG_FILE"
MAX_WAIT=360
WAITED=0
READY=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if "$PYTHON" -c "
import socket, sys
s = socket.socket()
s.settimeout(2)
try:
    s.connect(('127.0.0.1', 8000))
    s.close()
except Exception:
    sys.exit(1)
sys.exit(0)
" 2>/dev/null; then
        echo "[Phase2-Heatmap] Service port is open! (after ${WAITED}s)" | tee -a "$LOG_FILE"
        READY=1
        break
    fi
    if grep -q -E "Listening on port|WebSocket server started|ws.*8000|ready" "$SERVICE_LOG" 2>/dev/null; then
        echo "[Phase2-Heatmap] Service log indicates readiness (after ${WAITED}s)" | tee -a "$LOG_FILE"
        READY=1
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $READY -eq 0 ]; then
    echo "[Phase2-Heatmap] ERROR: Service did not become ready within ${MAX_WAIT}s" | tee -a "$LOG_FILE"
    tail -30 "$SERVICE_LOG" | tee -a "$LOG_FILE"
    kill $SERVICE_PID 2>/dev/null || true
    exit 1
fi

sleep 3

echo "" | tee -a "$LOG_FILE"
echo "[Phase2-Heatmap] Running heatmap client (5 frames, object=$OBJECT_TYPE, extra='$EXTRA_ARGS')..." | tee -a "$LOG_FILE"
DISPLAY=${DISPLAY:-:0} \
PYTHONPATH="$PROJECT_DIR/LabVLA" \
  "$PYTHON" scripts/mujoco_client_heatmap.py \
    --host 127.0.0.1 --port 8000 \
    --prompt "pick up the beaker" \
    --num_steps 5 \
    --object_type "$OBJECT_TYPE" \
    $EXTRA_ARGS \
  2>&1 | tee -a "$LOG_FILE"

CLIENT_EXIT=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
echo "[Phase2-Heatmap] Cleaning up service (PID $SERVICE_PID)..." | tee -a "$LOG_FILE"
kill $SERVICE_PID 2>/dev/null || true
wait $SERVICE_PID 2>/dev/null || true

if [ $CLIENT_EXIT -eq 0 ]; then
    echo "[Phase2-Heatmap] ✅ Heatmap closed-loop validation complete!" | tee -a "$LOG_FILE"
else
    echo "[Phase2-Heatmap] ❌ Client exited with code $CLIENT_EXIT" | tee -a "$LOG_FILE"
    exit $CLIENT_EXIT
fi
