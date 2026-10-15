#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2
export TF_CPP_MIN_LOG_LEVEL=2 MPLCONFIGDIR="$PROJECT_DIR/work/mpl"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
GPU_LIBRARIES="$($PYTHON_BIN -c 'import sys; sys.path.insert(0,"src"); from config import local_gpu_library_dirs; print(":".join(local_gpu_library_dirs()))')"
export LD_LIBRARY_PATH="$GPU_LIBRARIES${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUSIC_GPU_ENV_READY=1
case "${1:-help}" in
 gpu-check) shift; exec "$PYTHON_BIN" src/gpu_check.py "$@" ;;
 download) shift; exec "$PYTHON_BIN" scripts/download_dataset.py "$@" ;;
 prepare) shift; "$PYTHON_BIN" src/data.py "$@"; exec "$PYTHON_BIN" src/features.py "$@" ;;
 train) shift; exec "$PYTHON_BIN" src/train.py "$@" ;;
 evaluate) shift; exec "$PYTHON_BIN" src/evaluate.py "$@" ;;
 predict) shift; exec "$PYTHON_BIN" src/predict.py "$@" ;;
 app) shift; exec "$PYTHON_BIN" -m streamlit run app.py --server.address 127.0.0.1 --server.maxUploadSize 20 --browser.gatherUsageStats false "$@" ;;
 test) shift; exec "$PYTHON_BIN" -m pytest tests "$@" ;;
 *) echo 'Usage: scripts/run.sh {gpu-check|download|prepare|train|evaluate|predict AUDIO|app|test}' ;;
esac
