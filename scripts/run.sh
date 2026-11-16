#!/usr/bin/env bash
# Entry point used by the Kaggle notebooks. Usage:
#   bash scripts/run.sh smoke
#   bash scripts/run.sh data
#   bash scripts/run.sh model <hf-model-id> [extra cli args]
set -uo pipefail
cd /kaggle/working/neuraltrace
pip -q install -e . 2>&1 | tail -1
[ $# -ge 1 ] || set -- smoke
stage="$1"
case "$stage" in
  smoke)
    nvidia-smi --query-gpu=name,memory.total --format=csv
    python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'gpus', torch.cuda.device_count())"
    python -m pytest -q 2>&1 | tail -3
    python -m neuraltrace.cli build-data --langs en es hi --n-per-band 12 --out data/smoke.jsonl
    python -m neuraltrace.cli run --model Qwen/Qwen2.5-0.5B-Instruct --data data/smoke.jsonl --out results \
      --limit 300 --steer-per-class 12 --n-random 1 --n-shuffle 1 2>&1 | cut -c1-300 | tail -90
    ;;
  data)
    python -m neuraltrace.cli build-data --langs en es hi --n-per-band 120 --out data/entities.jsonl
    wc -l data/entities.jsonl
    ;;
  model)
    shift
    python -m neuraltrace.cli run --model "$@" 2>&1 | cut -c1-300 | tail -150
    ;;
  *)
    echo "unknown stage: $stage"; exit 2 ;;
esac
