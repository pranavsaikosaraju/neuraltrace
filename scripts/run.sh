#!/usr/bin/env bash
# Entry point used by the Kaggle notebooks. Usage:
#   bash scripts/run.sh smoke
#   bash scripts/run.sh data
#   bash scripts/run.sh model <hf-model-id> [extra cli args]
#   bash scripts/run.sh v2
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
  v2)
    # Second pass: selective abstention + steering layer sweep for every size,
    # and a full run of a second model family. Small models share the two GPUs.
    python -m neuraltrace.cli build-data --langs en es hi --n-per-band 120 --out data/entities.jsonl
    wc -l data/entities.jsonl
    mkdir -p logs v2
    run() {
      gpu="$1"; m="$2"; shift 2; s="${m##*/}"
      CUDA_VISIBLE_DEVICES="$gpu" python -m neuraltrace.cli run --model "$m" --out raw "$@" > "logs/$s.log" 2>&1
      echo "== $s exit $? =="; grep -E "selective|sweep layer|Error|error" "logs/$s.log" | cut -c1-330 | tail -12; tail -3 "logs/$s.log" | cut -c1-300
    }
    Q="--langs en es --steer-langs --sweep-langs en --steer-per-class 80 --no-probe"
    ( for z in 0.5B 1.5B 3B; do run 0 "Qwen/Qwen2.5-$z-Instruct" $Q --emit-extras "v2/Qwen2.5-$z-Instruct.json"; done ) &
    ( run 1 microsoft/Phi-3.5-mini-instruct --langs en es --steer-langs en --sweep-langs en --steer-per-class 80 \
        --n-random 4 --n-shuffle 2 --emit v2/Phi-3.5-mini-instruct.json ) &
    wait
    run 0,1 Qwen/Qwen2.5-7B-Instruct $Q --sweep-null 2 --emit-extras v2/Qwen2.5-7B-Instruct.json
    ls -la v2
    ;;
  *)
    echo "unknown stage: $stage"; exit 2 ;;
esac
