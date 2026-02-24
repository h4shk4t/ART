#!/bin/bash
cd /home/colligo/experiments/RLM/ART/examples/rlm-training
export WANDB_MODE=disabled
export UV_CACHE_DIR=/code/colligo/uv-cache
export PYTORCH_ALLOC_CONF=expandable_segments:True
exec uv run python train.py \
  --experiment-name debug-fix-test \
  --model-name r2e-rlm-debug-v5 \
  --docker-url https://ashutosh3002--rlm-docker-test-service-fastapi-app.modal.run \
  --max-concurrent 16 \
  --tensor-parallel-size 8 \
  --groups-per-step 2 \
  --rollouts-per-group 2 \
  --max-steps 15 \
  --num-epochs 1 \
  --dataset-size 10 \
  --log-sample-rate 0.5
