#!/bin/bash
cd /home/colligo/experiments/RLM/ART/examples/rlm-training
export WANDB_MODE=disabled
export UV_CACHE_DIR=/code/colligo/uv-cache
export PYTORCH_ALLOC_CONF=expandable_segments:True
exec uv run python train.py \
  --experiment-name first-real-run \
  --model-name r2e-rlm-qwen3-14b-v7 \
  --docker-url https://ashutosh3002--rlm-docker-test-service-fastapi-app.modal.run \
  --max-concurrent 16 \
  --tensor-parallel-size 8 \
  --groups-per-step 4 \
  --rollouts-per-group 4 \
  --max-steps 100 \
  --num-epochs 50 \
  --log-sample-rate 0.2
