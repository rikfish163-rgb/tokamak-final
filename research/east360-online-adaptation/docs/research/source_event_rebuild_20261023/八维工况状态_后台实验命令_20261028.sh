#!/usr/bin/env bash
# One GPU job at a time; new output roots; a failed audit stops the queue.
set -eu
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.context_memory_20261028.run --output experiments/risk_aligned_online/runs/context8_source_service_20261028
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.context_memory_20261028.checkpoint experiments/risk_aligned_online/runs/context8_source_service_20261028
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.lifecycle_stream_20261025.run --output experiments/risk_aligned_online/runs/lifecycle_context8360_service_20261028 --target-loss disruptive_positive_only --readout full --feedback progressive --context-memory experiments/risk_aligned_online/runs/context8_source_service_20261028/context8_epoch2.pt
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.lifecycle_stream_20261025.audit experiments/risk_aligned_online/runs/lifecycle_context8360_service_20261028
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.context_memory_20261028.compare experiments/risk_aligned_online/runs/lifecycle_prior_zero360_service_20261028 experiments/risk_aligned_online/runs/lifecycle_context8360_service_20261028
