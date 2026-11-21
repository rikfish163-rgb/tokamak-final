#!/usr/bin/env bash
# Fixed source-only trial. No automatic EAST promotion on a partial or failed gate.
set -eu
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.context_memory_20261028.run --scaled-forecast --output experiments/risk_aligned_online/runs/context8_scaled_source_service_20261028
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.context_memory_20261028.checkpoint experiments/risk_aligned_online/runs/context8_scaled_source_service_20261028
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.context_memory_20261028.forecast_audit experiments/risk_aligned_online/runs/context8_scaled_source_service_20261028
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.context_memory_20261028.forecast_gate experiments/risk_aligned_online/runs/context8_scaled_source_service_20261028
