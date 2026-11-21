#!/usr/bin/env bash
# Explicit new roots, no overwrite, no skipped audit, one CUDA job at a time.
set -eu
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.direction_teaching_20261026.run --zero-prior-ablation --output experiments/risk_aligned_online/runs/zero_prior_source_service_20261028
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.direction_teaching_20261026.audit experiments/risk_aligned_online/runs/zero_prior_source_service_20261028
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.lifecycle_stream_20261025.run --output experiments/risk_aligned_online/runs/lifecycle_prior_learned360_service_20261028 --target-loss disruptive_positive_only --readout full --feedback progressive --direction-memory experiments/risk_aligned_online/runs/zero_prior_source_service_20261028/detached_epoch2.pt
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.lifecycle_stream_20261025.audit experiments/risk_aligned_online/runs/lifecycle_prior_learned360_service_20261028
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.lifecycle_stream_20261025.run --output experiments/risk_aligned_online/runs/lifecycle_prior_zero360_service_20261028 --target-loss disruptive_positive_only --readout full --feedback progressive --direction-memory experiments/risk_aligned_online/runs/zero_prior_source_service_20261028/zero_prior_epoch2.pt
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.lifecycle_stream_20261025.audit experiments/risk_aligned_online/runs/lifecycle_prior_zero360_service_20261028
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.direction_teaching_20261026.compare experiments/risk_aligned_online/runs/lifecycle_prior_learned360_service_20261028 experiments/risk_aligned_online/runs/lifecycle_prior_zero360_service_20261028
