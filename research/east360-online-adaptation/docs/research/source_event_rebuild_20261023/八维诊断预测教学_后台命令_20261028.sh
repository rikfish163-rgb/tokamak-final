#!/usr/bin/env bash
# Source-only queue; EAST is considered only after inspecting admission.json.
set -eu
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.context_memory_20261028.run --forecast-objective --output experiments/risk_aligned_online/runs/context8_forecast_source_service_20261028
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.context_memory_20261028.checkpoint experiments/risk_aligned_online/runs/context8_forecast_source_service_20261028
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.context_memory_20261028.forecast_audit experiments/risk_aligned_online/runs/context8_forecast_source_service_20261028
/home/hetaisheng/miniconda3/envs/rl_env/bin/python -m experiments.risk_aligned_online.context_memory_20261028.forecast_gate experiments/risk_aligned_online/runs/context8_forecast_source_service_20261028
