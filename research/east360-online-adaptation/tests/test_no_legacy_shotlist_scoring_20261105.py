"""The maintained verification entry points must not emit a second label policy."""

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "path",
    [
        "experiments/risk_aligned_online/east_context_operator_20261031/audit.py",
        "SHORT_HISTORY_VERIFY_20261031.py",
        "INPUT_FACTORS_POSTHOC_20261031.py",
        "STRATIFIED_COLD20_VERIFY_20261031.py",
    ],
)
def test_hdf_only_scoring_entry_does_not_read_old_outcome_lists(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    assert not {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.endswith(
            ("/Disruption_EAST_Test.json", "/Non-disruption_EAST_Test.json")
        )
    }
