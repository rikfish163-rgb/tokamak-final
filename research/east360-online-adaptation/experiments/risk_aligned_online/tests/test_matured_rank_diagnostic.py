import pytest

from experiments.risk_aligned_online.matured_rank_20261023.diagnose_transfer import probe_counts


def test_probe_snapshots_are_prior_updates_only_and_unique():
    rows=[dict(first_affected_prediction=c) for c in (256,384,512,640)]
    assert probe_counts(rows,600)==[256,384,512]
    assert probe_counts(rows,384)==[256,384]
    assert probe_counts(rows,128)==[]


def test_probe_rejects_unordered_update_receipts():
    with pytest.raises(ValueError):probe_counts([dict(first_affected_prediction=384),dict(first_affected_prediction=256)],512)
