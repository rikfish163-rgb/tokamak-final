import pytest

from experiments.risk_aligned_online.east_decoder_calibration_20261102.resume import (
    completed_folds,
)


def test_resume_reuses_only_complete_checkpoint_pairs(tmp_path):
    assert completed_folds(tmp_path) == []
    (tmp_path / "fold0_training.json").touch()
    with pytest.raises(ValueError):
        completed_folds(tmp_path)
    (tmp_path / "fold0_heads.pt").touch()
    assert completed_folds(tmp_path) == [0]
    (tmp_path / "completed.json").touch()
    with pytest.raises(FileExistsError):
        completed_folds(tmp_path)
