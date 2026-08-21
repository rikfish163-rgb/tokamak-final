from __future__ import annotations

import importlib
from pathlib import Path
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/transfer_methods_2026"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def test_a1_feature_loader_does_not_require_outcome_metadata(
    tmp_path: Path, monkeypatch,
) -> None:
    runner = importlib.import_module("run_online_active_a1")
    shot = 12345
    path = tmp_path / "EAST/unified_hdf5/12300/12345.hdf5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for index, name in enumerate(runner.rebuilt.RAW23):
            signal = data.create_dataset(
                name, data=np.full(32, float(index + 1), dtype=np.float32),
            )
            signal.attrs["SampleRate"] = 1000.0
            signal.attrs["StartTime"] = 0.0
        # Deliberately omit /meta. Prediction-time loading must still succeed.

    monkeypatch.setattr(runner.rebuilt, "DATA", tmp_path)
    values, available, rate = runner.load_raw_rebuilt(shot)

    assert values.shape == (32, len(runner.rebuilt.REBUILD))
    assert available.shape == values.shape
    assert rate == 1000.0
