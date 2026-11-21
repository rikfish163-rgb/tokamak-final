from pathlib import Path

import pytest

from layer_timing_clock_20261105 import RevealClock
from moving_peak_replay_20261105 import sha, write_new


def test_chained_seal_outcome_reveal_events(tmp_path):
    calls = []

    def hdf(device, sid):
        calls.append((device, sid))
        return False, None

    run = tmp_path
    clock = RevealClock(run, write_new, hdf)
    write_new(run / "0001_123.npz", {"curve": 1})
    with pytest.raises(RuntimeError, match="no matching sealed"):
        clock.outcome("EAST", 123)
    c = run / "0001_123.npz"
    first = run / "0001_123_seal.json"
    clock.write(first, dict(shot=123, rank=1, curve_sha256=sha(c)))
    assert clock.outcome("EAST", 123) == (False, None)
    with pytest.raises(RuntimeError, match="already read"):
        clock.outcome("EAST", 123)
    revealed = run / "0001_123_revealed.json"
    clock.write(revealed, dict(shot=123, after_seal_sha256=sha(first)))
    assert clock.previous_sha == sha(revealed)
    assert calls == [("EAST", 123)]
    assert clock.next_rank == 2


def test_label_reader_refuses_missing_or_changed_sealed_curve(tmp_path):
    clock = RevealClock(tmp_path, write_new, lambda *_: pytest.fail("HDF was accessed"))
    c = tmp_path / "0001_123.npz"
    s = tmp_path / "0001_123_seal.json"
    with pytest.raises(FileNotFoundError):
        clock.write(s, dict(shot=123, rank=1, curve_sha256="x"))
    c.write_bytes(b"first")
    clock.write(s, dict(shot=123, rank=1, curve_sha256=sha(c)))
    c.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="curve changed"):
        clock.outcome("EAST", 123)


def test_reveal_cannot_be_recorded_before_hdf_read(tmp_path: Path):
    clock = RevealClock(tmp_path, write_new, lambda *_: (False, None))
    c, s = [tmp_path / name for name in ("0001_123.npz", "0001_123_seal.json")]
    c.write_bytes(b"curve")
    clock.write(s, dict(shot=123, rank=1, curve_sha256=sha(c)))
    with pytest.raises(RuntimeError, match="not read"):
        clock.write(
            tmp_path / "0001_123_revealed.json",
            dict(shot=123, after_seal_sha256=sha(s)),
        )
