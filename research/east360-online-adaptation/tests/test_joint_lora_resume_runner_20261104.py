import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from test_joint_lora_runner_20261104 import (
    test_complete_fixture_preserves_prediction_then_reveal_and_shared_slow as make_original,
)
from experiments.risk_aligned_online.joint_lora_20261104 import run as original
from experiments.risk_aligned_online.joint_lora_resume_20261104 import run as resume


def test_current_process_cannot_be_treated_as_a_terminal_run():
    text = Path(f"/proc/{os.getpid()}/stat").read_text()
    ticks = int(text[text.rfind(")") + 2 :].split()[19])
    assert resume.process_is_live(os.getpid(), ticks)
    assert not resume.process_is_live(os.getpid(), ticks + 1)
    with pytest.raises(RuntimeError, match="active"):
        resume.run(
            Path("/nonexistent"),
            Path("/nonexistent-output"),
            source_pid=os.getpid(),
            source_start_ticks=ticks,
            source_exit_code=124,
            device="cpu",
        )


@pytest.mark.parametrize("missing_last_eos", [False, True, "truncated"])
def test_resume_matches_uninterrupted_flow_without_rewriting_prefix(
    tmp_path, monkeypatch, missing_last_eos
):
    make_original(tmp_path, monkeypatch)
    reference = tmp_path / "output"
    source = tmp_path / "interrupted"
    source.mkdir()
    (source / "shots").mkdir()
    (source / "inter").mkdir()
    for name in ("warm.pt", "warm.json", "protocol.json", "code.zip"):
        shutil.copy2(reference / name, source / name)
    for path in (reference / "shots").iterdir():
        if int(path.name[:4]) <= 21:
            shutil.copy2(path, source / "shots" / path.name)
    if not missing_last_eos:
        shutil.copy2(
            reference / "inter" / "0021_21.json", source / "inter" / "0021_21.json"
        )
    elif missing_last_eos == "truncated":
        (source / "inter" / "0021_21.json").write_text('{"shot":21')
    before = {
        str(p.relative_to(source)): original.sha(p)
        for p in source.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(resume, "load_inputs", original.load_inputs)
    monkeypatch.setattr(resume, "load_parent", original.load_parent)
    output = tmp_path / "resumed"
    done = resume.run(
        source,
        output,
        source_pid=99999999,
        source_start_ticks=0,
        source_exit_code=124,
        device="cpu",
    )
    assert done["resumed_prefix_shots"] == 21
    assert done["recomputed_past_predictions"] == 0
    assert done["last_sealed_eos_completed_after_interrupt"] == bool(missing_last_eos)
    assert json.loads((output / "per_shot.json").read_text()) == json.loads(
        (reference / "per_shot.json").read_text()
    )
    assert json.loads((output / "summary.json").read_text()) == json.loads(
        (reference / "summary.json").read_text()
    )
    for path in (output / "shots").glob("*.npz"):
        with np.load(path) as a, np.load(reference / "shots" / path.name) as b:
            assert a.files == b.files
            for key in a.files:
                assert np.array_equal(a[key], b[key], equal_nan=True)
    for name, digest in before.items():
        assert original.sha(source / name) == digest
    assert (output / "checkpoint.pt").is_file()
    recovered = json.loads((output / "inter" / "0021_21.json").read_text())
    expected = json.loads((reference / "inter" / "0021_21.json").read_text())
    assert recovered["parent_before"] == expected["parent_before"]
    assert recovered["parent_after"] == expected["parent_after"]
