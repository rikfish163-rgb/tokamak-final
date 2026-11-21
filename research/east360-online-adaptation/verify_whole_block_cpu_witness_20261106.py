"""Replay selected bound VAL traces on CPU without reading any outcomes."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import run_whole_block_update_20261106 as run
import verify_whole_block_update_20261106 as verify
from verify_confirmation_raw_replay_20261105 import Hashes


def witness(root):
    h = Hashes()
    out = root / "cpu_witness.json"
    if out.exists():
        raise FileExistsError(out)
    cfg, seal, _ = verify.preflight(root, h)
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    base, _ = run.r.load_parent(run.r.DEFAULT_PARENT)
    fits = run.load_nets(root, base)
    before = run.r.state_digest(base)
    if before != cfg["state_sha256"]:
        raise ValueError("CPU witness needs the same qualified source state")
    rows, selected = [], {q["shot"] for q in seal["prefix"]}
    with patch.object(
        run.r.io23, "read_outcome", side_effect=RuntimeError("blind CPU witness")
    ):
        for row in seal["rows"]:
            sid = row["shot"]
            if sid not in selected:
                continue
            raw = run.r.simple_data.read_shot("J-TEXT", sid)
            a = run.predict(base, fits, raw)
            with np.load(root / "val" / row["file"], allow_pickle=False) as b:
                err = float(np.max(np.abs(a["logits"] - b["logits"])))
                norm = float(np.max(np.abs(a["norms"] - b["norms"])))
                same = np.array_equal(a["logits"] >= run.TAU, b["logits"] >= run.TAU)
                if (
                    err > 1e-4
                    or norm > 2e-6
                    or not same
                    or not np.array_equal(a["active"], b["active"])
                    or not np.array_equal(a["common"], b["common"])
                ):
                    raise ValueError(f"label-blind CPU/GPU witness disagrees: {sid}")
            rows.append(
                dict(
                    shot=sid,
                    logit_error=err,
                    norm_error=norm,
                    threshold_bits_equal=same,
                )
            )
    if len(rows) != len(selected) or run.r.state_digest(base) != before:
        raise ValueError("incomplete witness or changed source state")
    h.check()
    run.r.write_new(
        out,
        dict(
            status="REPLAYED_NO_OUTCOMES",
            rows=rows,
            held_outcome_reads=0,
            seal_sha256=h(root / "val" / "completed.json"),
            verifier_sha256=h(Path(__file__)),
        ),
    )
    print(rows, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    witness(p.parse_args().root)
