"""CPU-only causal source-event contexts for the already-ended EAST20 records."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..east_decoder_calibration_20261102.run import LOCAL
from ..event_conditional_feedback_20261102.context import probabilities
from ..event_prefix_probe import causal_summary
from ..history_gradient_metric_20261102.run import read, sha, write
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from ..warmstart_alignment_20261022.alignment import transform

CONTEXT = io23.ROOT / "probe_outputs/source_event_context_paired_v1"
TEACHER = io23.ROOT / "probe_outputs/east_channel_direction20_v1"
PLAN = io23.ROOT / "CONDITIONAL_CHANNEL_POLICY_PLAN_20261102.md"


def source_context_features(signal, origin, center, scale, heads):
    if (
        set(signal) != {"values", "available"}
        or type(origin) is not int
        or origin < 127
        or set(heads) != {"true", "shuffled"}
    ):
        raise ValueError("past-only full window and two frozen context heads required")
    x, mask = simple_data.window(signal, origin, 128)
    available = torch.as_tensor(mask[None], device="cpu").clone()
    available[:, :, io23.RAW23.index("MNM")] = False
    with torch.no_grad():
        raw = causal_summary(
            torch.as_tensor(x[None], device="cpu"), available, center.cpu(), scale.cpu()
        ).numpy()
    return {name: probabilities(raw, head)[0].tolist() for name, head in heads.items()}


def main(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    started = time.monotonic()
    torch.set_num_threads(4)
    done, context_done = (
        read(TEACHER / "completed.json"),
        read(CONTEXT / "completed.json"),
    )
    if (
        done["status"] != "COMPLETED_POSTHOC_DIAGNOSTIC"
        or context_done["status"] != "COMPLETED"
    ):
        raise ValueError("completed source context and local teacher required")
    for name, digest in done["artifact_sha256"].items():
        if sha(TEACHER / name) != digest:
            raise ValueError("channel teacher changed")
    if (
        sha(CONTEXT / "context_heads.json")
        != context_done["artifact_sha256"]["context_heads.json"]
    ):
        raise ValueError("paired source contexts changed")
    p = read(TEACHER / "protocol.json")
    local = read(LOCAL / "protocol.json")
    if local["ids"] != p["ids"] or len(p["ids"]) != 20:
        raise ValueError("ended20 roster mismatch")
    mapping_path = Path(local["calibration_reference"]) / "input_map.json"
    mapping = read(mapping_path)
    if mapping["calibration_ids"] != p["ids"]:
        raise ValueError("input map fitted to different startup roster")
    base, _ = load_parent(DEFAULT_PARENT)
    heads = read(CONTEXT / "context_heads.json")
    rows = read(TEACHER / "records.json")
    roster = {}
    for r in rows:
        key = r["shot"], r["query"]
        metadata = {
            k: r[k]
            for k in ("shot", "query", "label", "disruptive", "actual_lead_ms", "group")
        }
        if key in roster and metadata != roster[key]:
            raise ValueError("teacher modes disagree on episode identity")
        roster[key] = metadata
    if len(roster) != 100 or {sid for sid, _ in roster} != set(p["ids"]):
        raise ValueError("all fixed100 local episodes required")
    original_samples = {
        (r["shot"], r["query"]): r for r in read(LOCAL / "samples.json")
    }
    features = []
    for sid in p["ids"]:
        if sha(io23.path_for("EAST", sid)) != p["input_hashes"][str(sid)]:
            raise ValueError("EAST20 raw changed")
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
        for key, r in roster.items():
            if key[0] != sid:
                continue
            origin = int(r["query"]) - 26
            if original_samples[key]["origin"] != origin:
                raise ValueError("fixed feedback clock drift")
            context = source_context_features(
                signal, origin, base.center, base.scale, heads
            )
            features.append(dict(**r, origin=origin, context=context))
    if sha(DEFAULT_PARENT) != context_done["source_sha256"]:
        raise AssertionError("qualified source changed")
    root.mkdir()
    bound = imported_code_hashes()
    bound[str(PLAN)] = sha(PLAN)
    write(
        root / "protocol.json",
        dict(
            scope="ENDED20_SOURCE_CONTEXT_POLICY_PREPARATION",
            ids=p["ids"],
            teacher_completed_sha256=sha(TEACHER / "completed.json"),
            teacher_records_sha256=sha(TEACHER / "records.json"),
            context_completed_sha256=sha(CONTEXT / "completed.json"),
            context_heads_sha256=sha(CONTEXT / "context_heads.json"),
            source_sha256=sha(DEFAULT_PARENT),
            input_map_sha256=sha(mapping_path),
            input_hashes=p["input_hashes"],
            feature_clock="origin query-26; past128 only",
            context_between_feedback_modes="same factual prefix",
            bound_code_sha256=bound,
            gpu_used=False,
            source_parameter_updates=0,
            suffix_raw_reads=0,
            independent_validation=False,
        ),
    )
    write(root / "features.json", features)
    census = {}
    for name in ("true", "shuffled"):
        x = np.asarray([r["context"][name] for r in features])
        census[name] = dict(
            mean_probabilities=x.mean(0).tolist(),
            std_probabilities=x.std(0).tolist(),
            predicted_family_counts=np.bincount(x.argmax(1), minlength=6).tolist(),
            mean_entropy=float(-(x * np.log(x.clip(1e-30))).sum(1).mean()),
        )
    write(root / "census.json", census)
    result = dict(
        status="COMPLETED",
        episodes=len(features),
        raw_shots=20,
        gpu_used=False,
        source_parameter_updates=0,
        suffix_raw_reads=0,
        seconds=time.monotonic() - started,
        artifact_sha256={f.name: sha(f) for f in root.iterdir() if f.is_file()},
    )
    write(root / "completed.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
