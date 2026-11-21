"""Lock an outcome-blind, conditional EAST replay roster; no inference is run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.risk_aligned_online import io23, simple_data


ROOT = Path(__file__).resolve().parent
INPUT_PATHS = {
    "full": ROOT / "../Fuison/experiments/risk_aligned_online/runs/commissioning_current_all_20261009/full/roster.json",
    "pilot": ROOT / "../Fuison/experiments/risk_aligned_online/runs/commissioning_current_all_20261009/pilot/roster.json",
    "fixed360": ROOT / "config/splits/east_hdf5_authoritative_balanced360_v1.json",
    "paper131_stream": ROOT / "artifacts/east_mainA12_paper131_feature_contract_a12_20260517a12c/matrix/manifest.json",
    "support": ROOT / "config/splits/east_paper131_split_all_in.json",
    "formal51": ROOT / "artifacts/east_prequential_formal51_matrix/chronological_stream_split.json",
    "parent": ROOT / "probe_outputs/qualified_source_static360_20261104_v1/protocol.json",
}
OUTPUT = ROOT / "probe_outputs/conditional_confirmation_roster_20261105_v1.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_inputs():
    payloads, hashes = {}, {}
    for name, raw_path in INPUT_PATHS.items():
        path = raw_path.resolve(strict=True)
        content = path.read_bytes()
        payloads[name] = json.loads(content)
        hashes[str(path)] = hashlib.sha256(content).hexdigest()
    return payloads, hashes


def load_input_payloads():
    return _load_inputs()[0]


def _ids(values, name):
    ids = list(values)
    if any(type(sid) is not int or sid <= 0 for sid in ids):
        raise ValueError(f"{name} IDs must be positive integers")
    return ids


def fixed_ids(inputs):
    fixed = inputs["fixed360"]["test"]
    return set(_ids(fixed["disruption_shots"] + fixed["non_disruption_shots"], "fixed360"))


def extract_candidates(inputs):
    full = _ids(inputs["full"]["shot_ids"], "full")
    if len(full) != len(set(full)):
        raise ValueError("duplicate full-roster IDs")
    support = inputs["support"]["train"]
    excluded_groups = (
        _ids(inputs["pilot"]["shot_ids"], "pilot"),
        fixed_ids(inputs),
        _ids(inputs["paper131_stream"]["stream"]["shot_ids"], "paper131 stream"),
        _ids(support["disruption_shots"] + support["non_disruption_shots"], "support train"),
        _ids(inputs["formal51"]["D_stream"]["shot_ids"], "formal51 stream"),
    )
    return sorted(set(full).difference(*map(set, excluded_groups)))


def check_parent_matches_fixed(inputs):
    order = _ids([row["shot"] for row in inputs["parent"]["order"]], "parent order")
    fixed = fixed_ids(inputs)
    if len(order) != 360 or len(set(order)) != 360 or len(fixed) != 360 or set(order) != fixed:
        raise ValueError("parent order must contain exactly the locked 360 distinct IDs")


def parent_cutoff(parent):
    order = parent["order"]
    if len(order) < 100:
        raise ValueError("parent order must have at least 100 IDs")
    return max(_ids([row["shot"] for row in order[:100]], "parent first 100"))


def later_proxy_ids(ids, cutoff):
    return sorted(sid for sid in ids if sid > cutoff)


def signal_problem(record):
    """Check only the declared grid and flat23 alignment, not signal content."""
    times = np.asarray(record["times"])
    if times.ndim != 1:
        return "time grid must be one-dimensional"
    if times.size == 0:
        return "time grid must be nonempty"
    if not np.isfinite(times).all():
        return "time grid contains non-finite values"
    if not (np.diff(times) > 0).all():
        return "time grid must be strictly increasing"
    expected = (times.size, 23)
    for field in ("values", "available"):
        shape = np.shape(record[field])
        if shape != expected:
            return f"{field} shape must be {expected}; got {shape}"
    return None


def read_nominal(sid):
    return simple_data.read_shot("EAST", sid, rate_contract="nominal")


def _known_contract_failure(exc):
    if isinstance(exc, KeyError):
        return True  # A required signal dataset or signal attribute is absent.
    if isinstance(exc, ValueError):
        return str(exc).startswith((
            "1 kHz nominal contract requires", "unaligned channel length:",
            "unaligned SampleRate:", "unaligned StartTime:",
            "all input arrays must have the same shape", "could not convert string to float:",
        ))
    if isinstance(exc, TypeError):
        return str(exc).startswith(("len() of unsized object", "float() argument must be"))
    if isinstance(exc, IndexError):
        return str(exc).startswith("index 0 is out of bounds for axis 0 with size 0")
    return False


def screen_candidates(candidates, *, read_shot, hash_shot, on_progress=None):
    candidates = _ids(candidates, "candidate")
    if len(candidates) != len(set(candidates)):
        raise ValueError("duplicate candidate IDs")
    candidates.sort()
    ids, excluded, raw_sha256, samples = [], [], {}, {}
    for index, sid in enumerate(candidates, 1):
        raw_sha256[str(sid)] = hash_shot(sid)
        try:
            record = read_shot(sid)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            if not _known_contract_failure(exc):
                raise
            reason = f"{type(exc).__name__}: {exc}"
        else:
            reason = signal_problem(record)
        if reason is None:
            ids.append(sid)
            samples[str(sid)] = int(len(record["times"]))
        else:
            excluded.append({"sid": sid, "reason": reason})
        if on_progress is not None:
            on_progress(index, len(candidates), sid, reason)
    if (set(ids) & {row["sid"] for row in excluded} or
            set(ids) | {row["sid"] for row in excluded} != set(candidates)):
        raise AssertionError("signal screening did not partition the candidates")
    return {"ids": ids, "excluded": excluded, "raw_sha256": raw_sha256, "samples": samples}


def _signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _stable_readers():
    signatures = {}

    def hash_shot(sid):
        path = io23.path_for("EAST", sid)
        before = _signature(path)
        digest = sha256_file(path)
        if _signature(path) != before:
            raise RuntimeError(f"raw file changed while hashing shot {sid}")
        signatures[sid] = before
        return digest

    def read_shot(sid):
        path = io23.path_for("EAST", sid)
        try:
            return read_nominal(sid)
        finally:
            if _signature(path) != signatures[sid]:
                raise RuntimeError(f"raw file changed during signal QC for shot {sid}")

    return read_shot, hash_shot


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite locked roster: {OUTPUT}")
    inputs, input_hashes = _load_inputs()
    candidates = extract_candidates(inputs)
    if len(inputs["full"]["shot_ids"]) != 2953 or len(candidates) != 835:
        raise ValueError("expected 2953 distinct full-roster IDs and 835 initial candidates")
    check_parent_matches_fixed(inputs)
    cutoff = parent_cutoff(inputs["parent"])
    read_shot, hash_shot = _stable_readers()

    def progress(index, total, sid, reason):
        if reason is not None:
            print(f"excluded sid={sid}: {reason}", flush=True)
        if index % 25 == 0 or index == total:
            print(f"signal QC {index}/{total}", flush=True)

    screened = screen_candidates(candidates, read_shot=read_shot, hash_shot=hash_shot,
                                 on_progress=progress)
    for path, expected in input_hashes.items():
        if sha256_file(Path(path)) != expected:
            raise RuntimeError(f"input changed during roster construction: {path}")
    contract = {
        "status": "LOCKED_CONDITIONAL_SIGNAL_ONLY_ROSTER",
        "initial_candidate_count": len(candidates),
        **screened,
        "later_proxy_ids": later_proxy_ids(screened["ids"], cutoff),
        "max_parent_shot_id": cutoff,
        "input_sha256": input_hashes,
        "independent_validation": False,
        "ordering": "Ascending numeric shot ID; not verified acquisition chronology.",
        "later_proxy_definition": "ID greater than max of parent protocol order first 100 IDs, including fit and mapping; proxy only.",
        "signal_qc_definition": "Nominal EAST loader, finite strictly increasing nonempty time grid, aligned (n,23) values and masks; no minimum duration.",
        "raw_sha256_scope": "All initial candidates, including signal-QC exclusions.",
        "samples_scope": "Eligible IDs only, measured on the loader time grid.",
        "interpretation": "Conditional retrospective replay roster; no claim of prior-exposure exclusion or independent inference.",
        "code_sha256": {
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
            str(Path(simple_data.__file__).resolve()): sha256_file(Path(simple_data.__file__).resolve()),
            str(Path(io23.__file__).resolve()): sha256_file(Path(io23.__file__).resolve()),
        },
    }
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(contract, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    print(f"locked eligible={len(screened['ids'])} excluded={len(screened['excluded'])} "
          f"later_proxy={len(contract['later_proxy_ids'])} sha256={sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
