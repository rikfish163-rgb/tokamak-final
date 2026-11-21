"""CPU-only, post-hoc paired comparison of completed external static/fixed runs.

CLI paths name the arm directories, not their parents. Both must cover all
2,953 shots; there is deliberately no prefix/force/threshold-scan option.
Nothing imports a runner, loads a checkpoint, or changes an input artifact.
The only optional write is a new JSON report *outside* both run directories.

The historical FFN protocol lacks input_contract/rate_contract. Its reviewed
archived launcher explicitly selects the nominal EAST reader and delegates
to run_arm. Verify that launcher, the shared run_arm AST, and the input helper
bytes rather than silently supplying those missing declarations. A different
launcher requires a fresh review, not an automatic compatibility fallback.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
import re
import zipfile

import numpy as np

from .. import representation_analysis as analysis


FULL_SHOT_COUNT = 2953
_REPO = Path(__file__).resolve().parents[3]
_INPUT_CONTRACT = "flat23 + masks; CIII=-10 candidate sentinel only; no target scaler fit"
_INPUT_HELPERS = ("simple23.py", "simple_data.py", "io23.py", "causal_input.py")
# Reviewed from ffn_fixed_all2953_20261005/code_snapshot.zip, NOT live code.
_REVIEWED_FFN_LAUNCHER_SHA256 = "b4c93ea95120d9a9e435b123137749422db934e9f6b2cfed1dff22bf277eb5e3"
_METRIC_FIELDS = ("tp", "fp", "n_disrupt", "n_safe", "tpr", "fpr", "J",
                  "threshold", "median_lead_ms", "predictions")


def _need(value, fields, where):
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    for field in fields:
        if field not in value:
            raise ValueError(f"{where} missing {field}")
    return value


def _equal(actual, expected, where):
    if actual != expected:
        raise ValueError(f"{where} mismatch")


def _number(value, where):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{where} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{where} must be finite")
    return float(value)


def _check_sha(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"missing or invalid SHA256 for {path}")
    try:
        actual = analysis._sha256(path)
    except OSError as exc:
        raise ValueError(f"cannot verify SHA256 for {path}") from exc
    _equal(actual, expected, f"SHA256 {path}")


def _reference(value):
    if not isinstance(value, str) or not value:
        raise ValueError("artifact reference must be a nonempty path")
    path = Path(value)
    # The archived runners recorded absolute or repository-relative CLI paths.
    return (path if path.is_absolute() else _REPO / path).resolve()


def _read_run(arm_dir, arm, expected_count):
    root = arm_dir.parent
    summary_path = arm_dir / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"incomplete run: missing completed summary {summary_path}")
    summary = _need(analysis._read_json(summary_path),
                    ("arm", "observed_count", "roster_count", "metrics", "evidence",
                     "accepted_intra_updates", "accepted_inter_updates"), str(summary_path))
    for key in ("observed_count", "roster_count"):
        if type(summary[key]) is not int or summary[key] != expected_count:
            raise ValueError(f"incomplete {arm} {key}: require exactly {expected_count}")
    _equal(summary["arm"], "static" if arm == "static" else "intra", f"{arm} summary arm")
    _equal(summary["evidence"], "DEVELOPMENT_LOCKED_STREAM", f"{arm} completeness evidence")
    for key in ("accepted_intra_updates", "accepted_inter_updates"):
        if type(summary[key]) is not int or summary[key] < 0:
            raise ValueError(f"summary {key} must be a nonnegative integer")
    _equal(summary["accepted_inter_updates"], 0, f"{arm} inter updates")
    if arm == "static":
        _equal(summary["accepted_intra_updates"], 0, "static intra updates")
    metrics = _need(summary["metrics"], _METRIC_FIELDS, f"{arm} metrics")
    for key in ("tp", "fp", "n_disrupt", "n_safe"):
        if type(metrics[key]) is not int or metrics[key] < 0:
            raise ValueError(f"metrics {key} must be a nonnegative integer")
    for key in ("tpr", "fpr", "J", "median_lead_ms"):
        if metrics[key] is not None:
            _number(metrics[key], f"metrics {key}")
    predictions = metrics["predictions"]
    if not isinstance(predictions, list) or len(predictions) != expected_count:
        raise ValueError(f"{arm} prediction count must be {expected_count}")
    for prediction in predictions:
        _need(prediction, ("shot", "is_disrupt", "alarm", "timely", "lead_ms"), "prediction")
        if type(prediction["shot"]) is not int:
            raise ValueError("prediction shot must be an integer")
        for key in ("is_disrupt", "alarm", "timely"):
            if type(prediction[key]) is not bool:
                raise ValueError(f"prediction {key} must be boolean")
        if prediction["lead_ms"] is not None:
            _number(prediction["lead_ms"], "prediction lead_ms")
    completion = root / ("summary.json" if arm == "static" else "comparison.json")
    if not completion.is_file():
        raise ValueError(f"incomplete run: missing completion artifact {completion}")
    completed = _need(analysis._read_json(completion), (arm,), str(completion))
    _equal(completed[arm], summary, f"{arm} root/arm completed summary")
    protocol = _need(analysis._read_json(root / "protocol.json"),
                     ("lineage", "source_sha256", "prior_sha256", "source_gate", "model_config",
                      "threshold", "args", "order_sha256", "roster_sha256",
                      "code_snapshot_sha256", "alarm_tuning"), f"{arm} protocol")
    args = _need(protocol["args"], ("checkpoint", "prior", "order", "roster", "arms", "limit"),
                 f"{arm} args")
    if not isinstance(args["arms"], list) or arm not in args["arms"]:
        raise ValueError(f"{arm} absent from declared arms")
    if type(args["limit"]) is not int or args["limit"] not in (0, expected_count):
        raise ValueError(f"{arm} limit is incomplete; require full {expected_count}")
    threshold = _number(protocol["threshold"], f"{arm} threshold")
    if not 0 <= threshold <= 1:
        raise ValueError(f"{arm} threshold must be in [0, 1]")
    _equal(_number(metrics["threshold"], "metrics threshold"), threshold, f"{arm} threshold")
    order = analysis._read_json(root / "order.json")
    roster = _need(analysis._read_json(root / "roster.json"),
                   ("shot_ids", "selection_protocol", "order_sha256"), f"{arm} roster")
    if not isinstance(order, list) or len(order) != expected_count:
        raise ValueError(f"{arm} order must contain {expected_count} shots")
    for item in order:
        _need(item, ("shot", "order_method"), f"{arm} order row")
        if type(item["shot"]) is not int or item["shot"] <= 0:
            raise ValueError(f"{arm} order shot must be a positive integer")
        _equal(item["order_method"], "shot_number", f"{arm} order method")
    shots = [item["shot"] for item in order]
    if shots != sorted(set(shots)):
        raise ValueError(f"{arm} order must have unique, increasing shot IDs")
    _equal(roster["shot_ids"], shots, f"{arm} roster/order")
    _equal([p["shot"] for p in predictions], shots, f"{arm} summary roster/order")
    return dict(path=arm_dir, root=root, arm=arm, protocol=protocol, summary=summary,
                order=order, roster=roster, shots=shots)


def _snapshot(run):
    path = run["root"] / "code_snapshot.zip"
    _check_sha(path, run["protocol"]["code_snapshot_sha256"])
    names = (*_INPUT_HELPERS, "redesign_20261004/run.py")
    if run["arm"] == "fixed":
        names += ("ffn_intra_20261005/run.py",)
    try:
        with zipfile.ZipFile(path) as archive:
            for name in names:
                if archive.namelist().count(name) != 1 or archive.getinfo(name).file_size > 1024 * 1024:
                    raise ValueError(f"missing, duplicate or oversized snapshot member: {name}")
            return {name: archive.read(name) for name in names}
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f"invalid code snapshot {path}") from exc


def _function_ast(source, name):
    try:
        matches = [node for node in ast.parse(source).body
                   if isinstance(node, ast.FunctionDef) and node.name == name]
    except (SyntaxError, ValueError) as exc:
        raise ValueError("invalid archived runner syntax") from exc
    if len(matches) != 1:
        raise ValueError(f"snapshot must contain exactly one {name}")
    return ast.dump(matches[0], include_attributes=False)


def _contracts(static, fixed):
    left, right = static["protocol"], fixed["protocol"]
    for field in ("source_sha256", "prior_sha256", "threshold", "model_config",
                  "order_sha256", "roster_sha256"):
        _equal(right[field], left[field], field)
    _equal(fixed["order"], static["order"], "paired full order/provenance")
    _equal(fixed["roster"], static["roster"], "paired full roster/provenance")
    for run in (static, fixed):
        p = run["protocol"]
        gate = _need(p["source_gate"], ("qualified", "checkpoint_sha256"), "source_gate")
        if gate["qualified"] is not True or p["alarm_tuning"] is not False:
            raise ValueError("qualified source required and alarm_tuning must be false")
        _equal(gate["checkpoint_sha256"], p["source_sha256"], "qualified parent")
        for argument, sha_field in (("checkpoint", "source_sha256"), ("prior", "prior_sha256"),
                                    ("order", "order_sha256"), ("roster", "roster_sha256")):
            original = _reference(p["args"][argument])
            _check_sha(original, p[sha_field])
            if argument in ("order", "roster"):
                _equal(analysis._read_json(original), run[argument], f"saved {argument} vs hashed original")
        _equal(run["roster"]["order_sha256"], p["order_sha256"], "roster order_sha256")
        selection = _need(run["roster"]["selection_protocol"],
                          ("rate_contract", "rate_atol_hz", "actual_rate_and_times_preserved",
                           "label_blind", "outcome_class_filter", "prediction_filter"), "selection_protocol")
        for key, expected in dict(rate_contract="nominal", rate_atol_hz=.5,
                                  actual_rate_and_times_preserved=True, label_blind=True,
                                  outcome_class_filter=False, prediction_filter=False).items():
            _equal(selection[key], expected, f"roster {key}")
    static_fields = dict(
        lineage="INDEPENDENT_EAST_INTERNAL_REDESIGN_NOT_V3C", input_contract=_INPUT_CONTRACT,
        rate_contract="nominal", rate_atol_hz=.5, source_offline_only=True,
        target_labels_before_first_shot=0, support110_used_for_initialization=False,
        threshold_origin="fixed J-TEXT VAL160", east_label_ms=125., lead_rule="strict >30ms",
        roster_count=len(static["shots"]))
    fixed_fields = dict(
        lineage="FFN_INTRA_SOURCE_LEARNED_SIGNAL_LOSS_NOT_V3C", source_only_initialization=True,
        inter_updates=False, target_label_access="natural reveal after whole-shot durable seal")
    for p, fields in ((left, static_fields), (right, fixed_fields)):
        _need(p, fields, "protocol")
        for field, expected in fields.items():
            _equal(p[field], expected, field)
    _need(left["args"], ("rate_contract",), "static args")
    _equal(left["args"]["rate_contract"], "nominal", "static args rate_contract")
    for key, expected in (("input_contract", _INPUT_CONTRACT), ("rate_contract", "nominal")):
        if key in right:
            _equal(right[key], expected, f"fixed {key}")
    _need(left, ("prior_protocol", "helper_code_sha256", "code_sha256"), "static protocol")
    _need(right, ("source_rule_protocol",), "fixed protocol")
    for p, key, fields in (
        (left, "prior_protocol", dict(parent_sha256=left["source_sha256"], source_only=True,
                                      static_classifier_bitwise_unchanged=True)),
        (right, "source_rule_protocol", dict(parent_sha256=left["source_sha256"],
                                             prior_sha256=left["prior_sha256"], source_only=True,
                                             classifier_and_decoder_unchanged=True, threshold=left["threshold"]))):
        _need(p[key], fields, key)
        for field, expected in fields.items():
            _equal(p[key][field], expected, f"{key}.{field}")

    a, b = _snapshot(static), _snapshot(fixed)
    _need(left["helper_code_sha256"], _INPUT_HELPERS, "helper_code_sha256")
    hashes = {}
    for name in _INPUT_HELPERS:
        _equal(a[name], b[name], f"input_contract helper {name}")
        hashes[name] = hashlib.sha256(a[name]).hexdigest()
        _equal(hashes[name], left["helper_code_sha256"][name], f"input helper SHA256 {name}")
    runner = "redesign_20261004/run.py"
    _need(left["code_sha256"], ("run.py",), "code_sha256")
    _equal(hashlib.sha256(a[runner]).hexdigest(), left["code_sha256"]["run.py"], "static runner SHA256")
    _equal(_function_ast(a[runner], "run_arm"), _function_ast(b[runner], "run_arm"), "shared run_arm")
    _equal(hashlib.sha256(b["ffn_intra_20261005/run.py"]).hexdigest(),
           _REVIEWED_FFN_LAUNCHER_SHA256, "reviewed FFN launcher input_contract")
    return dict(basis="verified_archived_code", contract=_INPUT_CONTRACT, rate_contract="nominal",
                fixed_declaration_present="input_contract" in right,
                fixed_rate_declaration_present="rate_contract" in right,
                helper_sha256=hashes, shared_run_arm_ast_equal=True,
                reviewed_ffn_launcher_sha256=_REVIEWED_FFN_LAUNCHER_SHA256)


def _sealed_receipts(run):
    """Verify the whole arm before any NPZ or outcome is opened; retain no arrays."""
    directory, shots = run["path"], run["shots"]
    stems = [f"{index:04d}_{shot}" for index, shot in enumerate(shots, 1)]
    for suffix in ("_seal.json", "_sealed.npz", "_audit.json"):
        if {p.name for p in directory.glob(f"*{suffix}")} != {stem + suffix for stem in stems}:
            raise ValueError(f"incomplete or extra {suffix} artifacts in {directory}")
    receipts = []
    for stem, shot in zip(stems, shots):
        receipt = _need(analysis._read_json(directory / f"{stem}_seal.json"),
                        ("shot", "arm", "threshold", "npz", "sha256"), "seal receipt")
        _equal(receipt["shot"], shot, "seal shot")
        _equal(receipt["arm"], run["summary"]["arm"], "seal arm")
        _equal(_number(receipt["threshold"], "seal threshold"), run["protocol"]["threshold"], "seal threshold")
        _equal(receipt["npz"], f"{stem}_sealed.npz", "seal npz filename")
        path = (directory / receipt["npz"]).resolve()
        if path.parent != directory:
            raise ValueError("sealed NPZ escapes arm directory")
        _check_sha(path, receipt["sha256"])
        receipts.append(receipt)
    return receipts


def _trace(directory, receipt):
    path = directory / receipt["npz"]
    # Recheck at consumption, after the global preflight; never trust a cached
    # success if a file changed meanwhile. Arrays are processed one pair at a time.
    _check_sha(path, receipt["sha256"])
    try:
        with np.load(path, allow_pickle=False) as archive:
            for key in ("shot", "times", "scores"):
                if key not in archive:
                    raise ValueError(f"sealed NPZ missing {key}")
            shot = archive["shot"]
            if shot.shape != () or shot.dtype.kind not in "iu" or int(shot) != receipt["shot"]:
                raise ValueError("sealed NPZ shot mismatch")
            times, scores = analysis._row_arrays({"times": archive["times"], "scores": archive["scores"]})
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f"invalid sealed NPZ {path}") from exc
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("sealed times must be nonempty and strictly increasing")
    return dict(shot=receipt["shot"], times=times, scores=scores)


def _strict_outcomes(loader):
    def attach(source, device, advanced):
        result = _need(loader(source, device, advanced),
                       ("shot", "device", "is_disrupt", "down", "valid"), "outcome")
        _equal(result["shot"], source["shot"], "outcome shot")
        _equal(result["device"], "EAST", "outcome device")
        if not isinstance(result["is_disrupt"], (bool, np.bool_)):
            raise ValueError("outcome is_disrupt must be boolean")
        if result["is_disrupt"] or result["down"] is not None:
            _number(result["down"], "outcome down")
        valid = np.asarray(result["valid"])
        if valid.dtype != bool or valid.shape != source["times"].shape:
            raise ValueError("outcome valid must be an aligned boolean mask")
        return dict(result, valid=valid.copy())
    return attach


def _metrics(predictions, threshold):
    return analysis._metrics_from_class_counts(
        Counter(p["class"] for p in predictions), threshold=threshold, predictions=predictions,
        timely_leads=[p["lead_ms"] for p in predictions if p["timely"]])


def _segment(before, after, threshold, start, stop):
    a, b = before[start:stop], after[start:stop]
    metrics = {name: analysis._public_metrics(_metrics(rows, threshold))
               for name, rows in (("static", a), ("fixed", b))}
    early = {name: [p["shot"] for p in rows if p["timely"] and p["lead_ms"] > 3000.]
             for name, rows in (("static", a), ("fixed", b))}
    return dict(
        first_index=start + 1 if a else None, last_index=start + len(a) if a else None,
        shot_count=len(a), **metrics,
        delta_fixed_minus_static={key: (metrics["fixed"][key] - metrics["static"][key]
                                       if all(metrics[n][key] is not None for n in metrics) else None)
                                  for key in ("tp", "fp", "tpr", "fpr", "J", "median_lead_ms")},
        paired_flips=analysis.paired_flips(a, b, before_arm="static", after_arm="fixed"),
        early_above_3s=dict(
            **{name: dict(count=len(shots), shots=shots) for name, shots in early.items()},
            gained_shots=sorted(set(early["fixed"]) - set(early["static"])),
            lost_shots=sorted(set(early["static"]) - set(early["fixed"]))))


def compare_runs(static, fixed, output=None, *, expected_count=FULL_SHOT_COUNT, outcome_loader=None):
    """Verify and compare two completed arms; expected_count is injectable for CPU fixtures only."""
    paths = [Path(p).resolve() for p in (static, fixed)]
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError("expected_count must be a positive integer")
    if output is not None:
        output = Path(output).resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite analysis output: {output}")
        if any(output.is_relative_to(path.parent) for path in paths):
            raise ValueError("analysis output must be outside both input runs")
    runs = [_read_run(path, arm, expected_count) for path, arm in zip(paths, ("static", "fixed"))]
    input_contract = _contracts(*runs)
    receipts = [_sealed_receipts(run) for run in runs]
    if outcome_loader is None:
        # Check the live outcome reader against the archived contract without
        # importing/executing any archived code or model module.
        for module in (analysis.simple_data, analysis.simple_data.io23):
            path = Path(module.__file__).resolve()
            _check_sha(path, input_contract["helper_sha256"][path.name])
    loader = _strict_outcomes(outcome_loader or analysis.simple_data.attach_outcome)
    predictions = [[], []]
    threshold = runs[0]["protocol"]["threshold"]
    for a, b in zip(*receipts):
        traces = [_trace(path, receipt) for path, receipt in zip(paths, (a, b))]
        if not np.array_equal(traces[0]["times"], traces[1]["times"]):
            raise ValueError(f"paired time grid differs for shot {a['shot']}")
        rows = analysis._outcome_rows(traces, loader)
        for key in ("is_disrupt", "down"):
            _equal(rows[0][key], rows[1][key], f"paired outcome {key}")
        if not np.array_equal(rows[0]["times"], rows[1]["times"]):
            raise ValueError("paired outcome validity differs")
        for destination, row in zip(predictions, rows):
            destination.append(analysis.metrics_for_rows([row], threshold)["predictions"][0])
    for run, rows in zip(runs, predictions):
        analysis._verify_summary(run["summary"], run["summary"]["metrics"],
                                 _metrics(rows, threshold), expected_count)
    report = dict(
        format="ffn-external-paired-v1", analysis_only=True, model_write=False, gpu_used=False,
        evidence="DEVELOPMENT_PAIRED_COMPLETE_ROSTER", expected_count=expected_count,
        runs={run["arm"]: str(run["path"]) for run in runs},
        source_sha256=runs[0]["protocol"]["source_sha256"],
        prior_sha256=runs[0]["protocol"]["prior_sha256"], threshold=threshold,
        lead_rule="round((DownTime-time)*1000, 9) > 30.0; score >= fixed threshold",
        early_alarm_rule="first alarm on a disruptive shot has rounded lead_ms > 3000 (strict)",
        segments={name: _segment(*predictions, threshold, lo, hi) for name, lo, hi in
                  (("overall", 0, expected_count), ("first30", 0, 30),
                   ("31to120", 30, 120), ("121toend", 120, expected_count))},
        verification=dict(full2953_verified=expected_count == FULL_SHOT_COUNT,
                          completed_summaries_verified=True, exact_parent_and_prior_sha256_verified=True,
                          exact_roster_order_and_threshold_verified=True, input_contract=input_contract,
                          sealed_sha256_verified_before_outcome_join=True, summary_metrics_verified=True,
                          outcome_source=("injected CPU fixture loader" if outcome_loader is not None
                                          else "simple_data.attach_outcome")),
        interpretation="All data development-exposed; no independent validation, significance, or physics claim.",
        update_diagnostics="not collected; bounded per-shot trace processing only")
    if output is not None:
        analysis._write_json_once(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", type=Path, required=True, help="completed static arm directory")
    parser.add_argument("--fixed", type=Path, required=True, help="completed fixed arm directory")
    parser.add_argument("--output", type=Path, required=True, help="new external JSON report")
    args = parser.parse_args(argv)
    try:
        compare_runs(args.static, args.fixed, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"comparison refused: {exc}\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
