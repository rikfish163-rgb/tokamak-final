"""Train event-selected auxiliary teachers and screen actual later source risk.

This is source-only OFFLINE auxiliary teaching/validation, not J-TEXT online.
No source parameter, final risk head, alarm parameter or EAST model is trained.
"""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_vector_20261030.source_run import source_gate
from ..east_context_operator_20261031.run import configure
from ..event_prefix_probe import assign_events
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lora_stream_20261102.core import direct
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from ..source_event_tasks_20261102.data import (
    SOURCE_SHA,
    load_records,
    mask_task,
    load_events,
)
from ..source_event_tasks_20261102.trial import load_source, source_basis, windows_pack
from .features import representation, order_step, risk
from .fit import fit

PAIRS = io23.ROOT / "probe_outputs/event_temporal_pairs_v1"
PLAN = io23.ROOT / "EVENT_TEMPORAL_TEACHER_PLAN_20261103.md"
ARMS = (
    "true_order",
    "shuffled_order",
    "matched_other_order",
    "random",
    "reverse_update",
)
TASKS = ("native", "no_MNM_CIII")


def take(pack, lo, hi):
    return {k: v[lo:hi] for k, v in pack.items()}


@torch.no_grad()
def collect(model, basis, signals, rows, field, task):
    packs = []
    for lo in range(0, len(rows), 64):
        local = rows[lo : lo + 64]
        windows = [
            simple_data.window(mask_task(signals[r["shot"]], task), r[field], 128)
            for r in local
        ]
        x, a = zip(*windows, strict=True)
        packs.append(windows_pack(model, np.stack(x), np.stack(a), basis))
    return {k: torch.cat([p[k] for p in packs]) for k in packs[0]}


def loss_array(z, label, threshold):
    boundary = np.log(threshold / (1 - threshold))
    return np.logaddexp(0, (1 - 2 * label) * (z - boundary))


def summary(rows):
    result = {}
    for group in ("safe", "danger", "early", "disruptive"):
        local = [
            r
            for r in rows
            if (r["disruptive"] if group == "disruptive" else r["group"] == group)
        ]
        ids = sorted({r["shot"] for r in local})

        def macro(key):
            return float(
                np.mean(
                    [
                        np.mean([r[key] for r in local if r["shot"] == sid])
                        for sid in ids
                    ]
                )
            )

        if not ids:
            raise ValueError("all fixed evaluation groups required")
        result[group] = dict(
            shots=len(ids),
            episodes=len(local),
            mean_delta_margin_bce=macro("delta_margin_bce"),
            mean_delta_point_bce=macro("delta_point_bce"),
            mean_delta_logit=macro("delta_logit"),
            improved=sum(r["delta_margin_bce"] < -1e-6 for r in local),
            worsened=sum(r["delta_margin_bce"] > 1e-6 for r in local),
        )
    result["balanced"] = (
        result["safe"]["mean_delta_margin_bce"]
        + result["disruptive"]["mean_delta_margin_bce"]
    ) / 2
    return result


def main(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    configure()
    done = read(PAIRS / "completed.json")
    if done["status"] != "COMPLETED_METADATA_SELECTION":
        raise ValueError("completed metadata selection required")
    for name, h in done["artifact_sha256"].items():
        if sha(PAIRS / name) != h:
            raise ValueError("pair selection artifact changed")
    pair_rows = read(PAIRS / "pairs.json")
    records, _, val_prov = load_records("val")
    _, _, train_prov = load_records("train")
    expected_hashes = {**val_prov["input_hashes"], **train_prov["input_hashes"]}
    allowed = {r["shot"] for role in pair_rows.values() for r in role} | {
        r["shot"] for r in records
    }
    signals, times, outcomes, raw_hashes = {}, {}, {}, {}
    advanced = io23.advanced_times()
    with source_gate(allowed):
        for n, sid in enumerate(sorted(allowed), 1):
            path = io23.path_for("J-TEXT", sid)
            raw_hashes[str(sid)] = sha(path)
            if raw_hashes[str(sid)] != expected_hashes[str(sid)]:
                raise ValueError("source input changed")
            raw = apply_input_contract(simple_data.read_shot("J-TEXT", sid))
            flag, down = io23.read_outcome("J-TEXT", sid)
            signals[sid] = {k: raw[k] for k in ("values", "available")}
            times[sid], outcomes[sid] = raw["times"], (flag, down)
            if n % 100 == 0:
                print(
                    dict(stage="read_declared_source", shots=n, total=len(allowed)),
                    flush=True,
                )
    events, _ = load_events()
    for r in records:
        flag, down = outcomes[r["shot"]]
        horizon = io23.effective_horizon_ms(r["shot"], advanced)[0]
        lead = round((down - times[r["shot"]][r["query"]]) * 1000, 9) if flag else None
        group = "safe" if not flag else "danger" if lead <= horizon + 1e-6 else "early"
        if (
            r["disruptive"] != flag
            or r["group"] != group
            or (flag and (lead <= 10 or abs(lead - r["lead_ms"]) > 1e-6))
        ):
            raise ValueError("fixed source episode disagrees with fresh HDF")
        # Keep the original positive-window label; operational loss below has a
        # separate, explicit all-disruptive-query target and never trains source.
        if r["label"] != float(group == "danger"):
            raise ValueError("source window point label changed")
        r["maturity"] = r["origin"] + 25
        family, _ = assign_events(times[r["shot"]], events.get(r["shot"], []))
        r["origin_family"] = int(family[r["origin"]])
    for role, rows in pair_rows.items():
        for r in rows:
            if r["disruptive"] != outcomes[r["shot"]][0]:
                raise ValueError("pair HDF outcome changed")
    model, parent = load_source()
    basis = source_basis(model)
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    threshold = float(parent["threshold"])
    output.mkdir()
    bound = imported_code_hashes()
    bound[str(PLAN)] = sha(PLAN)
    bound[str(PAIRS / "completed.json")] = sha(PAIRS / "completed.json")
    protocol = dict(
        scope="SOURCE_OFFLINE_EVENT_TEACHING_AND_LOCAL_RISK_SCREEN_NOT_JTEXT_ONLINE",
        source_sha256=sha(DEFAULT_PARENT),
        threshold=threshold,
        bound_code_sha256=bound,
        raw_source_sha256=raw_hashes,
        waveform_shots=len(allowed),
        test_reads=0,
        east_reads=0,
        source_parameter_updates=0,
        point_window="max(25ms, AdvancedTime)",
        query_strict_lead_ms=10,
        primary_loss="threshold-margin BCE; all disruptive queries target1",
        point_loss="original window-label BCE retained separately, not optimized",
        update="one zero-start per-episode internal W4x64 step; norm0.1; per-window RMS cap0.25",
        training_task="no_MNM_CIII",
        evaluation_tasks=list(TASKS),
        standardizer="std over both endpoints of matched TRAIN event pairs; ddof0; floor1e-6; no difference centering",
        controls=list(ARMS),
        random_control="common seed131 untrained initial weight",
        family_truth_input=False,
        retrospective_peak_queries=True,
        independent_confirmation=False,
        full_stream_metric=False,
    )
    write(output / "protocol.json", protocol)
    with patch.object(
        io23,
        "read_outcome",
        side_effect=RuntimeError("no outcomes in model computation"),
    ):
        teaching = {
            role: {
                field: collect(model, basis, signals, rows, field, "no_MNM_CIII")
                for field in ("origin", "maturity", "other_origin", "other_maturity")
            }
            for role, rows in pair_rows.items()
        }
        good = torch.ones(len(pair_rows["train"]), dtype=torch.bool, device="cuda")
        values = {}
        for field, pack in teaching["train"].items():
            values[field], valid, _ = representation(pack, pack["phi"].new_zeros(4, 64))
            good &= valid
        exclusions = [
            r
            for r, valid in zip(pair_rows["train"], good.cpu().tolist(), strict=True)
            if not valid
        ]
        train_rows = [
            r
            for r, valid in zip(pair_rows["train"], good.cpu().tolist(), strict=True)
            if valid
        ]
        if not train_rows:
            raise ValueError("no matched valid training pairs")
        endpoints = (
            torch.cat((values["origin"][good], values["maturity"][good]))
            .double()
            .cpu()
            .numpy()
        )
        std = np.maximum(endpoints.std(0), 1e-6)
        event_d = (
            (values["maturity"][good] - values["origin"][good]).cpu().numpy() / std
        ).astype(np.float32)
        other_d = (
            (values["other_maturity"][good] - values["other_origin"][good])
            .cpu()
            .numpy()
            / std
        ).astype(np.float32)
        teachers = fit(
            dict(
                true_order=event_d,
                shuffled_order=event_d.copy(),
                matched_other_order=other_d,
            ),
            train_rows,
        )
        write(output / "teachers.json", teachers)
        write(
            output / "training.json",
            dict(records=train_rows, excluded=exclusions, std128=std.tolist()),
        )
        weights = {
            name: torch.tensor(w, device="cuda", dtype=basis.dtype)
            for name, w in teachers["heads"].items()
        }
        weights["random"] = torch.tensor(
            teachers["initial_weight128"], device="cuda", dtype=basis.dtype
        )
        scale = torch.tensor(std, device="cuda", dtype=basis.dtype)
        val_order = {}
        for role, row_ids in pair_rows.items():
            val_order[role] = {}
            for name, w in weights.items():
                a, va, _ = representation(
                    teaching[role]["origin"], basis.new_zeros(4, 64)
                )
                b, vb, _ = representation(
                    teaching[role]["maturity"], basis.new_zeros(4, 64)
                )
                margin = (((b - a) / scale) * w).sum(1).detach().cpu().numpy()
                valid = (va & vb).cpu().numpy()
                val_order[role][name] = dict(
                    known_pairs=int(valid.sum()),
                    correct=int((margin[valid] > 0).sum()),
                    mean_margin=float(margin[valid].mean()),
                )
        write(output / "order_diagnostic.json", val_order)
        evaluations, all_rows, states, certificates = {}, [], {}, []
        for task in TASKS:
            packs = {
                field: collect(model, basis, signals, records, field, task)
                for field in ("origin", "maturity", "query")
            }
            states[task], local = {arm: [] for arm in ARMS}, {arm: [] for arm in ARMS}
            for lo in range(0, len(records), 64):
                hi = min(lo + 64, len(records))
                origin, mature, later = (
                    take(packs[k], lo, hi) for k in ("origin", "maturity", "query")
                )
                with torch.no_grad():
                    baseline, _ = risk(model, later, basis.new_zeros(4, 64))
                base = baseline.double().cpu().numpy()
                op = np.asarray([r["disruptive"] for r in records[lo:hi]], np.float64)
                point = np.asarray([r["label"] for r in records[lo:hi]], np.float64)
                for arm in ARMS:
                    w = weights["true_order" if arm == "reverse_update" else arm]
                    state, receipt = order_step(
                        origin, mature, w, scale, reverse=arm == "reverse_update"
                    )
                    with torch.no_grad():
                        after, rms = risk(model, later, state)
                    new = after.double().cpu().numpy()
                    delta = loss_array(new, op, threshold) - loss_array(
                        base, op, threshold
                    )
                    point_delta = (np.logaddexp(0, new) - point * new) - (
                        np.logaddexp(0, base) - point * base
                    )
                    states[task][arm].append(state.cpu())
                    for j, r in enumerate(records[lo:hi]):
                        row = dict(
                            **r,
                            task=task,
                            arm=arm,
                            baseline_logit=float(base[j]),
                            logit=float(new[j]),
                            delta_logit=float(new[j] - base[j]),
                            delta_margin_bce=float(delta[j]),
                            delta_point_bce=float(point_delta[j]),
                            query_shift_rms=float(rms[j]),
                            **{k: v[j] for k, v in receipt.items()},
                        )
                        local[arm].append(row)
                    if lo == 0:
                        # Match GEMM batch shape too: each row retains its own
                        # independent state and exact 128-point causal window.
                        windows = [
                            simple_data.window(
                                mask_task(signals[r["shot"]], task), r["query"], 128
                            )
                            for r in records[lo:hi]
                        ]
                        x, a = zip(*windows, strict=True)
                        combined = dict(
                            values=np.stack(x).reshape(-1, 23),
                            available=np.stack(a).reshape(-1, 23),
                        )
                        ends = np.arange(hi - lo) * 128 + 127
                        with torch.no_grad():
                            actual = direct(model, combined, ends, basis, state)[
                                "logit"
                            ]
                            zero = direct(
                                model, combined, ends, basis, basis.new_zeros(4, 64)
                            )["logit"]
                        if not torch.equal(actual, after) or not torch.equal(
                            zero, baseline
                        ):
                            raise AssertionError(
                                (task, arm, "actual first-FFN parity failed")
                            )
                        certificates.append(
                            dict(
                                task=task,
                                arm=arm,
                                episodes=hi - lo,
                                zero_and_updated_exact=True,
                            )
                        )
            evaluations[task] = {
                arm: summary(entries) for arm, entries in local.items()
            }
            for arm in ARMS:
                states[task][arm] = torch.cat(states[task][arm])
                all_rows.extend(local[arm])
            print(
                dict(stage="evaluated", task=task, summary=evaluations[task]),
                flush=True,
            )
    gate = {}
    for task in TASKS:
        candidate = evaluations[task]["true_order"]
        gate[task + "_safe_improved"] = (
            candidate["safe"]["mean_delta_margin_bce"] < -1e-6
        )
        gate[task + "_danger_improved"] = (
            candidate["danger"]["mean_delta_margin_bce"] < -1e-6
        )
        gate[task + "_early_protected"] = (
            candidate["early"]["mean_delta_margin_bce"] <= 1e-6
        )
        for arm in ARMS[1:]:
            gate[task + "_better_than_" + arm] = (
                candidate["balanced"] < evaluations[task][arm]["balanced"] - 1e-6
            )
    gate["passed"] = all(gate.values())
    if (
        any(not torch.equal(v, model.state_dict()[k].cpu()) for k, v in frozen.items())
        or sha(DEFAULT_PARENT) != SOURCE_SHA
    ):
        raise AssertionError("qualified source or model tensors changed")
    if any(sha(Path(p)) != h for p, h in bound.items()):
        raise AssertionError("bound code changed during experiment")
    write(output / "records.json", all_rows)
    write(output / "summary.json", evaluations)
    write(output / "actual_ffn_parity.json", certificates)
    torch.save(states, output / "temporary_states.pt")
    result = dict(
        status="COMPLETED_SOURCE_LOCAL_DIAGNOSTIC",
        records=len(all_rows),
        episodes=len(records),
        gate=gate,
        seconds=time.monotonic() - started,
        source_unchanged=True,
        jtext_online_run=False,
        east_reads=0,
        full_stream_gain_measured=False,
        method_promoted=False,
        actual_ffn_checks=len(certificates),
        artifact_sha256={f.name: sha(f) for f in output.iterdir() if f.is_file()},
    )
    write(output / "completed.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
