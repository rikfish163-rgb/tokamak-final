"""Versioned motion probe with historical-view replay separated from analysis."""

from pathlib import Path

import joint_lora_persistent_motion_probe_20261104 as p
import numpy as np
import torch

ROOT = p.ROOT
OUTPUT = ROOT / "probe_outputs/joint_lora_persistent_motion_probe_20261104_v2"
REPORT = ROOT / "JOINT_LORA_PERSISTENT_MOTION_PROBE_RESULT_20261104_v2.md"


def verify_historical_view(sealed, historical, analysis):
    arrays = [np.asarray(x) for x in (sealed, historical, analysis)]
    if not arrays[0].shape or any(
        x.shape != arrays[0].shape or not np.isfinite(x).all() for x in arrays
    ):
        raise ValueError("finite matching historical/analysis views required")
    a, b, c = arrays
    gap = float(np.max(np.abs(a - b)))
    if gap > 3e-6:
        raise ValueError("historical source view must replay its own seal")
    return dict(historical=gap, analysis_vs_historical=float(np.max(np.abs(c - b))))


def read_features(base, index, sid, mapping, hashes, metadata):
    # This old reader verifies seals/raw SHA/v5 truth/clock and returns the
    # explicit mapped analysis view, not a reproduction of unmapped warm risk.
    analysis, selected, native, receipt = p._read_sealed(
        index, sid, mapping, hashes, metadata
    )
    if index <= 20:
        raw = p.simple_data.read_shot("EAST", sid, rate_contract="nominal")
        historical = p.prepare_actual(raw, index, mapping)
        if p.sha(p.io23.path_for("EAST", sid)) != hashes[str(sid)]:
            raise ValueError("raw changed while restoring historical source view")
    else:
        historical = analysis
    maturity = [case["arrival"] for case in selected]
    if not maturity:
        raise ValueError("fixed sample plan must produce cases")
    xh, ah = p.window_batch(historical, maturity, device="cpu")
    xa, aa = p.window_batch(analysis, maturity, device="cpu")
    with torch.no_grad():
        historic_p = base(xh, ah)["logit"].sigmoid().numpy()
        analytic_z = base(xa, aa)["logit"].numpy()
        analytic_p = torch.from_numpy(analytic_z).sigmoid().numpy()
    gaps = verify_historical_view(native[maturity], historic_p, analytic_p)
    rows = []
    for i, case in enumerate(selected):
        if (
            case["arrival"] != case["origin"] + 8
            or case["future"] != case["arrival"] + 24
        ):
            raise ValueError("unchanged current-only motion observation clock required")
        x, a = p.window_batch(analysis, [case["arrival"]], device="cpu")
        with torch.no_grad():
            level = p.levels(base, x, a)[0, -32:, list(p.CHANNEL_INDICES)].numpy()
        mask = a[0, -32:, list(p.CHANNEL_INDICES)].numpy()
        perm = p.bin_permutation(sid, case["arrival"])
        feature = p.motion_features(level, mask, perm)
        rows.append(
            dict(
                stream_index=index,
                shot=sid,
                **case,
                maturity=case["arrival"],
                native_logit=float(analytic_z[i]),
                native_parity_abs=float(abs(native[case["arrival"]] - historic_p[i])),
                analytic_vs_historic_abs=float(abs(analytic_p[i] - historic_p[i])),
                endpoint=np.where(mask[-1], level[-1], 0.0).astype(np.float64),
                endpoint_mask=mask[-1].copy(),
                permutation=perm,
                **feature,
            )
        )
    receipt.update(
        historical_view="unmapped" if index <= 20 else "fixed_map_after20",
        analytic_view="fixed_map",
        view_comparison=gaps,
        extra_historical_signal_read=index <= 20,
    )
    return rows, receipt


def run():
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("preserve both failed v1 and completed v2")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    paths = dict(
        run_protocol=p.RUN / "protocol.json",
        v5_completed=p.V5 / "completed.json",
        v5_audit=p.V5 / "audit.json",
        v5_metadata=p.V5 / "hdf5_metadata.json",
        subspace_completed=p.SUBSPACE / "completed.json",
        subspace_warm_cases=p.SUBSPACE / "warm_cases.json",
        subspace_test_cases=p.SUBSPACE / "cases.json",
        map=p.MAP,
        raw_hashes=p.RAW_HASHES,
        source=p.DEFAULT_PARENT,
    )
    hashes = {key: p.sha(path) for key, path in paths.items()}
    if hashes != p.EXPECTED:
        raise ValueError("original protocol/v5/parent/fixed cases changed")
    for key in ("v5", "subspace"):
        folder = p.V5 if key == "v5" else p.SUBSPACE
        done = p.read_json(folder / "completed.json")
        for name, digest in done["artifact_sha256"].items():
            if p.sha(folder / name) != digest:
                raise ValueError("upstream artifact SHA mismatch")
    original, mapping, raw_hashes = (
        p.read_json(paths[name]) for name in ("run_protocol", "map", "raw_hashes")
    )
    ids = [r["shot"] for r in original["order"]]
    if (
        len(ids) != len(set(ids))
        or len(ids) != 360
        or mapping["calibration_ids"] != ids[:20]
        or any(
            original["input_sha256"][str(paths[k])] != hashes[k]
            for k in ("map", "raw_hashes")
        )
    ):
        raise ValueError("locked order or ended20 map changed")
    truth = p.read_json(paths["v5_metadata"])["EAST"]
    metadata = {r["shot"]: r for r in truth}
    if len(metadata) != len(truth) or any(s not in metadata for s in ids[:100]):
        raise ValueError("v5 HDF identities must be available")
    code_names = (
        *p.CODE_PATHS,
        Path(__file__).name,
        "tests/test_joint_lora_persistent_motion_v2_20261104.py",
    )
    code = {name: p.sha(ROOT / name) for name in code_names}
    plan = p.fixed_plan(ids[:20], ids[20:100], hashes)
    plan.update(
        scope="VERSION2_SEPARATE_HISTORICAL_REPLAY_AND_FIXED_MAP_ANALYSIS",
        previous_failed_protocol_sha256=p.sha(p.OUTPUT / "protocol.json"),
        correction="Validate historic unmapped/mapped risk against its own seal; separately use fixed mapped analysis view in train/eval. No feature, fit, or case-selection change.",
        historical_first20_view="unmapped",
        analytical_first20_view="fixed_map",
        held_view="fixed_map",
        extra_historical_raw_reads=20,
        code_sha256=code,
        input_paths={k: str(v) for k, v in paths.items()},
        source_state_sha256=original["source_state_sha256"],
    )
    OUTPUT.mkdir()
    p.write_json(OUTPUT / "protocol.json", plan)
    base, parent = p.load_parent(p.DEFAULT_PARENT)
    base.eval().requires_grad_(False)
    before = p.state_digest(base)
    if (
        before != original["source_state_sha256"]
        or parent["threshold"] != original["threshold"]
    ):
        raise ValueError("fixed source/threshold identity changed")
    fit, held, receipts = [], [], []
    for i, sid in enumerate(ids[:20], 1):
        rows, receipt = read_features(base, i, sid, mapping, raw_hashes, metadata)
        fit.extend(rows)
        receipts.append(receipt)
    def identity(rows):
        return [(r["shot"], r["origin"], r["group"], r["label"]) for r in rows]
    if identity(fit) != identity(p.read_json(paths["subspace_warm_cases"])):
        raise ValueError("warm cases changed")
    weights = p.balanced_weights(fit)
    labels = np.asarray([r["label"] for r in fit], float)
    offsets = np.asarray([r["native_logit"] for r in fit], float)
    models = {
        a: p.fit_readout(p.design(fit, a), offsets, labels, weights) for a in p.ARMS[1:]
    }
    logits_fit = dict(
        native=offsets,
        **{
            a: p.predict_readout(m, p.design(fit, a), offsets)
            for a, m in models.items()
        },
    )
    print("mapped warm20 readouts fixed; held80 remains evaluation-only", flush=True)
    for i, sid in enumerate(ids[20:100], 21):
        rows, receipt = read_features(base, i, sid, mapping, raw_hashes, metadata)
        held.extend(rows)
        receipts.append(receipt)
        if i % 10 == 0:
            print(dict(processed=i, held_cases=len(held)), flush=True)
    if identity(held) != identity(p.read_json(paths["subspace_test_cases"])):
        raise ValueError("held cases changed")
    offsets = np.asarray([r["native_logit"] for r in held], float)
    logits_held = dict(
        native=offsets,
        **{
            a: p.predict_readout(m, p.design(held, a), offsets)
            for a, m in models.items()
        },
    )
    if (
        p.state_digest(base) != before
        or any(q.grad is not None or q.requires_grad for q in base.parameters())
        or any(p.sha(path) != hashes[k] for k, path in paths.items())
        or any(p.sha(ROOT / name) != digest for name, digest in code.items())
    ):
        raise ValueError("parent, inputs or code changed")
    summary = {
        key: dict(
            shots=n,
            cases=len(rows),
            metrics=p.summarize(rows, logits),
            coverage=p._motion_coverage(rows),
            native_parity_max_abs=max(r["native_parity_abs"] for r in rows),
            analysis_historical_max_abs=max(
                r["analytic_vs_historic_abs"] for r in rows
            ),
        )
        for key, rows, logits, n in (
            ("fit", fit, logits_fit, 20),
            ("held", held, logits_held, 80),
        )
    }
    array_fields = (
        "shot",
        "origin",
        "maturity",
        "future",
        "label",
        "native_logit",
        "endpoint",
        "endpoint_mask",
        "common",
        "short_supported",
        "long_supported",
        "short",
        "long",
        "permuted",
        "permutation",
        "long_bins",
        "permuted_bins",
    )
    arrays = {}
    serial = []
    omitted = set(array_fields) - {
        "shot",
        "origin",
        "maturity",
        "future",
        "label",
        "native_logit",
    }
    for stage, rows, logits in (("fit", fit, logits_fit), ("held", held, logits_held)):
        arrays.update(
            {
                f"{stage}_{field}": np.stack([r[field] for r in rows])
                for field in array_fields
            }
        )
        arrays.update({f"{stage}_logit_{a}": v for a, v in logits.items()})
        serial.extend(
            dict(stage=stage, **{k: v for k, v in r.items() if k not in omitted})
            for r in rows
        )
    with (OUTPUT / "features_and_logits.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    fit_json = {
        a: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in m.items()}
        for a, m in models.items()
    }
    for name, data in (
        ("cases.json", serial),
        ("shots.json", receipts),
        ("fit.json", fit_json),
        ("summary.json", summary),
    ):
        p.write_json(OUTPUT / name, data)
    with REPORT.open("x", encoding="utf-8") as stream:
        stream.write("# V2：历史回放校验与统一映射分析视图已分开\n\n")
        stream.write(
            "前20炮历史分数按未映射输入核对；另行使用结束20炮后的固定映射构造拟合与评估特征。"
            "映射分析分数不描述为前20炮的原始在线预测。V1失败记录保留。\n\n"
        )
        stream.write(p.make_report(summary, plan))
    names = (
        "protocol.json",
        "features_and_logits.npz",
        "cases.json",
        "shots.json",
        "fit.json",
        "summary.json",
    )
    p.write_json(
        OUTPUT / "completed.json",
        dict(
            status="COMPLETED_VERSION2_PERSISTENT_MOTION_DEVELOPMENT_PROBE",
            script_sha256=p.sha(Path(__file__)),
            report_sha256=p.sha(REPORT),
            artifact_sha256={name: p.sha(OUTPUT / name) for name in names},
        ),
    )
    print({name: s["metrics"]["arms"] for name, s in summary.items()}, flush=True)


if __name__ == "__main__":
    run()
