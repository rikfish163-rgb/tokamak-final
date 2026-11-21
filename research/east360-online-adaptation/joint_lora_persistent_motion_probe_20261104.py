"""Fixed warm20 / sealed80 offline diagnostic for persistent past motion."""

from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"
sys.dont_write_bytecode = True

import json
import math
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit
import torch

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.direct_fast_memory_20261104.model import levels
from experiments.risk_aligned_online.direct_fast_memory_20261104.timeline import window_batch
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import prepare_actual
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from joint_lora_gradient_probe_20261104 import sha, state_digest
from joint_lora_measurement_probe_20261104 import WARM_STATE as ORIGINAL_WARM_STATE, select_cases, validate_seal


ROOT = io23.ROOT
RUN = ROOT / "probe_outputs/joint_lora_paired360_20261104_v1"
V5 = ROOT / "probe_outputs/hdf5_outcome_authority_20261104_v5"
SUBSPACE = ROOT / "probe_outputs/joint_lora_subspace_probe_20261104_v1"
MAP = ROOT / "probe_outputs/stratified_cold20_context360_v1/input_map.json"
RAW_HASHES = ROOT / "probe_outputs/stratified_cold20_context360_v1/input_hashes.json"
OUTPUT = ROOT / "probe_outputs/joint_lora_persistent_motion_probe_20261104_v1"
REPORT = ROOT / "JOINT_LORA_PERSISTENT_MOTION_PROBE_RESULT_20261104.md"
LAMBDA = 0.1
SEED = 20261104
CHANNEL_INDICES = tuple(i for i in range(23) if i not in (2, 10))
CHANNEL_NAMES = tuple(io23.RAW23[i] for i in CHANNEL_INDICES)
GROUPS = ("safe", "earlier", "positive")
ARMS = ("native", "no_motion", "short", "long", "short_long", "permuted_long", "short_permuted")
COMPARISONS = (
    ("no_motion", "native"), ("short", "no_motion"), ("long", "no_motion"),
    ("long", "permuted_long"), ("short_long", "short"),
    ("short_long", "no_motion"), ("short_long", "short_permuted"),
)
EXPECTED = {
    "run_protocol": "5fe569459f9bb533d50b95adc9c7390585d2cfa846fb4493d39b589ff850a835",
    "v5_completed": "5d0e33682a1cd62ced8b39b428a1b0da885b93b49672ef6aa63b173cf6ae8222",
    "v5_audit": "3d57696edc26c71bffbfaed17375f305c8e5341246c7b29c66b5e8b70f4eedd0",
    "v5_metadata": "286f838542f5186be61696485bd3952fda66bcacc72a32a27f72a96971ea79b1",
    "subspace_completed": "034ff76a994f5b1fb3ece9f65ea77af03494f2dcb0667255ace941e19217e8ff",
    "subspace_warm_cases": "7c0507d3e6818548e0963d4cd7f25cbae5141f5f43edd11ad668f74679d45c0d",
    "subspace_test_cases": "8b25d342600613915cb38d245e82596d6459a3d72a4ee13f6ecba02b3e1056fe",
    "map": "43c37ad981261e059ac8bf8d11d89324bad375dd7405decc194c51e53614aff2",
    "raw_hashes": "c919aa60bd251eaaa5a7e43ef1ed8fa5d0e867f6552db0d53871ccd1e4ad5a86",
    "source": PARENT_SHA256,
}
CODE_PATHS = (
    "joint_lora_persistent_motion_probe_20261104.py",
    "tests/test_joint_lora_persistent_motion_probe_20261104.py",
    "joint_lora_measurement_probe_20261104.py",
    "joint_lora_gradient_probe_20261104.py",
    "experiments/risk_aligned_online/io23.py",
    "experiments/risk_aligned_online/simple_data.py",
    "experiments/risk_aligned_online/causal_input.py",
    "experiments/risk_aligned_online/simple23.py",
    "experiments/risk_aligned_online/direct_fast_memory_20261104/model.py",
    "experiments/risk_aligned_online/direct_fast_memory_20261104/timeline.py",
    "experiments/risk_aligned_online/mask_repair_20261104/east_evaluation.py",
    "experiments/risk_aligned_online/warmstart_alignment_20261022/alignment.py",
    "experiments/risk_aligned_online/redesign_20261004/prepare_prior.py",
)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def fixed_plan(train_ids, test_ids, bindings):
    if (len(train_ids) != 20 or len(set(train_ids)) != 20 or len(test_ids) != 80
            or len(set(test_ids)) != 80 or set(train_ids) & set(test_ids)):
        raise ValueError("distinct fixed warm20 and sealed80 required")
    return dict(
        scope="FIXED_PAST_ONLY_PERSISTENT_MOTION_WARM20_SEALED80_OFFLINE_DIAGNOSTIC",
        train_indices=[1, 20], train_shots=train_ids, test_indices=[21, 100], test_shots=test_ids,
        input_sha256=bindings, window_samples=[8, 32], bins=4, future_offset_samples=24,
        sampling="o=127+32k; t=o+8; endpoint=t+24=o+32; preterminal/in record; HDF EAST125 labels; positive lead>30ms; <=3 uniform eligible coordinates per shot; no score or gradient ranking",
        motion="last 8 or 32 already-arrived samples ending at t; 4 equal bins; observed-only bin median with >=half samples; median adjacent-bin difference divided by bin width and multiplied by fraction of adjacent differences with matching sign",
        masks="one shared per-channel support: current endpoint observed and all four bins supported in BOTH windows; no missing observation is replaced by an observed value; base and all controls receive the same support indicators",
        channels=list(CHANNEL_NAMES),
        channel_semantics="RAW23, excluding always-unobserved EAST MNM and excluded mode_number_n; ne_nG is the existing normalized density diagnostic, not reconstructed absolute density; fixed numerical density map applied",
        permutation="one deterministic pseudorandom permutation of the 4 long-bin medians per shot/maturity, common across channels, seed derived only from fixed seed, shot and maturity; preserves each bin median/mask multiset; one control, not a permutation significance test",
        arms=list(ARMS),
        readout="offline logistic diagnostic only: unmodified native risk logit at t is a fixed offset; all fitted arms include the same mapped source-asinh endpoint levels, endpoint masks and common support masks; add short, long and/or permuted trends as named",
        objective="warm-only group->shot->case equal weights; weighted BCE plus lambda/2 times squared standardized coefficients; intercept unpenalized; standardization estimated on warm only; L-BFGS-B fixed settings",
        **{"lambda": LAMBDA}, optimizer_max_iterations=1000, optimizer_ftol=1e-12, optimizer_gtol=1e-8,
        permutation_seed=SEED,
        metrics="fit and held-out shot-balanced endpoint BCE, per-group paired shot mean differences and counts; weighted point-level AUC overall and safe-vs-positive/earlier-vs-positive; no shot alarm TPR/FPR",
        test_labels_fit=False, test_hyperparameter_search=False, native_model_updated=False,
        independent_validation=False, online_policy_tested=False, cpu_threads=1,
    )


def bin_permutation(shot, maturity):
    if type(shot) is not int or type(maturity) is not int or shot <= 0 or maturity < 0:
        raise ValueError("integer shot and maturity required")
    seed = (SEED + shot * 1000003 + maturity) % (2**63)
    return np.random.default_rng(seed).permutation(4)


def _bin_medians(values, mask, width):
    output = np.full((4, values.shape[1]), np.nan, dtype=np.float64)
    supported = np.zeros_like(output, dtype=bool)
    minimum = (width + 1) // 2
    for j in range(4):
        block, observed = values[j * width : (j + 1) * width], mask[j * width : (j + 1) * width]
        for c in range(values.shape[1]):
            good = observed[:, c]
            if int(good.sum()) >= minimum:
                output[j, c] = np.median(block[good, c])
                supported[j, c] = True
    return output, supported


def _trend(medians, width, common):
    output = np.zeros(medians.shape[1], dtype=np.float64)
    for c in np.flatnonzero(common):
        delta = np.diff(medians[:, c])
        middle = np.median(delta)
        coherence = float(np.mean(np.sign(delta) == np.sign(middle))) if middle != 0 else 0.0
        output[c] = middle * coherence / width
    return output


def motion_features(values, available, permutation):
    values, available, permutation = np.asarray(values), np.asarray(available), np.asarray(permutation)
    if values.ndim != 2 or values.shape[0] != 32 or available.shape != values.shape or available.dtype != np.bool_:
        raise ValueError("32 past samples and a matching boolean mask are required")
    if permutation.shape != (4,) or sorted(permutation.tolist()) != [0, 1, 2, 3]:
        raise ValueError("a four-bin permutation is required")
    if not np.isfinite(values[available]).all():
        raise ValueError("observed inputs must be finite")
    long_bins, long_mask = _bin_medians(values, available, 8)
    short_bins, short_mask = _bin_medians(values[-8:], available[-8:], 2)
    common = available[-1] & long_mask.all(axis=0) & short_mask.all(axis=0)
    permuted_bins = long_bins[permutation].copy()
    return dict(short=_trend(short_bins, 2, common), long=_trend(long_bins, 8, common),
                permuted=_trend(permuted_bins, 8, common), common=common,
                long_bins=long_bins, permuted_bins=permuted_bins,
                short_supported=available[-1] & short_mask.all(axis=0),
                long_supported=available[-1] & long_mask.all(axis=0))


def history_motion(values, available, maturity, permutation):
    if type(maturity) is not int or maturity < 31 or maturity >= len(values):
        raise ValueError("a full 32-point causal history is required")
    return motion_features(values[maturity - 31 : maturity + 1], available[maturity - 31 : maturity + 1], permutation)


def balanced_weights(rows):
    counts = Counter((row["group"], row["shot"]) for row in rows)
    shots = Counter(group for group, _ in counts)
    if set(shots) != set(GROUPS):
        raise ValueError("all three strata are required for shot-balanced fitting")
    result = np.asarray([1 / (3 * shots[row["group"]] * counts[(row["group"], row["shot"])]) for row in rows], dtype=np.float64)
    if not np.isclose(result.sum(), 1.0, rtol=0, atol=1e-12):
        raise AssertionError("weights must sum to one")
    return result


def weighted_auc(labels, scores, weights):
    labels, scores, weights = np.asarray(labels), np.asarray(scores), np.asarray(weights)
    if labels.ndim != 1 or labels.shape != scores.shape or scores.shape != weights.shape or not np.isfinite(scores).all():
        raise ValueError("matching finite AUC arrays required")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("binary labels and nonnegative finite weights required")
    positive, negative = labels == 1, labels == 0
    denom = weights[positive].sum() * weights[negative].sum()
    if denom == 0:
        return None
    difference = scores[positive, None] - scores[None, negative]
    kernel = (difference > 0).astype(float) + 0.5 * (difference == 0)
    return float(np.sum(kernel * weights[positive, None] * weights[None, negative]) / denom)


def fit_readout(features, offset, labels, weights):
    features, offset, labels, weights = (np.asarray(x, dtype=np.float64) for x in (features, offset, labels, weights))
    if (features.ndim != 2 or offset.shape != (len(features),) or labels.shape != offset.shape or weights.shape != offset.shape
            or not all(np.isfinite(x).all() for x in (features, offset, labels, weights))
            or not np.isin(labels, (0, 1)).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1.0)):
        raise ValueError("finite warm features, offsets, labels and normalized weights required")
    mean = np.sum(features * weights[:, None], axis=0)
    scale = np.sqrt(np.sum((features - mean) ** 2 * weights[:, None], axis=0))
    scale[scale < 1e-12] = 1.0
    standard = (features - mean) / scale
    def objective(theta):
        z = offset + theta[0] + standard @ theta[1:]
        error = weights * (expit(z) - labels)
        loss = np.dot(weights, np.logaddexp(0, z) - labels * z) + LAMBDA * np.dot(theta[1:], theta[1:]) / 2
        grad = np.r_[error.sum(), standard.T @ error + LAMBDA * theta[1:]]
        return float(loss), grad
    result = minimize(objective, np.zeros(features.shape[1] + 1), jac=True, method="L-BFGS-B",
                      options=dict(maxiter=1000, ftol=1e-12, gtol=1e-8))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"fixed readout optimization failed: {result.message}")
    return dict(mean=mean, scale=scale, intercept=float(result.x[0]), beta=result.x[1:],
                objective=float(result.fun), iterations=int(result.nit), grad_inf=float(np.abs(result.jac).max()),
                optimizer_message=str(result.message))


def predict_readout(model, features, offset):
    features, offset = np.asarray(features, dtype=np.float64), np.asarray(offset, dtype=np.float64)
    return offset + model["intercept"] + ((features - model["mean"]) / model["scale"]) @ model["beta"]


def verify_v5(sid, times, disruptive, down, stat, meta):
    times = np.asarray(times, dtype=np.float64)
    expected_times = float(meta["start"]) + np.arange(meta["samples"], dtype=np.float64) / float(meta["rate"])
    if (meta["shot"] != sid or meta["disruptive"] is not disruptive or meta["down"] != down
            or stat.st_size != meta["size"] or stat.st_mtime_ns != meta["mtime_ns"]
            or not np.array_equal(times, expected_times) or not len(times)
            or not math.isclose(float(times[-1]), float(meta["end"]), rel_tol=0, abs_tol=1e-10)):
        raise ValueError(f"v5 HDF truth, stat or clock mismatch: {sid}")


def _read_sealed(index, sid, mapping, hashes, metadata):
    stem = f"{index:04d}_{sid}"
    seal_path, curve_path = RUN / "shots" / f"{stem}_seal.json", RUN / "shots" / f"{stem}.npz"
    forecast_path = RUN / "shots" / f"{stem}_forecasts.npz"
    seal = read_json(seal_path)
    seal_hash, curve_hash = sha(seal_path), sha(curve_path)
    forecast_hash = None
    if index <= 20:
        if (seal.get("shot") != sid or seal.get("current_outcome_used_in_prediction") is not False
                or seal.get("curve_sha256") != curve_hash or seal.get("forecast_sha256") is not None):
            raise ValueError("invalid original warm seal")
    else:
        forecast_hash = sha(forecast_path)
        validate_seal(seal, sid, curve_hash, forecast_hash, ORIGINAL_WARM_STATE)
    with np.load(curve_path, allow_pickle=False) as data:
        sealed_times, native = data["times"].copy(), data["native"].copy()
    hdf = io23.path_for("EAST", sid)
    stat = hdf.stat()
    expected = hashes[str(sid)]
    if sha(hdf) != expected:
        raise ValueError(f"fixed raw HDF SHA mismatch: {sid}")
    signal_reader, truth_reader = io23.read_signals, io23.read_outcome
    calls = []
    def only_signal(device, requested, *args, **kwargs):
        if device != "EAST" or requested != sid:
            raise ValueError("only current sealed shot may be read")
        calls.append("signal")
        return signal_reader(device, requested, *args, **kwargs)
    def only_truth(device, requested):
        if device != "EAST" or requested != sid or calls != ["signal"]:
            raise ValueError("only current sealed shot outcome may be read")
        calls.append("outcome")
        return truth_reader(device, requested)
    with (patch.object(io23, "read_signals", side_effect=only_signal),
          patch.object(io23, "read_outcome", side_effect=only_truth)):
        raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
        disruptive, down = io23.read_outcome("EAST", sid)
    if calls != ["signal", "outcome"] or not np.array_equal(raw["times"], sealed_times):
        raise ValueError("signal access or sealed clock mismatch")
    # No candidates are chosen until the old seal, raw SHA, v5 truth and clock agree.
    verify_v5(sid, raw["times"], disruptive, down, stat, metadata[sid])
    signal = prepare_actual(raw, 21, mapping)
    selected = select_cases(raw["times"], disruptive, down)
    after = hdf.stat()
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha(hdf) != expected:
        raise ValueError("HDF changed during scoped read")
    if sha(seal_path) != seal_hash or sha(curve_path) != curve_hash or (forecast_hash is not None and sha(forecast_path) != forecast_hash):
        raise ValueError("old seal or forecast changed during scoped read")
    receipt = dict(stream_index=index, shot=sid, disruptive=disruptive, selected=len(selected),
                   raw_sha256=expected, size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                   seal_sha256=seal_hash, curve_sha256=curve_hash, forecast_sha256=forecast_hash)
    return signal, selected, native, receipt


def extract_case(base, signal, case, sid, native_curve):
    maturity = case["arrival"]
    if maturity != case["origin"] + 8 or case["future"] != maturity + 24:
        raise ValueError("common maturity and fixed 24-sample future endpoint required")
    x, available = window_batch(signal, [maturity], device="cpu")
    with torch.no_grad():
        z = base(x, available)["logit"]
        normalized = levels(base, x, available)[0, -32:, list(CHANNEL_INDICES)].cpu().numpy()
    mask = available[0, -32:, list(CHANNEL_INDICES)].cpu().numpy()
    order = bin_permutation(sid, maturity)
    motion = motion_features(normalized, mask, order)
    native_logit = float(z)
    native_probability = float(z.sigmoid())
    parity = abs(native_probability - float(native_curve[maturity]))
    if parity > 5e-4:
        raise ValueError("native logit does not match original sealed curve")
    endpoint_mask = mask[-1].copy()
    endpoint = np.where(endpoint_mask, normalized[-1], 0.0).astype(np.float64)
    return dict(shot=sid, **case, maturity=maturity, native_logit=native_logit, native_parity_abs=parity,
                endpoint=endpoint, endpoint_mask=endpoint_mask, permutation=order, **motion)


def design(rows, arm):
    if arm not in ARMS[1:]:
        raise ValueError("a fitted diagnostic arm is required")
    matrix = []
    for row in rows:
        base = [row["endpoint"], row["endpoint_mask"].astype(float), row["common"].astype(float)]
        extra = {
            "no_motion": [], "short": [row["short"]], "long": [row["long"]],
            "short_long": [row["short"], row["long"]], "permuted_long": [row["permuted"]],
            "short_permuted": [row["short"], row["permuted"]],
        }[arm]
        matrix.append(np.concatenate(base + extra))
    return np.asarray(matrix, dtype=np.float64)


def summarize(rows, logits):
    labels = np.asarray([r["label"] for r in rows], dtype=int)
    weights = balanced_weights(rows)
    strata = np.asarray([r["group"] for r in rows])
    bce = {arm: np.logaddexp(0.0, value) - labels * value for arm, value in logits.items()}
    metrics = {}
    for arm in ARMS:
        aucs = {}
        for name, mask in (("balanced_all", np.ones(len(rows), bool)),
                           ("safe_vs_positive", strata != "earlier"),
                           ("earlier_vs_positive", strata != "safe")):
            aucs[name] = weighted_auc(labels[mask], logits[arm][mask], weights[mask])
        metrics[arm] = dict(weighted_bce=float(np.dot(weights, bce[arm])), auc=aucs)
    groups = {}
    for group in GROUPS:
        subset = [i for i, row in enumerate(rows) if row["group"] == group]
        ids = list(dict.fromkeys(rows[i]["shot"] for i in subset))
        value = dict(shots=len(ids), cases=len(subset))
        for a, b in COMPARISONS:
            differences = np.asarray([np.mean([bce[a][i] - bce[b][i] for i in subset if rows[i]["shot"] == sid]) for sid in ids])
            sd = float(differences.std(ddof=1)) if len(differences) > 1 else None
            value[f"{a}_vs_{b}"] = dict(mean_delta=float(differences.mean()) if len(differences) else None,
                                         shot_se=sd / math.sqrt(len(differences)) if sd is not None else None,
                                         improved=int((differences < 0).sum()), worse=int((differences > 0).sum()),
                                         ties=int((differences == 0).sum()))
        groups[group] = value
    return dict(arms=metrics, groups=groups)


def _cell(pair):
    if pair["mean_delta"] is None:
        return "—"
    mean = f"{pair['mean_delta']:+.5f}"
    if pair["shot_se"] is not None:
        mean += f" ± {pair['shot_se']:.5f}"
    return f"{mean}; {pair['improved']}/{pair['worse']}/{pair['ties']}"


def _motion_coverage(rows):
    common = np.stack([row["common"] for row in rows])
    short = np.stack([row["short_supported"] for row in rows])
    long = np.stack([row["long_supported"] for row in rows])
    actual = np.stack([row["long"] for row in rows])[common]
    shuffled = np.stack([row["permuted"] for row in rows])[common]
    corr = float(np.corrcoef(actual, shuffled)[0, 1]) if len(actual) > 1 and actual.std() > 0 and shuffled.std() > 0 else None
    return dict(shared_coordinate_fraction=float(common.mean()), short_coordinate_fraction=float(short.mean()),
                long_coordinate_fraction=float(long.mean()), no_shared_cases=int((~common.any(axis=1)).sum()),
                identical_permutation_cases=sum(np.array_equal(row["permutation"], np.arange(4)) for row in rows),
                actual_vs_permuted_trend_correlation=corr,
                mean_abs_trend_difference=float(np.mean(np.abs(actual - shuffled))) if len(actual) else None)


def make_report(summary, plan):
    lines = ["# 持续轨迹运动：固定 warm20 / 封存80 离线诊断", "",
             "## Material Passport", "", "类型：探索性离线机制诊断；非新的已部署风险头、非在线评估、非独立验证。"
             "来源：固定原流及 v5 HDF 结局／时钟；每炮先核验 seal、曲线及 raw SHA，再核对 v5 后采样。", "",
             f"训练 {summary['fit']['shots']} 炮/{summary['fit']['cases']} 例；后 warm 评估 {summary['held']['shots']} 炮/{summary['held']['cases']} 例。"
             "炮不按结局或模型分数筛选；case 按已结束 HDF 真值选取有效 o+32，正例提前量严格大于 30 ms，因而不是自然在线采样。", "",
             "共同成熟时刻 t=o+8；只读截至 t 的末 8/32 个样本，各分 4 个等长区间，"
             "用有效观测的区间中位数、相邻差的中位数与符号一致比例构造每样本趋势；预测端点固定 t+24。"
             "每个区间至少一半有观测，8/32 共用同一支持掩码；短窗每箱仅两个点，中位数的抗异常能力有限。", "",
             "所有拟合臂共享冻结 native logit（固定 offset）、t 时刻源坐标水平、该时刻掩码和共同支持掩码；"
             "只增加各臂所述趋势。训练仅用 warm20，组→炮→窗等权、相同 L2=0.1 和优化器；"
             "后80标签未用于拟合或调参。单次分箱顺序置乱保留每箱中位数和掩码，不是显著性检验。", "",
             "真实 RAW23 名称沿用；MNM 和 mode_number_n 未作为运动通道，ne_nG 是已有归一化密度诊断，经既有数值映射后进入特征，不推算原始密度。", ""]
    for section, label in (("fit", "warm20 拟合内"), ("held", "封存后80炮开发留出")):
        s = summary[section]
        coverage = s["coverage"]
        lines += [f"## {label}", "",
                  f"共同支持坐标比例 {coverage['shared_coordinate_fraction']:.3f}；无共同支持 {coverage['no_shared_cases']} 例；"
                  f"实序与乱序长窗趋势相关 {coverage['actual_vs_permuted_trend_correlation']}；"
                  f"置乱恰为原顺序 {coverage['identical_permutation_cases']} 例。", "",
                  "| 臂 | 组→炮→窗等权 BCE | 点级平衡 AUC | safe vs positive AUC | earlier vs positive AUC |",
                  "|---|---:|---:|---:|---:|"]
        for arm in ARMS:
            m = s["metrics"]["arms"][arm]
            lines.append(f"| {arm} | {m['weighted_bce']:.6f} | {m['auc']['balanced_all']:.5f} | {m['auc']['safe_vs_positive']:.5f} | {m['auc']['earlier_vs_positive']:.5f} |")
        lines += ["", "ΔBCE 为左臂减右臂，负数有利于左臂；先同炮同组平均，± 为炮间标准误，后为改善/恶化/持平炮数。", "",
                  "| 组（炮/例） | short−no_motion | long−no_motion | long−permuted | short+long−short | short+long−short+permuted |",
                  "|---|---:|---:|---:|---:|---:|"]
        for group in GROUPS:
            g = s["metrics"]["groups"][group]
            names = ("short_vs_no_motion", "long_vs_no_motion", "long_vs_permuted_long", "short_long_vs_short", "short_long_vs_short_permuted")
            lines.append(f"| {group} ({g['shots']}/{g['cases']}) | " + " | ".join(_cell(g[name]) for name in names) + " |")
        lines.append("")
    lines += ["## 解释与来源核验", "",
              "AUC 是筛选后时间点的加权区分度，不是逐炮报警表现；同一破裂炮可能同时出现在 earlier 和 positive，"
              "拟合炮只有 20 个且正例炮更少，所有估计均需独立炮集和不同运行条件复核。", "",
              "[Zheng et al., Communications Physics 6, 181 (2023)](https://www.nature.com/articles/s42005-023-01296-9) "
              "讨论 J-TEXT 到 EAST 的迁移及不同时间尺度；原文提及 ‘different temporal features with different time scales’。"
              "[Montes et al., Nuclear Fusion 61, 026022 (2021)](https://doi.org/10.1088/1741-4326/abcdb9) "
              "在[作者单位公开稿](https://dspace.mit.edu/entities/publication/e800cf5e-03cf-4caf-a394-af232f5bce5b)中描述从放电中 ‘detecting distinct events’。"
              "两篇仅支持检查时序／事件演化的动机，不证明本探针的 8/32 点尺度、特征或结果。"
              "来源文字由当前会话核对；无可用独立质疑子代理，不能冒称独立复核。", ""]
    return "\n".join(lines)


def run():
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite the probe or report")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    paths = dict(run_protocol=RUN / "protocol.json", v5_completed=V5 / "completed.json",
                 v5_audit=V5 / "audit.json", v5_metadata=V5 / "hdf5_metadata.json",
                 subspace_completed=SUBSPACE / "completed.json", subspace_warm_cases=SUBSPACE / "warm_cases.json",
                 subspace_test_cases=SUBSPACE / "cases.json", map=MAP, raw_hashes=RAW_HASHES, source=DEFAULT_PARENT)
    hashes = {key: sha(path) for key, path in paths.items()}
    if hashes != EXPECTED:
        raise ValueError("bound original/v5/source/case input changed")
    v5_done, v5_audit = read_json(paths["v5_completed"]), read_json(paths["v5_audit"])
    sub_done = read_json(paths["subspace_completed"])
    if (v5_done["status"] != "PASS" or v5_audit["status"] != "PASS"
            or v5_done["artifact_sha256"]["hdf5_metadata.json"] != hashes["v5_metadata"]
            or v5_done["artifact_sha256"]["audit.json"] != hashes["v5_audit"]
            or sub_done["artifact_sha256"]["warm_cases.json"] != hashes["subspace_warm_cases"]
            or sub_done["artifact_sha256"]["cases.json"] != hashes["subspace_test_cases"]):
        raise ValueError("v5 or previous case receipt mismatch")
    original, mapping, raw_hashes = read_json(paths["run_protocol"]), read_json(MAP), read_json(RAW_HASHES)
    order = [row["shot"] for row in original["order"]]
    if (len(order) != 360 or len(set(order)) != 360 or mapping["calibration_ids"] != order[:20]
            or original["input_sha256"][str(MAP)] != hashes["map"]
            or original["input_sha256"][str(RAW_HASHES)] != hashes["raw_hashes"]):
        raise ValueError("fixed original stream or map mismatch")
    metadata_rows = read_json(paths["v5_metadata"])["EAST"]
    metadata = {row["shot"]: row for row in metadata_rows}
    if len(metadata) != len(metadata_rows) or any(sid not in metadata for sid in order[:100]):
        raise ValueError("v5 identities missing or duplicated")
    code = {name: sha(ROOT / name) for name in CODE_PATHS}
    plan = fixed_plan(order[:20], order[20:100], hashes)
    plan.update(input_paths={key: str(path) for key, path in paths.items()}, code_sha256=code,
                source_state_sha256=original["source_state_sha256"], numpy_version=np.__version__,
                scipy_version=scipy.__version__, torch_version=torch.__version__)
    OUTPUT.mkdir(parents=False, exist_ok=False)
    write_json(OUTPUT / "protocol.json", plan)
    print("fixed protocol sealed; reading first20 only for fit", flush=True)
    base, parent = load_parent(DEFAULT_PARENT)
    base.eval().requires_grad_(False)
    source_before = state_digest(base)
    if source_before != original["source_state_sha256"] or float(parent["threshold"]) != float(original["threshold"]):
        raise ValueError("frozen native source changed")
    fit_rows, held_rows, receipts = [], [], []
    for index, sid in enumerate(order[:20], 1):
        signal, selected, native, receipt = _read_sealed(index, sid, mapping, raw_hashes, metadata)
        fit_rows.extend(dict(stream_index=index, **extract_case(base, signal, case, sid, native)) for case in selected)
        receipts.append(receipt)
        print(f"fit sealed shots {index}/20; cases {len(fit_rows)}", flush=True)
    previous_train = read_json(paths["subspace_warm_cases"])
    identity = lambda rows: [(row["shot"], row["origin"], row["group"], row["label"]) for row in rows]
    if identity(fit_rows) != identity(previous_train):
        raise ValueError("warm coordinates differ from fixed previous probe")
    weights = balanced_weights(fit_rows)
    labels = np.asarray([row["label"] for row in fit_rows], dtype=np.float64)
    offsets = np.asarray([row["native_logit"] for row in fit_rows], dtype=np.float64)
    models = {arm: fit_readout(design(fit_rows, arm), offsets, labels, weights) for arm in ARMS[1:]}
    fit_logits = {"native": offsets}
    fit_logits.update({arm: predict_readout(model, design(fit_rows, arm), offsets) for arm, model in models.items()})
    print("all readouts frozen; reading sealed 21..100 for evaluation only", flush=True)
    for index, sid in enumerate(order[20:100], 21):
        signal, selected, native, receipt = _read_sealed(index, sid, mapping, raw_hashes, metadata)
        held_rows.extend(dict(stream_index=index, **extract_case(base, signal, case, sid, native)) for case in selected)
        receipts.append(receipt)
        print(f"held sealed shots {index - 20}/80; cases {len(held_rows)}", flush=True)
    previous_test = read_json(paths["subspace_test_cases"])
    if identity(held_rows) != identity(previous_test):
        raise ValueError("held-out coordinates differ from fixed previous probe")
    held_offsets = np.asarray([row["native_logit"] for row in held_rows], dtype=np.float64)
    held_logits = {"native": held_offsets}
    held_logits.update({arm: predict_readout(model, design(held_rows, arm), held_offsets) for arm, model in models.items()})
    if state_digest(base) != source_before or any(p.grad is not None or p.requires_grad for p in base.parameters()):
        raise AssertionError("frozen native model changed")
    if any(sha(path) != hashes[key] for key, path in paths.items()) or any(sha(ROOT / name) != value for name, value in code.items()):
        raise ValueError("bound inputs or code changed during the probe")
    summary = {}
    for name, rows, logits, nshots in (("fit", fit_rows, fit_logits, 20), ("held", held_rows, held_logits, 80)):
        summary[name] = dict(shots=nshots, cases=len(rows), metrics=summarize(rows, logits), coverage=_motion_coverage(rows),
                             native_parity_max_abs=max(row["native_parity_abs"] for row in rows))
    arrays = {}
    for prefix, rows, logits in (("fit", fit_rows, fit_logits), ("held", held_rows, held_logits)):
        for field in ("shot", "origin", "maturity", "future", "label", "native_logit", "endpoint", "endpoint_mask",
                      "common", "short_supported", "long_supported", "short", "long", "permuted", "permutation", "long_bins", "permuted_bins"):
            arrays[f"{prefix}_{field}"] = np.stack([row[field] for row in rows])
        for arm, value in logits.items():
            arrays[f"{prefix}_logit_{arm}"] = value
    with (OUTPUT / "features_and_logits.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    fit_receipt = {arm: {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in model.items()}
                   for arm, model in models.items()}
    serial_rows = []
    for stage, rows in (("fit", fit_rows), ("held", held_rows)):
        for row in rows:
            serial_rows.append(dict(stage=stage, **{key: value for key, value in row.items()
                if key not in ("endpoint", "endpoint_mask", "permutation", "short", "long", "permuted", "common", "long_bins", "permuted_bins", "short_supported", "long_supported")}))
    for name, value in (("cases.json", serial_rows), ("shots.json", receipts), ("fit.json", fit_receipt), ("summary.json", summary)):
        write_json(OUTPUT / name, value)
    with REPORT.open("x", encoding="utf-8") as stream:
        stream.write(make_report(summary, plan))
    artifacts = ("protocol.json", "features_and_logits.npz", "cases.json", "shots.json", "fit.json", "summary.json")
    completed = dict(status="COMPLETED_PAST_ONLY_PERSISTENT_MOTION_CPU_PROBE", script_sha256=sha(__file__),
                     report_sha256=sha(REPORT), artifact_sha256={name: sha(OUTPUT / name) for name in artifacts})
    write_json(OUTPUT / "completed.json", completed)
    print(json.dumps(completed), flush=True)


if __name__ == "__main__":
    run()
