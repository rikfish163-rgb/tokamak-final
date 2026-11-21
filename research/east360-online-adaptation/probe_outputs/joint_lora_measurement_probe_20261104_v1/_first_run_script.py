"""Sealed shots 21..100: finite-step measurement-gradient decomposition.

Offline development probe, not an online policy or independent validation.
Every candidate starts from the same frozen warm20 state. No hyperparameters
are fit and no state is carried between candidates or shots.
"""

from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"
sys.dont_write_bytecode = True

import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.direct_fast_memory_20261104.timeline import window_batch
from experiments.risk_aligned_online.joint_lora_20261104.timeline import _project, _targets
from experiments.risk_aligned_online.joint_lora_resume_20261104.state import restore_warm
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import prepare_actual
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from joint_lora_gradient_probe_20261104 import sha, state_digest


ROOT = io23.ROOT
RUN = ROOT / "probe_outputs/joint_lora_paired360_20261104_v1"
MAP = ROOT / "probe_outputs/stratified_cold20_context360_v1/input_map.json"
RAW_HASHES = ROOT / "probe_outputs/stratified_cold20_context360_v1/input_hashes.json"
OUTPUT = ROOT / "probe_outputs/joint_lora_measurement_probe_20261104_v1"
REPORT = ROOT / "JOINT_LORA_MEASUREMENT_PROBE_RESULT_20261104.md"
EXPECTED = {
    "run_protocol": "5fe569459f9bb533d50b95adc9c7390585d2cfa846fb4493d39b589ff850a835",
    "warm_json": "2e908bc822d633c2ec009548317467e30f6a1cb03cf1e5b48873d812035d8e09",
    "warm_pt": "d237001ff8febf82f17eb974850c48db115fd81f212d7695b78bc76b078407b5",
    "map": "43c37ad981261e059ac8bf8d11d89324bad375dd7405decc194c51e53614aff2",
    "raw_hashes": "c919aa60bd251eaaa5a7e43ef1ed8fa5d0e867f6552db0d53871ccd1e4ad5a86",
    "source": PARENT_SHA256,
}
WARM_STATE = "69a3b183a5ce708293cfc23ff4871db5e2e9c29b8c0fa7f174ccf1b193033582"
ARMS = ("none", "real", "hold", "reverse", "motion", "anti_motion")
GROUPS = ("safe", "earlier", "positive")
NEAR_HALF_WIDTH = 0.05  # Fixed probability band, not fitted to outcomes.
CODE_PATHS = (
    "joint_lora_measurement_probe_20261104.py",
    "joint_lora_gradient_probe_20261104.py",
    "experiments/risk_aligned_online/io23.py",
    "experiments/risk_aligned_online/simple_data.py",
    "experiments/risk_aligned_online/causal_input.py",
    "experiments/risk_aligned_online/simple23.py",
    "experiments/risk_aligned_online/joint_lora_20261104/model.py",
    "experiments/risk_aligned_online/joint_lora_20261104/timeline.py",
    "experiments/risk_aligned_online/joint_lora_resume_20261104/state.py",
    "experiments/risk_aligned_online/cold20_compare_20261104/adapters.py",
    "experiments/risk_aligned_online/direct_fast_memory_20261104/model.py",
    "experiments/risk_aligned_online/direct_fast_memory_20261104/memory.py",
    "experiments/risk_aligned_online/direct_fast_memory_20261104/timeline.py",
    "experiments/risk_aligned_online/mask_repair_20261104/east_evaluation.py",
    "experiments/risk_aligned_online/warmstart_alignment_20261022/alignment.py",
    "experiments/risk_aligned_online/redesign_20261004/prepare_prior.py",
)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def select_cases(times, disruptive, down):
    """Uniformly select <=3 valid deployment origins, independent of scores."""
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or (len(times) > 1 and not (np.diff(times) > 0).all()):
        raise ValueError("finite strictly increasing clock required")
    if type(disruptive) is not bool or (disruptive and (down is None or not math.isfinite(down))) or (not disruptive and down is not None):
        raise ValueError("valid revealed HDF truth required")
    labels, valid = io23.labels_at(times, disruptive, down, 125.0)
    candidates = []
    for origin in range(127, len(times) - 32, 32):
        future = origin + 32
        if not valid[future] or (disruptive and times[future] > down):
            continue
        label = int(labels[future])
        lead = round(float((down - times[future]) * 1000), 9) if disruptive else None
        if label and lead <= 30.0:
            continue
        candidates.append(dict(origin=origin, arrival=origin + 8, future=future,
                               lead_ms=lead, label=label,
                               group="safe" if not disruptive else ("positive" if label else "earlier")))
    if len(candidates) <= 3:
        return candidates
    return [candidates[i] for i in np.linspace(0, len(candidates) - 1, 3, dtype=int)]


def measurement_losses(prediction, target, mask):
    if prediction.shape != target.shape or prediction.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("matching prediction, target and boolean mask required")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("finite prediction and matured target required")
    if not mask.any():
        return None
    denominator = mask.sum()
    real = (prediction - target).square().masked_fill(~mask, 0).sum() / denominator
    hold = prediction.square().masked_fill(~mask, 0).sum() / denominator
    reverse = (prediction + target).square().masked_fill(~mask, 0).sum() / denominator
    motion = -2 * (prediction * target).masked_fill(~mask, 0).sum() / denominator
    return dict(real=real, hold=hold, reverse=reverse, motion=motion, anti_motion=-motion)


def step_from_gradients(parameters, start, gradients):
    """One true PyTorch Adam step from zero moments, independently clipped."""
    if len(parameters) != len(start) or len(parameters) != len(gradients):
        raise ValueError("matching parameter, initial and gradient tuples required")
    if any(not torch.equal(p.detach(), s) for p, s in zip(parameters, start, strict=True)):
        raise ValueError("every arm must start at the same warm state")
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    for parameter, gradient in zip(parameters, gradients, strict=True):
        if gradient.shape != parameter.shape or not torch.isfinite(gradient).all():
            raise ValueError("finite matching gradient required")
        parameter.grad = gradient.detach().clone()
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
    optimizer.step()
    displacement = _project(parameters, start, 0.5)
    return norm, displacement


def validate_seal(seal, sid, curve_hash, forecast_hash, warm_digest):
    if (seal.get("shot") != sid or seal.get("curve_sha256") != curve_hash
            or seal.get("forecast_sha256") != forecast_hash
            or seal.get("current_outcome_used_in_prediction") is not False
            or seal.get("same_slow_for_inter_and_both") is not True
            or not isinstance(seal.get("shot_start_parents"), dict)
            or seal["shot_start_parents"].get("static") != warm_digest):
        raise ValueError(f"invalid old seal or curve/forecast hash for shot {sid}")


def summarize(rows, *, threshold):
    pairs = [(arm, "none") for arm in ARMS if arm != "none"] + [
        ("real", "hold"), ("real", "reverse"), ("motion", "anti_motion"), ("motion", "hold")]

    def comparisons(subset, ids, metric):
        result = {}
        for a, b in pairs:
            values = np.asarray([np.mean([r["arms"][a][metric] - r["arms"][b][metric]
                                          for r in subset if r["shot"] == sid]) for sid in ids], dtype=float)
            std = float(values.std(ddof=1)) if len(values) > 1 else None
            result[f"{a}_vs_{b}"] = dict(
                mean_delta=float(values.mean()) if len(values) else None,
                shot_sd=std, shot_se=std / math.sqrt(len(values)) if std is not None else None,
                improved_shots=int((values < 0).sum()), worse_shots=int((values > 0).sum()),
                tied_shots=int((values == 0).sum()))
        return result

    result = {}
    for group in GROUPS:
        subset = [r for r in rows if r["group"] == group]
        ids = list(dict.fromkeys(r["shot"] for r in subset))
        near = [r for r in subset if abs(r["arms"]["none"]["prob"] - threshold) <= NEAR_HALF_WIDTH]
        near_arms = {}
        for arm in ARMS[1:]:
            differences = [r["arms"][arm]["prob"] - r["arms"]["none"]["prob"] for r in near]
            near_arms[arm] = dict(
                mean_delta_probability=float(np.mean(differences)) if differences else None,
                upward_crossings=sum(r["arms"]["none"]["prob"] < threshold <= r["arms"][arm]["prob"] for r in near),
                downward_crossings=sum(r["arms"][arm]["prob"] < threshold <= r["arms"]["none"]["prob"] for r in near))
        result[group] = dict(
            shots=len(ids), cases=len(subset), no_support_cases=sum(r["supported"] == 0 for r in subset),
            unshifted_bce=comparisons(subset, ids, "bce"),
            threshold_centered_bce=comparisons(subset, ids, "bce_centered"),
            threshold_near=dict(half_width=NEAR_HALF_WIDTH, cases=len(near),
                                shots=len({r["shot"] for r in near}), arms=near_arms))
    return dict(groups=result)


def _gradient_receipt(gradients):
    flat = {name: torch.cat([g.detach().reshape(-1) for g in values]).double()
            for name, values in gradients.items()}
    if any(g.numel() != 512 or not torch.isfinite(g).all() for g in flat.values()):
        raise ValueError("finite 512-coordinate LoRA gradients required")
    result = {}
    for name, error, terms in (
        ("real_minus_hold_motion", flat["real"] - flat["hold"] - flat["motion"], ("real", "hold", "motion")),
        ("reverse_minus_hold_plus_motion", flat["reverse"] - flat["hold"] + flat["motion"], ("reverse", "hold", "motion")),
        ("anti_plus_motion", flat["anti_motion"] + flat["motion"], ("anti_motion", "motion")),
    ):
        maximum = float(error.abs().max())
        norm = float(torch.linalg.vector_norm(error))
        scale = sum(float(torch.linalg.vector_norm(flat[t])) for t in terms)
        relative = norm / max(scale, 1e-30)
        if relative > 2e-4 and maximum > 2e-7:
            raise ValueError(f"gradient decomposition failed: {name}")
        result[name] = dict(max_abs=maximum, l2=norm, relative_l2=relative)
    return result


def _restore(parameters, start):
    with torch.no_grad():
        for parameter, initial in zip(parameters, start, strict=True):
            parameter.copy_(initial)
            parameter.grad = None


def evaluate_case(model, signal, case, forecast, static_curve, threshold):
    parameters = model.neural_parameters()
    start = tuple(p.detach().clone() for p in parameters)
    origin = case["origin"]
    x, available, target, mask = _targets(model, signal, origin, device="cpu")
    supported = int(mask.sum())
    idx = int(np.searchsorted(forecast["origin"], origin))
    if (idx >= len(forecast["origin"]) or int(forecast["origin"][idx]) != origin
            or int(forecast["released_at"][idx]) != origin + 8
            or not np.array_equal(forecast["mask"][idx], mask[0].numpy())):
        raise ValueError("sealed static forecast origin, maturity or mask mismatch")
    target_np = target[0].numpy()
    target_error = float(np.max(np.abs(forecast["target"][idx] - target_np)))
    if not np.allclose(forecast["target"][idx], target_np, rtol=2e-4, atol=2e-5):
        raise ValueError("matured actual target differs from sealed static forecast")
    pred = model(x, available)["diagnostic"]
    pred_np = pred.detach().numpy()[0]
    prediction_error = float(np.max(np.abs(forecast["prediction"][idx] - pred_np)))
    if not np.allclose(forecast["prediction"][idx], pred_np, rtol=3e-4, atol=3e-5):
        raise ValueError("warm diagnostic differs from sealed static prediction")
    losses = measurement_losses(pred, target, mask)
    states, step_info = {"none": start}, {"none": dict(grad_norm=None, displacement=0.0)}
    gradient_errors = None
    loss_values = None
    if losses is not None:
        gradients = {}
        for i, (name, loss) in enumerate(losses.items()):
            gradients[name] = torch.autograd.grad(loss, parameters, retain_graph=i < len(losses) - 1)
        gradient_errors = _gradient_receipt(gradients)
        loss_values = {name: float(loss.detach()) for name, loss in losses.items()}
        for name in ARMS[1:]:
            _restore(parameters, start)
            grad_norm, displacement = step_from_gradients(parameters, start, gradients[name])
            states[name] = tuple(p.detach().clone() for p in parameters)
            step_info[name] = dict(grad_norm=grad_norm, displacement=displacement)
    else:
        for name in ARMS[1:]:
            states[name] = start
            step_info[name] = dict(grad_norm=None, displacement=0.0)

    # The query and its actual BCE are consulted only after all candidate
    # updates have been constructed from origin + eight matured samples.
    xr, ar = window_batch(signal, [case["future"]], device="cpu")
    tau = math.log(threshold / (1 - threshold))
    arms = {}
    for name in ARMS:
        _restore(parameters, states[name])
        with torch.no_grad():
            logit = model(xr, ar)["logit"]
            desired = torch.full_like(logit, float(case["label"]))
            arms[name] = dict(
                bce=float(F.binary_cross_entropy_with_logits(logit, desired)),
                bce_centered=float(F.binary_cross_entropy_with_logits(logit - tau, desired)),
                logit=float(logit), prob=float(logit.sigmoid()), **step_info[name])
    _restore(parameters, start)
    static_error = abs(arms["none"]["prob"] - float(static_curve[case["future"]]))
    if static_error > 5e-4:
        raise ValueError("warm no-update probability differs from the sealed static curve")
    return dict(case, supported=supported, no_op=supported == 0, diagnostic_losses=loss_values,
                gradient_decomposition=gradient_errors, arms=arms,
                parity=dict(target_max_abs=target_error, prediction_max_abs=prediction_error,
                            static_probability_abs=static_error))


def _pair_cell(pair):
    if pair["mean_delta"] is None:
        return "—"
    return (f"{pair['mean_delta']:+.7f} ± {pair['shot_se']:.7f}; "
            f"{pair['improved_shots']}/{pair['worse_shots']}/{pair['tied_shots']}") if pair["shot_se"] is not None else (
                f"{pair['mean_delta']:+.7f}; {pair['improved_shots']}/{pair['worse_shots']}/{pair['tied_shots']}")


def make_report(summary, protocol):
    lines = [
        "# JointLoRA 诊断运动梯度分解：80 炮开发探针", "",
        "本报告只使用原 paired360 已封存流索引 21–100；这 80 炮不是 warm20，选炮不依结局。"
        "这是离线开发探针，不是独立验证，也不是部署策略或炮级 TPR。", "",
        f"总计 {summary['shots_processed']} 炮、{summary['cases']} 个固定坐标；无候选 {summary['shots_without_cases']} 炮，"
        f"无诊断支持 {summary['no_support_cases']} 例（保留为所有臂的 no-op）。", "",
        "每例从同一 warm20 模型出发，o=127+32k，已成熟目标为 o+1..o+8 相对 o 的实际增量，"
        "在 source asinh 归一化坐标计算且受观测掩码约束；仅在更新构造后查询 o+32 的 HDF 真值标签与模型风险。"
        "每个臂独立执行一次 clip=1、lr=1e-4、零动量 Adam，LoRA 512 参数，半径 0.5；不拟合模型或超参。", "",
        "表格为逐炮先平均该组内各 case、再对炮等权：ΔBCE = 左臂 − 右臂，负值表示左臂较好。"
        "误差项为炮间标准误；后续三数为逐炮改善/恶化/精确持平。不同风险组的炮可重叠。", "",
    ]
    for key, title in (("unshifted_bce", "原始模型 logit 的实际 BCE（主结果）"),
                       ("threshold_centered_bce", "减去固定阈值 logit 后的实际 BCE（与旧探针对照）")):
        lines += [f"## {title}", "", "| 组 | 炮/例/no-op | real−none | hold−none | reverse−none | motion−none | anti−none |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for group in GROUPS:
            g = summary["groups"][group]
            c = g[key]
            lines.append(f"| {group} | {g['shots']}/{g['cases']}/{g['no_support_cases']} | " + " | ".join(_pair_cell(c[f"{a}_vs_none"]) for a in ARMS[1:]) + " |")
        lines += ["", "| 组 | real−hold | real−reverse | motion−anti_motion | motion−hold |",
                  "|---|---:|---:|---:|---:|"]
        for group in GROUPS:
            c = summary["groups"][group][key]
            lines.append(f"| {group} | " + " | ".join(_pair_cell(c[p]) for p in ("real_vs_hold", "real_vs_reverse", "motion_vs_anti_motion", "motion_vs_hold")) + " |")
        lines.append("")
    lines += ["## 固定阈值邻近点（不是炮级 TPR）", "",
              f"事先固定无更新概率与阈值 {protocol['threshold']:.9f} 相差不超过 {NEAR_HALF_WIDTH:.2f} 的点；"
              "上穿/下穿是这些所选时间点的计数，不是独立炮数或整炮报警。", "",
              "| 组 | 邻近点/炮 | real 上/下 | hold 上/下 | reverse 上/下 | motion 上/下 | anti 上/下 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for group in GROUPS:
        near = summary["groups"][group]["threshold_near"]
        cells = [f"{near['arms'][arm]['upward_crossings']}/{near['arms'][arm]['downward_crossings']}" for arm in ARMS[1:]]
        lines.append(f"| {group} | {near['cases']}/{near['shots']} | " + " | ".join(cells) + " |")
    errors = summary["decomposition_error"]
    lines += ["", "## 身份与解释边界", "",
              f"各炮在读取 HDF 前核验原 seal、曲线与 forecast SHA，随后核验该炮固定 raw HDF SHA；"
              f"封存模型、时间轴、目标和静态概率均已核对。数据与脚本哈希见 protocol/completed。", "",
              f"梯度分解最大绝对误差：real−hold−motion={errors['real_minus_hold_motion']['max_abs']:.3g}，"
              f"reverse−hold+motion={errors['reverse_minus_hold_plus_motion']['max_abs']:.3g}。"
              "梯度恒等式不代表分别裁剪并通过 Adam 后的参数步长也线性相加。", "",
              "每例结果和每炮身份见 cases.json 和 shots.json；本报告不推断总体显著性，也不据此选择超参数。", ""]
    return "\n".join(lines)


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run():
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite the probe directory or report")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    paths = dict(run_protocol=RUN / "protocol.json", warm_json=RUN / "warm.json", warm_pt=RUN / "warm.pt",
                 map=MAP, raw_hashes=RAW_HASHES, source=DEFAULT_PARENT)
    hashes = {name: sha(path) for name, path in paths.items()}
    if hashes != EXPECTED:
        raise ValueError("original run/warm/map/source receipt mismatch")
    original = read_json(paths["run_protocol"])
    mapping, raw_hashes, warm = read_json(MAP), read_json(RAW_HASHES), read_json(paths["warm_json"])
    order = [row["shot"] for row in original["order"]]
    if (len(order) != 360 or len(set(order)) != 360 or warm["label_shots"] != order[:20]
            or mapping["calibration_ids"] != order[:20] or warm["fitted_state_sha256"] != WARM_STATE
            or original["input_sha256"][str(MAP)] != hashes["map"]
            or original["input_sha256"][str(RAW_HASHES)] != hashes["raw_hashes"]
            or original["source_sha256"] != PARENT_SHA256
            or original["intra_parameters"] != 512 or original["intra_lr"] != 1e-4
            or original["intra_radius"] != 0.5):
        raise ValueError("original order, warm, raw-hash, or step contract changed")
    code = {name: sha(ROOT / name) for name in CODE_PATHS}
    for name in CODE_PATHS[2:]:
        historical = original["code_sha256"].get(str(ROOT / name))
        if historical is not None and code[name] != historical:
            raise ValueError(f"original bound code changed: {name}")
    base, parent = load_parent(DEFAULT_PARENT)
    base.eval().requires_grad_(False)
    source_digest = state_digest(base)
    threshold = float(parent["threshold"])
    if source_digest != original["source_state_sha256"] or threshold != original["threshold"] or not 0 < threshold < 1:
        raise ValueError("source state or threshold changed")
    model = restore_warm(base, paths["warm_pt"], warm, device="cpu")
    parameters = model.neural_parameters()
    if sum(p.numel() for p in parameters) != 512:
        raise ValueError("exactly 512 LoRA parameters required")
    for parameter in parameters:
        parameter.requires_grad_(True)
    if {id(p) for p in parameters} != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError("only LoRA parameters may receive gradients")
    cases, shots = [], []
    selected_ids = order[20:100]
    for index, sid in enumerate(selected_ids, 21):
        prefix = f"{index:04d}_{sid}"
        seal_path = RUN / "shots" / f"{prefix}_seal.json"
        curve_path = RUN / "shots" / f"{prefix}.npz"
        forecast_path = RUN / "shots" / f"{prefix}_forecasts.npz"
        seal = read_json(seal_path)
        seal_hash, curve_hash, forecast_hash = sha(seal_path), sha(curve_path), sha(forecast_path)
        validate_seal(seal, sid, curve_hash, forecast_hash, WARM_STATE)
        with np.load(curve_path, allow_pickle=False) as file:
            old_times, static_curve = file["times"].copy(), file["static"].copy()
        with np.load(forecast_path, allow_pickle=False) as file:
            forecast = {name: file[f"static__{name}"].copy() for name in ("origin", "released_at", "prediction", "target", "mask")}
        hdf = io23.path_for("EAST", sid)
        before = hdf.stat()
        expected_raw = raw_hashes[str(sid)]
        if sha(hdf) != expected_raw:
            raise ValueError(f"raw HDF SHA mismatch: {sid}")
        reader = io23.read_signals
        def only_current(device, requested, *args, **kwargs):
            if device != "EAST" or requested != sid:
                raise ValueError("only the current sealed shot is authorized")
            return reader(device, requested, *args, **kwargs)
        with patch.object(io23, "read_signals", side_effect=only_current):
            raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
        if not np.array_equal(raw["times"], old_times) or len(static_curve) != len(old_times):
            raise ValueError("original sealed curve clock changed")
        # The curve and forecast have already passed their old seals here.
        disruptive, down = io23.read_outcome("EAST", sid)
        signal = prepare_actual(raw, index, mapping)
        selected = select_cases(raw["times"], disruptive, down)
        before_count = len(cases)
        for case in selected:
            result = evaluate_case(model, signal, case, forecast, static_curve, threshold)
            cases.append(dict(shot=sid, stream_index=index, disruptive=disruptive, **result))
        after = hdf.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha(hdf) != expected_raw:
            raise ValueError(f"HDF changed during shot {sid}")
        if sha(seal_path) != seal_hash or sha(curve_path) != curve_hash or sha(forecast_path) != forecast_hash:
            raise ValueError(f"old seal or artifacts changed during shot {sid}")
        if state_digest(model) != WARM_STATE or state_digest(base) != source_digest or any(p.grad is not None for p in model.parameters()):
            raise AssertionError("probe update escaped the frozen starting state")
        shots.append(dict(stream_index=index, shot=sid, disruptive=disruptive, cases=len(cases) - before_count,
                          no_support_cases=sum(r["supported"] == 0 for r in cases[before_count:]),
                          raw_sha256=expected_raw, size=before.st_size, mtime_ns=before.st_mtime_ns,
                          seal_sha256=seal_hash, curve_sha256=curve_hash, forecast_sha256=forecast_hash))
        print(f"verified sealed shots {len(shots)}/80; retained cases {len(cases)}", flush=True)
    if len(shots) != 80 or [row["shot"] for row in shots] != selected_ids:
        raise AssertionError("the fixed 80-shot stream selection changed")
    if any(sha(path) != hashes[name] for name, path in paths.items()) or any(sha(ROOT / name) != value for name, value in code.items()):
        raise ValueError("bound inputs or code changed during the probe")
    summary = summarize(cases, threshold=threshold)
    error_names = ("real_minus_hold_motion", "reverse_minus_hold_plus_motion", "anti_plus_motion")
    measured = [r for r in cases if r["gradient_decomposition"] is not None]
    summary.update(shots_processed=len(shots), cases=len(cases),
                   shots_without_cases=sum(row["cases"] == 0 for row in shots),
                   no_support_cases=sum(row["supported"] == 0 for row in cases),
                   decomposition_error={name: {metric: max((r["gradient_decomposition"][name][metric] for r in measured), default=0.0)
                                               for metric in ("max_abs", "l2", "relative_l2")} for name in error_names},
                   parity_max_abs={name: max((r["parity"][name] for r in cases), default=0.0)
                                   for name in ("target_max_abs", "prediction_max_abs", "static_probability_abs")})
    protocol = dict(scope="SEALED_STREAM21_100_FROZEN_WARM_MEASUREMENT_DEVELOPMENT_PROBE",
                    input_sha256=hashes, input_paths={k: str(v) for k, v in paths.items()}, code_sha256=code,
                    stream_indexes=[21, 100], shots=selected_ids, warm_state_sha256=WARM_STATE,
                    source_state_sha256=source_digest, threshold=threshold,
                    device="cpu", torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                    selection="o=127+32k; q=o+32 in record/preterminal; positives have lead>30ms; select <=3 uniform eligible indices",
                    target="o+1..o+8 actual source-asinh-level minus actual level at o; both endpoints observed; 21 channels",
                    update="fresh Adam each arm/case, lr=1e-4, clip=1, 512 LoRA coordinates, radius=.5, no carried state",
                    loss="real=(p-y)^2/D; hold=p^2/D; reverse=(p+y)^2/D; motion=-2*p*y/D; anti=-motion; mask applies",
                    primary_bce="unshifted logits", secondary_bce="logits minus fixed threshold logit",
                    threshold_near_half_width=NEAR_HALF_WIDTH, independent_validation=False, online_policy_tested=False,
                    outcome_revealed_after_each_old_seal=True, fitting_or_hyperparameter_search=False)
    report = make_report(summary, protocol)
    OUTPUT.mkdir(parents=False, exist_ok=False)
    for name, value in (("cases.json", cases), ("shots.json", shots), ("summary.json", summary), ("protocol.json", protocol)):
        write_json(OUTPUT / name, value)
    with REPORT.open("x", encoding="utf-8") as stream:
        stream.write(report)
    completed = dict(status="COMPLETED_SEALED80_CPU_MEASUREMENT_PROBE", input_sha256=hashes,
                     script_sha256=sha(__file__), report_sha256=sha(REPORT),
                     artifact_sha256={name: sha(OUTPUT / name) for name in ("cases.json", "shots.json", "summary.json", "protocol.json")})
    write_json(OUTPUT / "completed.json", completed)
    print(json.dumps(completed), flush=True)


if __name__ == "__main__":
    run()
