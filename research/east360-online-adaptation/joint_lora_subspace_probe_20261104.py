"""Warm20-only Q/V input-subspace fit; sealed 21..100 B-only CPU probe."""

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
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.direct_fast_memory_20261104.timeline import window_batch
from experiments.risk_aligned_online.joint_lora_20261104.timeline import _targets
from experiments.risk_aligned_online.joint_lora_resume_20261104.state import restore_warm
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import prepare_actual
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from joint_lora_gradient_probe_20261104 import sha, state_digest
from joint_lora_measurement_probe_20261104 import (
    WARM_STATE as ORIGINAL_WARM_STATE,
    measurement_losses,
    select_cases,
    step_from_gradients,
    validate_seal,
)


ROOT = io23.ROOT
RUN = ROOT / "probe_outputs/joint_lora_paired360_20261104_v1"
DECODER = ROOT / "probe_outputs/joint_lora_decoder_only_warm_20261104_v1"
MEASUREMENT = ROOT / "probe_outputs/joint_lora_measurement_probe_20261104_v1"
MAP = ROOT / "probe_outputs/stratified_cold20_context360_v1/input_map.json"
RAW_HASHES = ROOT / "probe_outputs/stratified_cold20_context360_v1/input_hashes.json"
OUTPUT = ROOT / "probe_outputs/joint_lora_subspace_probe_20261104_v1"
REPORT = ROOT / "JOINT_LORA_SUBSPACE_PROBE_RESULT_20261104.md"
PARENT_STATE = "087a78d76b2aa906d850ca9ce3acca39e96694938fa4bdeb8a60c437cfc71666"
EXPECTED = {
    "run_protocol": "5fe569459f9bb533d50b95adc9c7390585d2cfa846fb4493d39b589ff850a835",
    "decoder_completed": "0a1e39af577b7688cfccde1e5eae51262646b4d7cf3a5bcc6d720f77733c472e",
    "decoder_warm_json": "668fb38ac7c11da5f075a574deec6d7463620badb5e87f028d83634712d7688d",
    "decoder_warm_pt": "a42e1e3cf03e584792ddd9cd5c1880fb1612f34f1eaf4d62dd5c8651b8f390ea",
    "measurement_completed": "2fea375b67cfee35323a64673af51bd16a436825e3fdb69e71b6bd9a31c7468a",
    "measurement_cases": "a7ec1bbe2d8a157d518896393751dec81f0c1afcedd059d721953acc3be6ad82",
    "map": "43c37ad981261e059ac8bf8d11d89324bad375dd7405decc194c51e53614aff2",
    "raw_hashes": "c919aa60bd251eaaa5a7e43ef1ed8fa5d0e867f6552db0d53871ccd1e4ad5a86",
    "source": PARENT_SHA256,
}
ARMS = ("none", "old_motion", "new_motion", "new_anti_motion", "new_hold", "old_mse", "new_mse")
GROUPS = ("safe", "earlier", "positive")
COMPARISONS = tuple((a, "none") for a in ARMS[1:]) + (
    ("new_motion", "old_motion"), ("new_motion", "new_anti_motion"),
    ("new_motion", "new_hold"), ("new_mse", "old_mse"),
    ("new_mse", "new_hold"), ("new_mse", "new_anti_motion"),
)
CODE_PATHS = (
    "joint_lora_subspace_probe_20261104.py",
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


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def fixed_plan(train_ids, test_ids, bindings):
    if (len(train_ids) != 20 or len(set(train_ids)) != 20 or len(test_ids) != 80
            or len(set(test_ids)) != 80 or set(train_ids) & set(test_ids)):
        raise ValueError("distinct original warm20 and fixed sealed80 required")
    return dict(
        scope="FIXED_WARM20_QV_RANK2_SUBSPACE_SEALED80_DEVELOPMENT_PROBE",
        input_sha256=bindings,
        train_indices=[1, 20], train_shots=train_ids,
        test_indices=[21, 100], test_shots=test_ids,
        selection="Per shot o=127+32k; o+8 matured; o+32 in record/preterminal; HDF 125ms labels; positives lead strictly >30ms; <=3 uniformly spaced eligible indices; no score/gradient selection",
        train_uses_original_packed_gradient=True,
        train_objective="D=grad_W[-2 sum(p*y*mask)/Dobs]; R=grad_W[BCE(logit(o+32)-logit(threshold), HDF label)] on original last-layer packed W; only Q and V 64x64 blocks",
        covariance="C=(D.T R + R.T D)/(2 ||D||F ||R||F), separately Q/V; zero norm is zero contribution retaining its group/shot/window weight; average equally by group, then shot, then window",
        basis="Top two eigenvectors per symmetric C, descending; orient each so largest-absolute coordinate is nonnegative. SVD A0=U diag(S) Vh; Anew=U diag(S) E.T; retain A A.T; B stays exactly zero",
        theory="Only an initial B-only SGD directional argument; not a guarantee for independently clipped Adam",
        test_arms=list(ARMS),
        test_update="Fresh Adam on B_q and B_v only (256 parameters), lr=1e-4, clip=1, radius=.5; A frozen; one step per case; no state carry",
        test_labels_fit=False,
        metrics="Actual threshold-centered BCE at o+32; group then per-shot mean delta, shot SE and improvement/worsening/ties; paired comparisons to old A, inverse motion and hold",
        outcome="HDF IsDisrupt and DownTime, only after old per-shot seal and curve/forecast hashes are checked",
        independent_validation=False, online_policy_tested=False,
        hyperparameters_fitted=False, device="cpu", threads=1,
    )


def alignment_matrix(d, r):
    if d.ndim != 2 or d.shape != r.shape or not torch.isfinite(d).all() or not torch.isfinite(r).all():
        raise ValueError("matching finite matrix gradients required")
    d, r = d.detach().double(), r.detach().double()
    nd, nr = torch.linalg.vector_norm(d), torch.linalg.vector_norm(r)
    if nd == 0 or nr == 0:
        return torch.zeros(d.shape[1], d.shape[1], dtype=torch.float64), False
    return (d.T @ r + r.T @ d) / (2 * nd * nr), True


def balanced_weights(rows):
    counts = Counter((row["group"], row["shot"]) for row in rows)
    group_shots = Counter(group for group, _ in counts)
    if set(group_shots) != set(GROUPS):
        raise ValueError("all three risk groups required for fixed group weighting")
    weights = np.asarray([1 / (3 * group_shots[r["group"]] * counts[(r["group"], r["shot"])]) for r in rows], dtype=np.float64)
    if not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0):
        raise AssertionError("balanced training weights do not sum to one")
    return weights


def weighted_alignment(rows, key):
    weights = balanced_weights(rows)
    total = torch.zeros_like(rows[0][key], dtype=torch.float64)
    for weight, row in zip(weights, rows, strict=True):
        total += float(weight) * row[key].double()
    if not torch.allclose(total, total.T, rtol=0, atol=1e-12):
        raise AssertionError("weighted alignment must remain symmetric")
    return total


def new_subspace(a0, covariance):
    if (a0.ndim != 2 or a0.shape[0] != 2 or covariance.shape != (a0.shape[1], a0.shape[1])
            or not torch.isfinite(a0).all() or not torch.isfinite(covariance).all()
            or not torch.allclose(covariance, covariance.T, rtol=0, atol=1e-12)):
        raise ValueError("rank-two A and finite symmetric input covariance required")
    eigenvalues, vectors = torch.linalg.eigh(covariance.double())
    eigenvalues, vectors = eigenvalues.flip(0), vectors.flip(1)[:, :2].clone()
    for column in range(2):
        pivot = int(vectors[:, column].abs().argmax())
        if vectors[pivot, column] < 0:
            vectors[:, column].neg_()
    u, singular, _ = torch.linalg.svd(a0.double(), full_matrices=False)
    candidate = (u @ torch.diag(singular) @ vectors.T).to(dtype=a0.dtype)
    if not torch.allclose(candidate.double() @ candidate.double().T, a0.double() @ a0.double().T, rtol=2e-6, atol=2e-9):
        raise AssertionError("A A.T was not preserved")
    return candidate, eigenvalues, vectors


def b_only_step(parameters, start, gradients):
    if len(parameters) != 2 or sum(p.numel() for p in parameters) != 256 or any(p.shape != (64, 2) for p in parameters):
        raise ValueError("exactly Q/V B matrices (256 parameters) required")
    return step_from_gradients(parameters, start, gradients)


def summarize(rows):
    groups = {}
    for group in GROUPS:
        subset = [r for r in rows if r["group"] == group]
        ids = list(dict.fromkeys(r["shot"] for r in subset))
        result = dict(shots=len(ids), cases=len(subset), no_support=sum(r["supported"] == 0 for r in subset))
        for a, b in COMPARISONS:
            values = np.asarray([np.mean([r["arms"][a]["bce"] - r["arms"][b]["bce"]
                                          for r in subset if r["shot"] == sid]) for sid in ids], dtype=np.float64)
            sd = float(values.std(ddof=1)) if len(values) > 1 else None
            result[f"{a}_vs_{b}"] = dict(mean_delta=float(values.mean()) if len(values) else None,
                                         shot_sd=sd, shot_se=sd / math.sqrt(len(values)) if sd is not None else None,
                                         improved=int((values < 0).sum()), worse=int((values > 0).sum()),
                                         ties=int((values == 0).sum()))
        groups[group] = result
    return groups


def _set_a(lora, values):
    with torch.no_grad():
        lora.lora_A_q.copy_(values[0])
        lora.lora_A_v.copy_(values[1])


def _set_b(lora, values):
    with torch.no_grad():
        for p, value in zip((lora.lora_B_q, lora.lora_B_v), values, strict=True):
            p.copy_(value)
            p.grad = None


def _parts(model):
    attention = model.clone.encoder.layers[-1].self_attn
    packed = attention.parametrizations.in_proj_weight.original
    lora = attention.parametrizations.in_proj_weight[0]
    if packed.shape != (192, 64) or any(p.shape != (2, 64) for p in (lora.lora_A_q, lora.lora_A_v)):
        raise ValueError("qualified packed Q/K/V and Q/V rank-two A required")
    if any(p.shape != (64, 2) for p in (lora.lora_B_q, lora.lora_B_v)):
        raise ValueError("qualified Q/V rank-two B required")
    return packed, lora


def _neutral_logits(base, model, lora, a0, anew, signal, case):
    if torch.count_nonzero(lora.lora_B_q) or torch.count_nonzero(lora.lora_B_v):
        raise ValueError("initial B must be exactly zero")
    x, available = window_batch(signal, [case["origin"], case["future"]], device="cpu")
    _set_a(lora, a0)
    with torch.no_grad():
        native = base(x, available)["logit"]
        old = model(x, available)["logit"]
        _set_a(lora, anew)
        new = model(x, available)["logit"]
    _set_a(lora, a0)
    if not torch.equal(native, old) or not torch.equal(old, new):
        raise AssertionError("native, A0 and Anew initial logits must be exactly equal")
    return int(len(native))


def _read_sealed(index, sid, mapping, hashes):
    stem = f"{index:04d}_{sid}"
    seal_path = RUN / "shots" / f"{stem}_seal.json"
    curve_path = RUN / "shots" / f"{stem}.npz"
    seal = read_json(seal_path)
    seal_sha, curve_sha = sha(seal_path), sha(curve_path)
    forecast_path = RUN / "shots" / f"{stem}_forecasts.npz"
    forecast_sha = None
    if index <= 20:
        if (seal.get("shot") != sid or seal.get("current_outcome_used_in_prediction") is not False
                or seal.get("curve_sha256") != curve_sha or seal.get("forecast_sha256") is not None):
            raise ValueError("warm shot requires a valid original pre-outcome seal")
    else:
        forecast_sha = sha(forecast_path)
        validate_seal(seal, sid, curve_sha, forecast_sha, ORIGINAL_WARM_STATE)
    with np.load(curve_path, allow_pickle=False) as archive:
        sealed_times = archive["times"].copy()
    path = io23.path_for("EAST", sid)
    stat = path.stat()
    expected = hashes[str(sid)]
    if sha(path) != expected:
        raise ValueError("fixed raw HDF hash mismatch")
    signal_reader, truth_reader = io23.read_signals, io23.read_outcome
    calls = []
    def signal_guard(device, requested, *args, **kwargs):
        if device != "EAST" or requested != sid:
            raise ValueError("only the current sealed shot may be read")
        calls.append("signal")
        return signal_reader(device, requested, *args, **kwargs)
    def truth_guard(device, requested):
        if device != "EAST" or requested != sid or calls != ["signal"]:
            raise ValueError("only the current sealed shot outcome may be read")
        calls.append("outcome")
        return truth_reader(device, requested)
    with (patch.object(io23, "read_signals", side_effect=signal_guard),
          patch.object(io23, "read_outcome", side_effect=truth_guard)):
        raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
        disruptive, down = io23.read_outcome("EAST", sid)
    if calls != ["signal", "outcome"] or not np.array_equal(raw["times"], sealed_times):
        raise ValueError("sealed clock or scoped HDF read mismatch")
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha(path) != expected:
        raise ValueError("HDF changed during scoped read")
    if sha(seal_path) != seal_sha or sha(curve_path) != curve_sha or (forecast_sha is not None and sha(forecast_path) != forecast_sha):
        raise ValueError("old seal/artifact changed during scoped read")
    signal = prepare_actual(raw, 21, mapping)
    selected = select_cases(raw["times"], disruptive, down)
    receipt = dict(stream_index=index, shot=sid, disruptive=disruptive, cases=len(selected),
                   size=stat.st_size, mtime_ns=stat.st_mtime_ns, raw_sha256=expected,
                   seal_sha256=seal_sha, curve_sha256=curve_sha, forecast_sha256=forecast_sha)
    return signal, selected, receipt


def _warm_gradient(model, packed, signal, case, threshold):
    x, available, target, mask = _targets(model, signal, case["origin"], device="cpu")
    prediction = model(x, available)["diagnostic"]
    losses = measurement_losses(prediction, target, mask)
    if losses is None:
        d = torch.zeros_like(packed)
    else:
        d, = torch.autograd.grad(losses["motion"], (packed,))
    xr, ar = window_batch(signal, [case["future"]], device="cpu")
    logit = model(xr, ar)["logit"]
    label = torch.full_like(logit, float(case["label"]))
    risk = F.binary_cross_entropy_with_logits(logit - math.log(threshold / (1 - threshold)), label)
    r, = torch.autograd.grad(risk, (packed,))
    if not torch.isfinite(d).all() or not torch.isfinite(r).all():
        raise ValueError("finite original packed-matrix gradients required")
    dq, dv = d[:64].detach().clone(), d[128:].detach().clone()
    rq, rv = r[:64].detach().clone(), r[128:].detach().clone()
    cq, aq = alignment_matrix(dq, rq)
    cv, av = alignment_matrix(dv, rv)
    return dict(supported=int(mask.sum()), q=cq, v=cv, active_q=aq, active_v=av,
                dq=dq, dv=dv, rq=rq, rv=rv)


def _evaluate(model, lora, signal, case, a0, anew, threshold):
    parameters = (lora.lora_B_q, lora.lora_B_v)
    zero = tuple(torch.zeros_like(p) for p in parameters)
    _set_a(lora, a0)
    _set_b(lora, zero)
    x, available, target, mask = _targets(model, signal, case["origin"], device="cpu")
    supported = int(mask.sum())
    gradients, losses_record = {}, {}
    configs = (("old", a0, ("motion", "real")), ("new", anew, ("motion", "anti_motion", "hold", "real")))
    issued = None
    for prefix, basis, names in configs:
        _set_a(lora, basis)
        _set_b(lora, zero)
        pred = model(x, available)["diagnostic"]
        if issued is None:
            issued = pred.detach().clone()
        elif not torch.equal(issued, pred.detach()):
            raise AssertionError("zero-B diagnostic prediction changed with A")
        losses = measurement_losses(pred, target, mask)
        if losses is not None:
            for i, name in enumerate(names):
                gradients[f"{prefix}_{name}"] = torch.autograd.grad(losses[name], parameters, retain_graph=i < len(names) - 1)
                losses_record[f"{prefix}_{name}"] = float(losses[name].detach())
    arm_configs = {
        "old_motion": (a0, "old_motion"), "new_motion": (anew, "new_motion"),
        "new_anti_motion": (anew, "new_anti_motion"), "new_hold": (anew, "new_hold"),
        "old_mse": (a0, "old_real"), "new_mse": (anew, "new_real"),
    }
    states = {"none": (a0, zero)}
    steps = {"none": dict(grad_norm=None, displacement=0.0)}
    for arm, (basis, grad_key) in arm_configs.items():
        _set_a(lora, basis)
        _set_b(lora, zero)
        if supported:
            grad_norm, displacement = b_only_step(parameters, zero, gradients[grad_key])
        else:
            grad_norm, displacement = None, 0.0
        if not torch.equal(lora.lora_A_q, basis[0]) or not torch.equal(lora.lora_A_v, basis[1]):
            raise AssertionError("A changed during B-only Adam")
        states[arm] = (basis, tuple(p.detach().clone() for p in parameters))
        steps[arm] = dict(grad_norm=grad_norm, displacement=displacement)
    # All candidate updates were formed from o and matured o+1..o+8 only.
    xr, ar = window_batch(signal, [case["future"]], device="cpu")
    tau = math.log(threshold / (1 - threshold))
    arms = {}
    for arm in ARMS:
        basis, b = states[arm]
        _set_a(lora, basis)
        _set_b(lora, b)
        with torch.no_grad():
            logit = model(xr, ar)["logit"]
            label = torch.full_like(logit, float(case["label"]))
            arms[arm] = dict(bce=float(F.binary_cross_entropy_with_logits(logit - tau, label)),
                             bce_unshifted=float(F.binary_cross_entropy_with_logits(logit, label)),
                             logit=float(logit), prob=float(logit.sigmoid()), **steps[arm])
    _set_a(lora, a0)
    _set_b(lora, zero)
    return dict(case, supported=supported, no_op=not supported, losses=losses_record or None, arms=arms)


def _cell(pair):
    if pair["mean_delta"] is None:
        return "—"
    mean = f"{pair['mean_delta']:+.7f}"
    if pair["shot_se"] is not None:
        mean += f" ± {pair['shot_se']:.7f}"
    return f"{mean}; {pair['improved']}/{pair['worse']}/{pair['ties']}"


def make_report(summary):
    lines = ["# Q/V rank-2 输入子空间：warm20 拟合、封存后80炮机制探针", "",
             "训练仅使用原 warm20；评估固定流 21–100，不能据已见后80炮标签拟合。"
             "后80炮此前已用于开发，不是独立验证，不构成部署或炮级 TPR。", "",
             f"训练 {summary['warm_shots']} 炮/{summary['warm_cases']} 例；评估 {summary['test_shots']} 炮/{summary['test_cases']} 例，"
             f"无支持保留为 no-op：训练 {summary['warm_no_support']}、评估 {summary['test_no_support']}。", "",
             "训练只对最后层 packed attention 原始 W 求真实 motion 和阈值中心化风险 BCE 的梯度；"
             "Q/V 分别组→炮→窗等权累计归一化对称矩阵，取 top-2，按 A0 的 SVD 奇异值重建 Anew。"
             "准备阶段 B 恒为零，所有初始风险与 native、A0 逐点精确相同；A Aᵀ 保持。", "",
             "评估每例从同一个 decoder-only 父状态出发，A 冻结、仅 Q/V 的 256 个 B 参数执行一次真实 clip1、"
             "lr=1e-4 零动量 Adam；然后计算 o+32 实际阈值中心化 BCE，绝不把风险标签用于该步。", "",
             "表中 ΔBCE=左臂−右臂，负值表示左臂较好；先对同炮同组窗口平均，再按炮等权。"
             "± 为炮间标准误，后面是改善/恶化/持平炮数。", "",
             "| 组 | 炮/例/no-op | A0 motion−none | Anew motion−none | Anew anti−none | Anew hold−none | A0 MSE−none | Anew MSE−none |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in GROUPS:
        g = summary["groups"][group]
        lines.append(f"| {group} | {g['shots']}/{g['cases']}/{g['no_support']} | " + " | ".join(_cell(g[f"{a}_vs_none"]) for a in ARMS[1:]) + " |")
    lines += ["", "| 组 | 新 motion−旧 motion | 新 motion−反向 motion | 新 motion−hold | 新 MSE−旧 MSE | 新 MSE−hold |",
              "|---|---:|---:|---:|---:|---:|"]
    for group in GROUPS:
        g = summary["groups"][group]
        names = ("new_motion_vs_old_motion", "new_motion_vs_new_anti_motion", "new_motion_vs_new_hold", "new_mse_vs_old_mse", "new_mse_vs_new_hold")
        lines.append(f"| {group} | " + " | ".join(_cell(g[name]) for name in names) + " |")
    lines += ["", "## 机制与边界", "",
              f"风险中性逐点严格相等检查：{summary['neutrality_points']} 点；"
              f"A Aᵀ 最大绝对误差 Q={summary['gram_error_q']:.3g}、V={summary['gram_error_v']:.3g}；"
              f"训练零范数贡献 Q={summary['zero_contribution_q']}、V={summary['zero_contribution_v']}（均保留分母）。", "",
              f"Q 前三特征值：{summary['eigenvalues_q'][:3]}；V 前三特征值：{summary['eigenvalues_v'][:3]}。", "",
              "谱分解只针对初始 B-only SGD 的方向推导；逐臂裁剪与 Adam 的实际有限步不受此推导保证。"
              "若真实反馈不胜固定基线或匹配伪反馈，不推荐部署；无论数值方向如何，本探针仍需独立验证。", ""]
    return "\n".join(lines)


def run():
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite an existing probe or report")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    paths = dict(run_protocol=RUN / "protocol.json", decoder_completed=DECODER / "completed.json",
                 decoder_warm_json=DECODER / "warm.json", decoder_warm_pt=DECODER / "warm.pt",
                 measurement_completed=MEASUREMENT / "completed.json", measurement_cases=MEASUREMENT / "cases.json",
                 map=MAP, raw_hashes=RAW_HASHES, source=DEFAULT_PARENT)
    hashes = {key: sha(path) for key, path in paths.items()}
    if hashes != EXPECTED:
        raise ValueError("fixed decoder/source/sealed80 input hash mismatch")
    original, warm = read_json(paths["run_protocol"]), read_json(paths["decoder_warm_json"])
    done, measurement_done = read_json(paths["decoder_completed"]), read_json(paths["measurement_completed"])
    if (done["status"] != "COMPLETED_RISK_NEUTRAL_DECODER_ONLY_PREPARATION"
            or done["artifact_sha256"]["warm.pt"] != hashes["decoder_warm_pt"]
            or done["artifact_sha256"]["warm.json"] != hashes["decoder_warm_json"]
            or measurement_done["artifact_sha256"]["cases.json"] != hashes["measurement_cases"]):
        raise ValueError("decoder or measurement completion receipt mismatch")
    order = [item["shot"] for item in original["order"]]
    mapping, raw_hashes = read_json(MAP), read_json(RAW_HASHES)
    if (len(order) != 360 or len(set(order)) != 360 or warm["label_shots"] != order[:20]
            or mapping["calibration_ids"] != order[:20] or warm["fitted_state_sha256"] != PARENT_STATE
            or warm["lora_updates"] != 0 or original["input_sha256"][str(MAP)] != hashes["map"]
            or original["input_sha256"][str(RAW_HASHES)] != hashes["raw_hashes"]):
        raise ValueError("fixed order, neutral parent or map identity mismatch")
    code = {name: sha(ROOT / name) for name in CODE_PATHS}
    plan = fixed_plan(order[:20], order[20:100], hashes)
    plan.update(input_paths={key: str(path) for key, path in paths.items()}, code_sha256=code,
                parent_state_sha256=PARENT_STATE, original_warm_state_sha256=ORIGINAL_WARM_STATE,
                threshold=original["threshold"])
    # This fixed plan is durable before any warm or test HDF is opened.
    OUTPUT.mkdir(parents=False, exist_ok=False)
    write_json(OUTPUT / "protocol.json", plan)
    print("fixed protocol sealed; fitting original packed gradients on warm20 only", flush=True)
    base, source = load_parent(DEFAULT_PARENT)
    base.eval().requires_grad_(False)
    source_before = state_digest(base)
    if source_before != warm["source_state_sha256"] or float(source["threshold"]) != float(original["threshold"]):
        raise ValueError("source state or threshold mismatch")
    threshold = float(source["threshold"])
    model = restore_warm(base, paths["decoder_warm_pt"], warm, device="cpu")
    packed, lora = _parts(model)
    a0 = (lora.lora_A_q.detach().clone(), lora.lora_A_v.detach().clone())
    if torch.count_nonzero(lora.lora_B_q) or torch.count_nonzero(lora.lora_B_v):
        raise ValueError("risk-neutral decoder parent requires exactly zero Q/V B")
    packed.requires_grad_(True)
    if {id(p) for p in model.parameters() if p.requires_grad} != {id(packed)}:
        raise AssertionError("only original packed attention matrix may have a warm gradient")
    warm_rows, warm_signals, warm_receipts = [], {}, []
    for index, sid in enumerate(order[:20], 1):
        signal, selected, receipt = _read_sealed(index, sid, mapping, raw_hashes)
        warm_signals[sid] = signal
        warm_receipts.append(receipt)
        for case in selected:
            grads = _warm_gradient(model, packed, signal, case, threshold)
            warm_rows.append(dict(shot=sid, stream_index=index, **case, **grads))
        if state_digest(model) != PARENT_STATE or any(p.grad is not None for p in model.parameters()):
            raise AssertionError("warm gradient phase changed parent state")
        print(f"warm sealed shots {index}/20; cases {len(warm_rows)}", flush=True)
    cq, cv = weighted_alignment(warm_rows, "q"), weighted_alignment(warm_rows, "v")
    new_q, eigen_q, vectors_q = new_subspace(a0[0], cq)
    new_v, eigen_v, vectors_v = new_subspace(a0[1], cv)
    anew = (new_q, new_v)
    gram_q = float((new_q.double() @ new_q.double().T - a0[0].double() @ a0[0].double().T).abs().max())
    gram_v = float((new_v.double() @ new_v.double().T - a0[1].double() @ a0[1].double().T).abs().max())
    packed.requires_grad_(False)
    neutrality = 0
    for row in warm_rows:
        neutrality += _neutral_logits(base, model, lora, a0, anew, warm_signals[row["shot"]], row)
    if state_digest(model) != PARENT_STATE or state_digest(base) != source_before:
        raise AssertionError("subspace preparation mutated the parent")
    print("warm-only subspace fixed; opening sealed 21..100 for evaluation", flush=True)
    # Parse held-out cases only after the learned subspace is fixed.
    old_cases = read_json(paths["measurement_cases"])
    expected_cases = [(r["shot"], r["origin"], r["group"], r["label"]) for r in old_cases]
    if len(expected_cases) != 240:
        raise ValueError("original fixed 240 evaluation coordinates required")
    for p in (lora.lora_B_q, lora.lora_B_v):
        p.requires_grad_(True)
    if {id(p) for p in model.parameters() if p.requires_grad} != {id(lora.lora_B_q), id(lora.lora_B_v)}:
        raise AssertionError("only the two B matrices may receive evaluation gradients")
    test_rows, test_receipts = [], []
    for index, sid in enumerate(order[20:100], 21):
        signal, selected, receipt = _read_sealed(index, sid, mapping, raw_hashes)
        test_receipts.append(receipt)
        for case in selected:
            neutrality += _neutral_logits(base, model, lora, a0, anew, signal, case)
            result = _evaluate(model, lora, signal, case, a0, anew, threshold)
            test_rows.append(dict(shot=sid, stream_index=index, **result))
        if state_digest(model) != PARENT_STATE or state_digest(base) != source_before or any(p.grad is not None for p in model.parameters()):
            raise AssertionError("B update or A installation escaped a case")
        print(f"sealed evaluation shots {index - 20}/80; cases {len(test_rows)}", flush=True)
    if [(r["shot"], r["origin"], r["group"], r["label"]) for r in test_rows] != expected_cases:
        raise ValueError("held-out coordinates, groups or HDF labels changed")
    if any(sha(path) != hashes[key] for key, path in paths.items()) or any(sha(ROOT / name) != value for name, value in code.items()):
        raise ValueError("bound input or code changed during the probe")
    groups = summarize(test_rows)
    summary = dict(warm_shots=len(warm_receipts), warm_cases=len(warm_rows),
                   warm_no_support=sum(r["supported"] == 0 for r in warm_rows),
                   test_shots=len(test_receipts), test_cases=len(test_rows),
                   test_no_support=sum(r["supported"] == 0 for r in test_rows),
                   zero_contribution_q=sum(not r["active_q"] for r in warm_rows),
                   zero_contribution_v=sum(not r["active_v"] for r in warm_rows),
                   neutrality_points=neutrality, gram_error_q=gram_q, gram_error_v=gram_v,
                   eigenvalues_q=eigen_q.tolist(), eigenvalues_v=eigen_v.tolist(), groups=groups)
    warm_meta = [{key: value for key, value in row.items() if key not in ("q", "v", "dq", "dv", "rq", "rv")}
                 for row in warm_rows]
    for meta, weight in zip(warm_meta, balanced_weights(warm_rows), strict=True):
        meta["balance_weight"] = float(weight)
    with (OUTPUT / "subspace.npz").open("xb") as stream:
        np.savez_compressed(stream, C_q=cq.numpy(), C_v=cv.numpy(), eigenvalues_q=eigen_q.numpy(),
                            eigenvalues_v=eigen_v.numpy(), E_q=vectors_q.numpy(), E_v=vectors_v.numpy(),
                            A0_q=a0[0].numpy(), A0_v=a0[1].numpy(), Anew_q=new_q.numpy(), Anew_v=new_v.numpy())
    with (OUTPUT / "warm_gradients.npz").open("xb") as stream:
        np.savez_compressed(stream, **{key: torch.stack([r[key] for r in warm_rows]).numpy() for key in ("dq", "rq", "dv", "rv")})
    for name, value in (("warm_cases.json", warm_meta), ("cases.json", test_rows),
                        ("shots.json", dict(warm=warm_receipts, test=test_receipts)), ("summary.json", summary)):
        write_json(OUTPUT / name, value)
    with REPORT.open("x", encoding="utf-8") as stream:
        stream.write(make_report(summary))
    artifacts = ("protocol.json", "subspace.npz", "warm_gradients.npz", "warm_cases.json", "cases.json", "shots.json", "summary.json")
    completed = dict(status="COMPLETED_WARM20_SUBSPACE_SEALED80_CPU_PROBE",
                     script_sha256=sha(__file__), report_sha256=sha(REPORT),
                     artifact_sha256={name: sha(OUTPUT / name) for name in artifacts})
    write_json(OUTPUT / "completed.json", completed)
    print(json.dumps(completed), flush=True)


if __name__ == "__main__":
    run()
