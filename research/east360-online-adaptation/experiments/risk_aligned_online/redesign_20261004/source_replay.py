"""Fixed TRAIN-only source replay candidate. No optimizer, update or runner hooks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..simple23 import Config, Simple23


PARENT_SHA256 = "c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02"
DEFAULT_PARENT = Path(__file__).resolve().parents[1] / "runs/gradient_aligned512_20260929/source.pt"
TENSOR_FIELDS = ("x", "available", "labels", "teacher_logits", "shotids", "endindices",
                 "horizon_ms", "lead_ms")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_teacher(checkpoint=DEFAULT_PARENT):
    """CPU-load the original qualified classifier, hash checked before unpickling.

    This exact trusted local parent has NumPy metadata. No other pickle is
    accepted; forecast qualification/training is irrelevant to source risk.
    """
    if sha256(checkpoint) != PARENT_SHA256:
        raise ValueError("parent SHA256 does not match the original qualified source")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(**payload["model_config"])
    if (cfg.window != 128 or cfg.patch != 8 or cfg.dim != 64
            or cfg.input_view != "source" or cfg.fast_site != "tokens"):
        raise ValueError("original source-normalized flat23 token model required")
    with torch.random.fork_rng(devices=[]):
        base = Simple23(cfg, payload["source_center"], payload["source_scale"])
    base.load_state_dict(payload["model"], strict=True)
    return base.requires_grad_(False).eval()


def bank_sha256(bank):
    """Content identity, independent of torch.save's ZIP serialization."""
    digest = hashlib.sha256()
    meta = {key: bank[key] for key in ("source_parent_sha256", "protocol")}
    digest.update(json.dumps(meta, sort_keys=True, allow_nan=False).encode())
    for name in TENSOR_FIELDS:
        tensor = bank[name].detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_bank(bank):
    """Fail closed on malformed, nonfinite, non-TRAIN or modified bank content."""
    if not isinstance(bank, dict) or not all(k in bank for k in
            (*TENSOR_FIELDS, "source_parent_sha256", "protocol", "bank_sha256")):
        raise ValueError("source bank fields missing")
    if not _valid_sha(bank["source_parent_sha256"]):
        raise ValueError("valid source parent SHA256 required")
    for name in TENSOR_FIELDS:
        t = bank[name]
        if not isinstance(t, torch.Tensor) or t.device.type != "cpu" or t.requires_grad:
            raise ValueError("detached CPU source bank tensors required")
        if not torch.isfinite(t).all():
            raise ValueError(f"finite source bank {name} required")
    if bank["labels"].ndim != 1:
        raise ValueError("one actual source label per window required")
    n = len(bank["labels"])
    if not 2 <= n <= 128 or n % 2 or bank["x"].shape != (n, 128, 23) or bank["x"].dtype != torch.float32:
        raise ValueError("balanced nonempty [N,128,23] float32 source bank required")
    if bank["available"].shape != bank["x"].shape or bank["available"].dtype != torch.bool:
        raise ValueError("matching boolean source availability required")
    for name in TENSOR_FIELDS[2:]:
        if bank[name].shape != (n,):
            raise ValueError(f"one {name} per source window required")
    if any(bank[k].dtype != torch.int64 for k in ("shotids", "endindices")):
        raise ValueError("integer source shot IDs/endindices required")
    if any(not bank[k].is_floating_point() for k in ("labels", "teacher_logits", "horizon_ms", "lead_ms")):
        raise ValueError("floating source labels/logits/timing required")
    positive = bank["labels"] == 1
    safe = bank["labels"] == 0
    if not (positive | safe).all() or int(positive.sum()) != n // 2:
        raise ValueError("actual binary balanced source labels required")
    protocol = bank["protocol"]
    if (not isinstance(protocol, dict) or protocol.get("source_role") != "TRAIN"
            or protocol.get("source_device") != "J-TEXT"
            or protocol.get("source_label") != "max(25ms,AdvancedTime)"
            or protocol.get("input_coordinates") != "simple_data.read_shot raw before source asinh"
            or protocol.get("target_labels_accessed") != 0
            or protocol.get("source_holdout_signals_accessed") != 0
            or protocol.get("channels") != list(io23.RAW23)
            or protocol.get("per_class") != n // 2):
        raise ValueError("source TRAIN-only protocol required")
    ids = bank["shotids"].tolist()
    train = protocol.get("train_ids", [])
    if len(set(ids)) != n or len(train) != len(set(train)) or not set(ids) <= set(train):
        raise ValueError("distinct authorized source TRAIN shots required")
    if ((bank["endindices"] < 127).any()
            or not bank["available"].flatten(1).any(1).all()
            or (bank["horizon_ms"] < 25).any()
            or not (bank["lead_ms"][positive] > 10).all()
            or not (bank["lead_ms"][positive] <= bank["horizon_ms"][positive]).all()
            or not (bank["lead_ms"][safe] == -1).all()):
        raise ValueError("valid causal windows and strict source lead >10 <=horizon required")
    if not _valid_sha(bank["bank_sha256"]) or bank_sha256(bank) != bank["bank_sha256"]:
        raise ValueError("source bank content hash changed")
    return bank


def _positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def build_source_bank(teacher, *, source_parent_sha256, data_api=simple_data,
                      per_class=64, seed=131, safe_end_index=None):
    """One deterministic full causal window per distinct disruptive/safe TRAIN shot.

    No writes. TRAIN IDs are sorted then seed-permuted, independently of risk.
    Positives use the middle actual positive endpoint with strict lead >10 ms;
    safes use their middle valid endpoint (or the explicit fixed endpoint).
    """
    _positive_int(per_class, "per_class")
    if per_class > 64:
        raise ValueError("per_class must be <=64 for a bounded source guard")
    if not _valid_sha(source_parent_sha256):
        raise ValueError("valid source parent SHA256 required")
    if (not isinstance(teacher, Simple23) or teacher.cfg.window != 128
            or teacher.cfg.input_view != "source" or teacher.training
            or any(p.requires_grad for p in teacher.parameters())
            or any(t.device.type != "cpu" for t in teacher.state_dict().values())):
        raise ValueError("fully frozen original Simple23 CPU teacher in eval mode required")
    if safe_end_index is not None and (not isinstance(safe_end_index, int)
            or isinstance(safe_end_index, bool) or safe_end_index < 127):
        raise ValueError("safe_end_index must be an integer >=127")
    roles = data_api.source_roles()
    train = sorted(int(s) for s in roles["train"])
    if not train or len(train) != len(set(train)):
        raise ValueError("distinct source TRAIN IDs required")
    if set(train) & (set(roles["val"]) | set(roles["test"])):
        raise ValueError("source TRAIN/holdout overlap")
    all_advanced = data_api.advanced_times()
    advanced = {sid: all_advanced[sid] for sid in train if sid in all_advanced}
    rng = np.random.default_rng(seed)
    rows = {1: [], 0: []}
    for sid in rng.permutation(train).tolist():
        signals = data_api.read_shot("J-TEXT", sid)
        shot = data_api.attach_outcome(signals, "J-TEXT", advanced)
        if shot["device"] != "J-TEXT" or int(shot["shot"]) != sid:
            raise ValueError("source TRAIN signal identity mismatch")
        times = np.asarray(shot["times"])
        if (times.ndim != 1 or not np.isfinite(times).all()
                or not (np.diff(times) > 0).all()
                or shot["values"].shape != (len(times), 23)
                or shot["available"].shape != shot["values"].shape
                or shot["available"].dtype != bool
                or not np.isfinite(shot["values"]).all()):
            raise ValueError("finite aligned source signals/times required")
        horizon = io23.effective_horizon_ms(sid, advanced)[0]
        expected_y, expected_valid = io23.labels_at(times, shot["is_disrupt"], shot["down"], horizon)
        if (shot["horizon"] != horizon or not np.array_equal(shot["y"], expected_y)
                or not np.array_equal(shot["valid"], expected_valid)):
            raise ValueError("source labels must be actual max(25ms,AdvancedTime) outcomes")
        label = int(shot["is_disrupt"])
        if len(rows[label]) == per_class:
            continue
        eligible = expected_valid & (expected_y == label) & shot["available"].any(1)
        eligible[:127] = False
        lead = np.round((shot["down"] - times) * 1000, 9) if label else np.full(len(times), -1.)
        if label:
            eligible &= (lead > 10.) & (lead <= horizon)
        endpoints = np.flatnonzero(eligible)
        if not len(endpoints):
            continue
        end = int(endpoints[len(endpoints) // 2])
        if not label and safe_end_index is not None:
            if safe_end_index >= len(times) or not eligible[safe_end_index]:
                continue
            end = safe_end_index
        if not expected_valid[end - 127:end + 1].all():
            continue
        x, available = simple_data.window(shot, end, 128)
        rows[label].append((x, available, label, sid, end, horizon, float(lead[end])))
        if all(len(rows[y]) == per_class for y in (1, 0)):
            break
    if any(len(rows[y]) < per_class for y in (1, 0)):
        raise ValueError(f"insufficient distinct valid TRAIN shots: disruptive={len(rows[1])}, safe={len(rows[0])}, required={per_class} each")
    selected = rows[1] + rows[0]
    bank = dict(x=torch.from_numpy(np.stack([r[0] for r in selected])),
                available=torch.from_numpy(np.stack([r[1] for r in selected])),
                labels=torch.tensor([r[2] for r in selected], dtype=torch.float32),
                shotids=torch.tensor([r[3] for r in selected], dtype=torch.int64),
                endindices=torch.tensor([r[4] for r in selected], dtype=torch.int64),
                horizon_ms=torch.tensor([r[5] for r in selected], dtype=torch.float64),
                lead_ms=torch.tensor([r[6] for r in selected], dtype=torch.float64),
                source_parent_sha256=source_parent_sha256,
                protocol=dict(version=1, source_device="J-TEXT", source_role="TRAIN",
                              train_ids=train, source_label="max(25ms,AdvancedTime)",
                              positive_lead_rule="round(lead_ms,9)>10 and <=horizon_ms",
                              window=128, channels=list(io23.RAW23), per_class=per_class,
                              seed=int(seed), safe_end_index=safe_end_index,
                              selection="seed-permuted distinct TRAIN shots; middle eligible endpoint",
                              input_coordinates="simple_data.read_shot raw before source asinh",
                              target_labels_accessed=0, source_holdout_signals_accessed=0,
                              teacher="original frozen classifier; original fast_origin; risk logit only",
                              code_sha256={"source_replay.py": sha256(__file__),
                                           "simple_data.py": sha256(simple_data.__file__),
                                           "io23.py": sha256(io23.__file__)}))
    with torch.no_grad():
        bank["teacher_logits"] = teacher(bank["x"], bank["available"])["logit"].detach().cpu().clone()
    if not torch.isfinite(bank["teacher_logits"]).all():
        raise ValueError("finite original teacher logits required")
    bank["bank_sha256"] = bank_sha256(bank)
    return validate_bank(bank)


def sample_source(bank, rng, batch_size=64):
    """Sample without replacement, exactly half actual positives and half safes."""
    validate_bank(bank)
    _positive_int(batch_size, "batch_size")
    if batch_size % 2:
        raise ValueError("source batch_size must be even")
    if batch_size > len(bank["labels"]):
        raise ValueError("insufficient distinct source samples for balanced batch")
    groups = [torch.where(bank["labels"] == y)[0].numpy() for y in (1, 0)]
    indices = np.concatenate([rng.choice(g, batch_size // 2, replace=False) for g in groups])
    return torch.from_numpy(rng.permutation(indices).astype(np.int64))


def source_batch(bank, indices=None, device="cpu"):
    """Detached copies: retain indices to reuse identical windows before/after."""
    validate_bank(bank)
    indices = torch.arange(len(bank["labels"])) if indices is None else torch.as_tensor(indices)
    if (indices.ndim != 1 or indices.dtype != torch.int64 or not len(indices)
            or (indices < 0).any() or (indices >= len(bank["labels"])).any()
            or len(indices.unique()) != len(indices)):
        raise ValueError("distinct in-range integer source indices required")
    indices = indices.detach().cpu()
    return {k: bank[k].index_select(0, indices).detach().clone().to(device) for k in TENSOR_FIELDS}


def _risk_logits(model, batch):
    fast = model.initial_fast(len(batch["labels"])).detach()
    if not torch.isfinite(fast).all() or fast.count_nonzero():
        raise ValueError("wrapper source replay requires zero temporary fast delta")
    logits = model(batch["x"], batch["available"], fast)["logit"]
    if logits.shape != batch["labels"].shape or not torch.isfinite(logits).all():
        raise ValueError("finite per-window risk logits required")
    return logits


def source_loss(model, bank, device, rng, batch_size=64, *, indices=None):
    """Balanced true-label BCE and audit; never backward, zero_grad, step or mutate."""
    if (not hasattr(model, "base") or model.base.training
            or any(p.requires_grad for p in model.base.parameters())):
        raise ValueError("wrapper with fully frozen eval-mode original base required")
    persistent = tuple(model.persistent_parameters())
    if {id(p) for p in persistent} != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError("only persistent wrapper parameters may require gradients")
    indices = sample_source(bank, rng, batch_size) if indices is None else torch.as_tensor(indices)
    batch = source_batch(bank, indices, device)
    positive = batch["labels"] == 1
    if len(indices) != batch_size or batch_size % 2 or int(positive.sum()) != batch_size // 2:
        raise ValueError("selected source minibatch must be balanced and match batch_size")
    losses = F.binary_cross_entropy_with_logits(_risk_logits(model, batch), batch["labels"], reduction="none")
    p, s = losses[positive].mean(), losses[~positive].mean()
    loss = .5 * (p + s)
    if not torch.isfinite(loss):
        raise ValueError("finite balanced source loss required")
    return loss, dict(indices=indices.tolist(), positive_count=int(positive.sum()),
                      safe_count=int((~positive).sum()), bce=float(loss.detach()),
                      positive_bce=float(p.detach()), safe_bce=float(s.detach()),
                      bank_sha256=bank["bank_sha256"], temporary_fast="zero")


@torch.no_grad()
def evaluate_source_guard(model, bank, device, threshold, baselineoriginal_teacher=None, *, indices=None):
    """Protect original-teacher correct TP/TN points at one fixed source threshold.

    Default: all bank points in one bounded forward (128 with defaults).
    This is not a whole-shot alarm or generalization guarantee.
    """
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("finite fixed source threshold in (0,1) required")
    validate_bank(bank)
    if baselineoriginal_teacher is not None:
        baseline = torch.as_tensor(baselineoriginal_teacher).detach().cpu()
        if not torch.equal(baseline, bank["teacher_logits"]):
            raise ValueError("baseline must equal immutable original teacher logits, not a moving model")
    indices = torch.arange(len(bank["labels"])) if indices is None else torch.as_tensor(indices)
    batch = source_batch(bank, indices, device)
    original_positive = batch["teacher_logits"].sigmoid() >= threshold
    current_positive = _risk_logits(model, batch).sigmoid() >= threshold
    tp = (batch["labels"] == 1) & original_positive
    tn = (batch["labels"] == 0) & ~original_positive
    lost_tp, lost_tn = tp & ~current_positive, tn & current_positive
    violations = lost_tp | lost_tn
    protected = int((tp | tn).sum())
    return dict(passed=protected > 0 and not bool(violations.any()),
                protected_tp=int(tp.sum()), protected_tn=int(tn.sum()),
                lost_tp=int(lost_tp.sum()), lost_tn=int(lost_tn.sum()),
                violating_indices=indices.detach().cpu()[violations.cpu()].tolist(),
                checked_points=len(indices), threshold=float(threshold),
                reason="no_original_correct_points" if not protected else
                       "source_threshold_crossing" if violations.any() else "retained_original_correct_points",
                bank_sha256=bank["bank_sha256"], temporary_fast="zero")


def load_source_bank(path, *, expected_parent_sha256=PARENT_SHA256):
    """Restricted tensor/primitive deserialization and full content validation."""
    bank = torch.load(path, map_location="cpu", weights_only=True)
    validate_bank(bank)
    if bank["source_parent_sha256"] != expected_parent_sha256:
        raise ValueError("source bank parent SHA256 mismatch")
    return bank


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare a source TRAIN-only bank on CPU; no training")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, required=True, help="explicit NEW artifact file; never overwritten")
    parser.add_argument("--per-class", type=int, default=64)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--safe-end-index", type=int)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite source bank: {args.output}")
    if not args.output.parent.is_dir():
        raise FileNotFoundError("output parent must already exist; CLI creates no directories")
    teacher = load_teacher(args.checkpoint)
    bank = build_source_bank(teacher, source_parent_sha256=PARENT_SHA256,
                             data_api=simple_data, per_class=args.per_class,
                             seed=args.seed, safe_end_index=args.safe_end_index)
    if sha256(args.checkpoint) != PARENT_SHA256:
        raise ValueError("parent SHA256 changed during CPU preparation")
    # Only this explicitly invoked entry point writes. Exclusive create also
    # catches races/symlinks; helpers never save, update, or open output paths.
    with args.output.open("xb") as handle:
        torch.save(bank, handle)
    print(json.dumps(dict(output=str(args.output), device="cpu", shape=list(bank["x"].shape),
                          positive_count=args.per_class, safe_count=args.per_class,
                          bank_sha256=bank["bank_sha256"], artifact_sha256=sha256(args.output),
                          source_parent_sha256=PARENT_SHA256), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
