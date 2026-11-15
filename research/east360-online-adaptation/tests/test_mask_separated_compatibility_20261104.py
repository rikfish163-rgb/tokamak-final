"""CPU-only compatibility checks for two authorized J-TEXT signal prefixes.

No outcome, EAST signal, cold state, target head, training, or GPU access.
Run from the worktree root with the command recorded in the companion audit.
"""

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.adaptivity_capacity_20261102.model import source_basis
from experiments.risk_aligned_online.causal_input import apply_input_contract
from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter
from experiments.risk_aligned_online.direct_decoder_meta_20261102 import core as decoder_core
from experiments.risk_aligned_online.lora_stream_20261102.core import encode, forward
from experiments.risk_aligned_online.mask_separated_intra_20261103.model import (
    MaskSeparatedContext,
    mask_embedding_shift,
)
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    load_parent,
)


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "probe_outputs/direct_decoder_source_v1/decoders.pt"
SOURCE_SHA = "c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02"
PRIOR_SHA = "cdde96086517cbc468a911bbb5926dde1c783b3711b0a8abc09caf0a6d2e6730"
SHOT_IDS = (1051510, 1051650)
ENDPOINTS = (127, 255)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_authorized_prefix(shot):
    """Mirror source input conversion, reading at most 256 rows of /data only."""
    assert shot in SHOT_IDS
    path = io23.path_for("J-TEXT", shot)
    with h5py.File(path, "r") as handle:
        channels = [handle["data"][name] for name in io23.RAW23]
        lengths = [len(channel) for channel in channels]
        assert len(set(lengths)) == 1
        n = min(max(ENDPOINTS) + 1, lengths[0])
        assert n > min(ENDPOINTS), "neither declared endpoint is available"
        rate = float(np.asarray(handle["data/ip"].attrs["SampleRate"]).reshape(-1)[0])
        start = float(np.asarray(handle["data/ip"].attrs["StartTime"]).reshape(-1)[0])
        assert np.isfinite(rate) and abs(rate - 1000.0) <= io23.RATE_ATOL_HZ
        for channel in channels:
            assert np.isclose(float(np.asarray(channel.attrs["SampleRate"]).reshape(-1)[0]), rate)
            assert np.isclose(float(np.asarray(channel.attrs["StartTime"]).reshape(-1)[0]), start)
        raw = np.stack([np.asarray(channel[:n], dtype=np.float32) for channel in channels], axis=1)

    prefix_sha = hashlib.sha256(raw.tobytes(order="C")).hexdigest()
    available = np.isfinite(raw)
    values = np.where(available, raw, 0.0).astype(np.float32)
    values[:, io23.RAW23.index("ip")] *= 1000.0
    available &= np.isfinite(values)
    values[~available] = 0.0
    signal = {"values": values, "available": available}
    contracted = apply_input_contract(signal)
    # These real prefixes contain no CIII sentinel; do not force coavailability.
    assert np.array_equal(contracted["available"], available)
    assert np.array_equal(contracted["values"], values)
    print(f"prefix shot={shot} rows={n} raw_flat23_float32_sha256={prefix_sha}")
    return contracted


@pytest.fixture(scope="module")
def source():
    assert sha256(DEFAULT_PARENT) == SOURCE_SHA
    assert sha256(PRIOR) == PRIOR_SHA
    torch.set_num_threads(1)
    previous_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    base, _ = load_parent(DEFAULT_PARENT)
    source_before_wrapping = {k: v.detach().clone() for k, v in base.state_dict().items()}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(131)
        original = MaskSeparatedContext(base, policy="original")
        torch.random.default_generator.manual_seed(131)
        reference = MaskSeparatedContext(base, policy="reference_bt")
    assert all(torch.equal(v, reference.state_dict()[k]) for k, v in original.state_dict().items())
    assert all(not p.requires_grad and p.device.type == "cpu" for p in original.parameters())
    assert not base.training and not original.training and not reference.training
    assert all(torch.equal(v, base.state_dict()[k]) for k, v in source_before_wrapping.items())
    assert not original.slow.weight.count_nonzero() and not reference.slow.weight.count_nonzero()

    stored = torch.load(PRIOR, map_location="cpu", weights_only=True)
    assert stored["source_sha256"] == SOURCE_SHA
    weights = stored["heads"]["forecast_only"]
    assert set(weights) == {"weight", "bias"}
    assert weights["weight"].shape == (575, 1088) and weights["bias"].shape == (575,)
    head = decoder_core.clone_head(original)
    result = head.load_state_dict(weights, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert all(torch.equal(v, base.state_dict()[k]) for k, v in source_before_wrapping.items())
    head.eval().requires_grad_(False)
    basis = source_basis(original)
    print(f"source_sha256={SOURCE_SHA} decoder_sha256={PRIOR_SHA}")
    yield base, original, reference, head, basis
    torch.backends.mha.set_fastpath_enabled(previous_fastpath)


@pytest.fixture(scope="module")
def prefixes():
    return {shot: read_authorized_prefix(shot) for shot in SHOT_IDS}


def hook_snapshot(module):
    return {
        name: (tuple(m._forward_pre_hooks), tuple(m._forward_hooks))
        for name, m in module.named_modules()
    }


@pytest.fixture(autouse=True)
def immutable_state_and_hooks(source):
    modules = source[:4]
    states = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in modules]
    hooks = [hook_snapshot(m) for m in modules]
    yield
    for model, before, registered in zip(modules, states, hooks):
        assert set(before) == set(model.state_dict())
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
        assert hook_snapshot(model) == registered


def run_external(model, pack, head):
    captured = []
    hook = model.base.forecast_head.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    try:
        result = decoder_core.prediction(model, pack, torch.zeros(4, 64), head)
    finally:
        hook.remove()
    assert len(captured) == 1 and captured[0].shape == (1, 1088)
    assert result.shape == (1, 25, 23)
    flat = head(captured[0])
    assert flat.shape == (1, 575)
    assert torch.equal(result, flat.reshape(1, 25, 23))
    for horizon in (0, 12, 24):
        for channel in (0, 1, 2, 12, 22):
            assert torch.equal(result[:, horizon, channel], flat[:, horizon * 23 + channel])
    return result, captured[0]


def assert_source_target_coordinates(base, prediction):
    assert io23.RAW23[:3] == ("Bt", "CIII", "MNM")
    for horizon in (0, 12, 24):
        for channel in (0, 1, 2, 12, 22):
            truth = base.center.reshape(1, 1, 23).expand(1, 25, 23).clone()
            truth[0, horizon, channel] += 0.3 * base.scale[channel]
            observed = torch.zeros_like(truth, dtype=torch.bool)
            observed[0, horizon, channel] = True
            target = torch.asinh(
                (truth[0, horizon, channel] - base.center[channel]) / base.scale[channel]
            )
            expected = F.smooth_l1_loss(prediction[0, horizon, channel], target)
            assert torch.equal(base.forecast_loss(prediction, truth, observed), expected)


@pytest.mark.parametrize("shot", SHOT_IDS)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_real_coavailable_source_window(source, prefixes, shot, endpoint):
    base, original, reference, head, basis = source
    signal = prefixes[shot]
    if endpoint >= len(signal["values"]):
        pytest.skip("declared endpoint not present in this source shot")
    values, available = simple_data.window(signal, endpoint, 128)
    assert np.array_equal(available[:, 0], available[:, 1])
    assert np.array_equal(available[:, 0], available[:, 2])
    assert available[:, 0].any(), "coavailability must include actual observations"
    before_values, before_available = values.copy(), available.copy()
    x = torch.from_numpy(values[None])
    a = torch.from_numpy(available[None])
    with torch.no_grad():
        h0, valid0 = original.tokens(x, a)
        h1, valid1 = reference.tokens(x, a)
        pack0 = encode(original, signal, [endpoint], basis)
        pack1 = encode(reference, signal, [endpoint], basis)
        out0 = forward(original, pack0, torch.zeros(4, 64), forecast=True)
        out1 = forward(reference, pack1, torch.zeros(4, 64), forecast=True)
        pred0, input0 = run_external(original, pack0, head)
        pred1, input1 = run_external(reference, pack1, head)
        native = base(x, a)
        assert_source_target_coordinates(base, pred0)
    assert torch.equal(h0, h1) and torch.equal(valid0, valid1)
    assert all(torch.equal(pack0[k], pack1[k]) for k in pack0)
    assert torch.equal(out0["logit"], out1["logit"])
    assert torch.equal(out0["forecast"], out1["forecast"])
    assert torch.equal(input0, input1) and torch.equal(pred0, pred1)
    assert torch.equal(native["logit"], out0["logit"])
    assert torch.equal(native["forecast"], out0["forecast"])
    assert np.array_equal(values, before_values) and np.array_equal(available, before_available)
    base_logit_gap = (native["logit"] - out0["logit"]).abs().max().item()
    base_forecast_gap = (native["forecast"] - out0["forecast"]).abs().max().item()
    print(
        f"real shot={shot} endpoint={endpoint} observed_coavailable={int(available[:, 0].sum())}/128 "
        f"tokens=exact valid=exact logits=exact native_forecast=exact external_forecast=exact "
        f"base_logit_maxdiff={base_logit_gap:.9g} base_forecast_maxdiff={base_forecast_gap:.9g}"
    )


@pytest.mark.parametrize("missing", [(1,), (2,), (1, 2)], ids=["CIII", "MNM", "CIII_MNM"])
def test_synthetic_missing_mask_is_a_static_truthful_contrast(source, missing):
    base, original, reference, head, basis = source
    x = base.center.detach().reshape(1, 1, 23).expand(1, 128, 23).clone()
    available = torch.ones_like(x, dtype=torch.bool)
    available[:, :8] = False  # all-missing patch, including Bt
    available[:, 8:, list(missing)] = False
    before_x, before_available = x.clone(), available.clone()
    patch = torch.zeros(1, 16, 8, 46, dtype=x.dtype)
    patch[..., 23:] = available.reshape(1, 16, 8, 23)
    flat = patch.flatten(-2)
    weight = base.embedding.weight.detach()
    alternate = weight.clone()
    for p in range(8):
        bt, ciii, mnm = p * 46 + 23, p * 46 + 24, p * 46 + 25
        alternate[:, bt] += weight[:, ciii] + weight[:, mnm]
        alternate[:, ciii] = 0
        alternate[:, mnm] = 0
    expected = F.linear(flat, alternate) - F.linear(flat, weight)
    actual = mask_embedding_shift(flat, weight)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
    assert not actual[:, 0].any() and actual[:, 1:].abs().max() > 0
    with torch.no_grad():
        h0, valid0 = original.tokens(x, available)
        h1, valid1 = reference.tokens(x, available)
        signal = {"values": x[0].numpy().copy(), "available": available[0].numpy().copy()}
        pack0 = encode(original, signal, [127], basis)
        pack1 = encode(reference, signal, [127], basis)
        out0 = forward(original, pack0, torch.zeros(4, 64), forecast=True)
        out1 = forward(reference, pack1, torch.zeros(4, 64), forecast=True)
        forecast0, _ = run_external(original, pack0, head)
        forecast1, _ = run_external(reference, pack1, head)
    assert torch.equal(valid0, valid1) and not valid0[0, 0] and valid0[0, -1]
    assert not torch.equal(h0, h1)
    assert not torch.equal(out0["logit"], out1["logit"])
    assert not torch.equal(forecast0, forecast1)
    assert torch.equal(x, before_x) and torch.equal(available, before_available)
    print(
        f"synthetic missing={missing} shift_max={actual.abs().max().item():.9g} "
        f"token_maxdiff={(h1-h0).abs().max().item():.9g} "
        f"logit_maxdiff={(out1['logit']-out0['logit']).abs().max().item():.9g} "
        f"external_forecast_maxdiff={(forecast1-forecast0).abs().max().item():.9g} valid_unchanged=true"
    )


def test_reference_embedding_hook_is_removed_on_exception(source, monkeypatch):
    base, _, reference, _, _ = source
    sentinel = base.embedding.register_forward_hook(lambda *_args: None)
    before = tuple(base.embedding._forward_hooks)

    def fail(*_args, **_kwargs):
        raise RuntimeError("deliberate embedding-path failure")

    try:
        monkeypatch.setattr(ContextAdapter, "tokens", fail)
        with pytest.raises(RuntimeError, match="deliberate embedding-path failure"):
            reference.tokens(torch.zeros(1, 128, 23), torch.ones(1, 128, 23, dtype=torch.bool))
        assert tuple(base.embedding._forward_hooks) == before
    finally:
        sentinel.remove()


def test_external_decoder_hook_is_removed_on_exception(source, monkeypatch):
    base, _, reference, head, _ = source
    sentinel = base.forecast_head.register_forward_pre_hook(lambda *_args: None)
    before = tuple(base.forecast_head._forward_pre_hooks)

    def fail(*_args, **_kwargs):
        raise RuntimeError("deliberate decoder-path failure")

    try:
        monkeypatch.setattr(decoder_core, "forward", fail)
        with pytest.raises(RuntimeError, match="deliberate decoder-path failure"):
            decoder_core.prediction(reference, {}, torch.zeros(4, 64), head)
        assert tuple(base.forecast_head._forward_pre_hooks) == before
    finally:
        sentinel.remove()
