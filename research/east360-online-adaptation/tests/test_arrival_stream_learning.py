import copy

import numpy as np
import pytest
import torch
from torch import nn

from experiments.risk_aligned_online.arrival_reference_20261030.run import ARMS
from experiments.risk_aligned_online.arrival_stream_20261030 import learning


class ToyNet(nn.Module):
    """A real differentiable Q/K input and a hookable 17-by-64 readout."""

    def __init__(self, *, zero_gradient=False):
        super().__init__()
        self.base = nn.Module()
        self.base.register_buffer("center", torch.zeros(23))
        self.base.register_buffer("scale", torch.ones(23))
        self.base.forecast_head = nn.Linear(17 * 64, 1, bias=False)
        self.zero_gradient = zero_gradient
        self.calls = []

    def forward(self, x, available, state, slow, mode, *, forecast=True):
        assert mode == "qk"
        assert available.dtype == torch.bool
        factor = 0.0 if self.zero_gradient else 1.0
        delta = state[:, 0] * factor
        slow_value = 0.0 if slow is None else slow.reshape(-1)[0]
        logit = x[:, -1, 0] + delta + slow_value
        cls = x[:, -1, 0] + 100 * state[:, 0] + slow_value
        tokens = torch.cat((x.new_zeros(len(x), 16 * 64),
                            cls[:, None].expand(-1, 64)), dim=1)
        self.calls.append({"forecast": forecast,
                           "ends": x[:, -1, 2].detach().cpu().numpy().copy(),
                           "state": state[:, 0].detach().cpu().numpy().copy()})
        if forecast:
            self.base.forecast_head(tokens)
        return {"logit": logit}


def signal(n=210):
    values = np.zeros((n, 23), dtype=np.float32)
    values[:, 0] = np.arange(n, dtype=np.float32) / 200
    values[:, 2] = np.arange(n, dtype=np.float32)
    return {"values": values, "available": np.ones_like(values, dtype=bool)}


def fits():
    q1 = np.zeros(64 + 23 + 23 * 4 + 1)
    q1[64 + 23] = 15.0  # Mean arrived increment of channel 0.
    pair = {"prefix_center": np.zeros(64), "prefix_scale": np.ones(64),
            "arrival_scale": np.ones(23 * 4), "q0_coef": np.zeros(64 + 23 + 1),
            "q1_coef": q1}
    channels = [{"coefficients": np.zeros((65, 4)), "mean": np.zeros(4)}
                for _ in range(23)]
    reference = {"supported": np.array([True] + [False] * 22), "channels": channels}
    return pair, copy.deepcopy(pair), reference, copy.deepcopy(reference)


def run(s, net=None, *, slow=None):
    net = ToyNet() if net is None else net
    result = learning.predict(net, s, slow, *fits())
    return net, *result


def risk_state(net, arm, point):
    names = ("cold",) + tuple(ARMS)
    for call in net.calls:
        if call["forecast"]:
            continue
        count = len(call["ends"]) // len(names)
        ends = call["ends"].reshape(len(names), count)
        state = call["state"].reshape(len(names), count)
        matches = np.flatnonzero(ends[names.index(arm)] == point)
        if len(matches):
            return float(state[names.index(arm), matches[0]])
    raise AssertionError(f"point {point} was not forwarded for {arm}")


def test_full_stream_shapes_cadence_old_state_and_replace():
    net, scores, receipts = run(signal(210), slow=torch.tensor([.25]))
    assert set(scores) == set(ARMS) | {"cold"}
    for value in scores.values():
        assert value.shape == (210,)
        assert np.isfinite(value).all() and ((0 <= value) & (value <= 1)).all()
    np.testing.assert_array_equal(scores["identical"], scores["cold"])
    assert [(r["origin"], r["maturity"], r["first_affected"], r["apply_count"])
            for r in receipts] == [(127, 152, 153, 25), (152, 177, 178, 25),
                                   (177, 202, 203, 7)]
    assert all(r["support_count"] == 1 for r in receipts)
    assert all(r["gradnorm"] == pytest.approx(1) for r in receipts)
    assert all(r["state_norm"]["real"] > 0 for r in receipts)
    assert all(r["state_norm"]["cold"] == r["state_norm"]["identical"] == 0
               for r in receipts)
    assert risk_state(net, "real", 152) == 0
    first = -.05 * receipts[0]["feedback"]["real"]
    second = -.05 * receipts[1]["feedback"]["real"]
    assert risk_state(net, "real", 153) == pytest.approx(first)
    assert risk_state(net, "real", 177) == pytest.approx(first)
    assert risk_state(net, "real", 178) == pytest.approx(second)
    assert second != pytest.approx(first + second)
    # Origin 152 is scored with the old state but independently anchored at zero.
    anchors = [call for call in net.calls if call["forecast"]]
    assert [int(c["ends"][0]) for c in anchors] == [127, 152, 177, 202]
    assert all(c["state"][0] == 0 for c in anchors)
    assert receipts[1]["origin_logit"] == pytest.approx(.25 + 152 / 200)
    assert all(len(c["ends"]) <= 9 * 32 for c in net.calls if not c["forecast"])


def test_future_perturbation_and_shortened_prefix_never_rewrite_history():
    original = signal(220)
    altered = copy.deepcopy(original)
    altered["values"][165:, 0] += 20
    altered["available"][165:, 0] = False
    _, before, original_receipts = run(original)
    _, after, altered_receipts = run(altered)
    for arm in before:
        np.testing.assert_array_equal(before[arm][:165], after[arm][:165])
    assert original_receipts[0] == altered_receipts[0]
    assert original_receipts[1]["origin_logit"] == altered_receipts[1]["origin_logit"]
    _, short, short_receipts = run({k: v[:165] for k, v in original.items()})
    for arm in before:
        np.testing.assert_array_equal(before[arm][:165], short[arm])
    assert short_receipts[0]["apply_count"] == 12
    assert short_receipts[0]["e"] == original_receipts[0]["e"]


@pytest.mark.parametrize("length, receipts_expected, applied", [
    (1, 0, []), (127, 0, []), (128, 0, []), (152, 0, []),
    (153, 1, [0]), (154, 1, [1]), (160, 1, [7]),
    (177, 1, [24]), (178, 2, [25, 0]), (180, 2, [25, 2]),
])
def test_incomplete_horizon_and_partial_tails(length, receipts_expected, applied):
    _, scores, receipts = run(signal(length))
    assert all(len(value) == length for value in scores.values())
    assert len(receipts) == receipts_expected
    assert [r["apply_count"] for r in receipts] == applied
    if length > 153:
        assert scores["real"][153] != scores["cold"][153]


def test_unsupported_maturity_replaces_previous_state_with_zero():
    s = signal(190)
    s["available"][153:178, 0] = False
    net, scores, receipts = run(s)
    assert receipts[0]["support_count"] == 1
    assert receipts[1]["support_count"] == 0
    assert receipts[0]["state_norm"]["real"] > 0
    assert receipts[1]["state_norm"]["real"] == 0
    assert receipts[1]["feedback"]["real"] == 0
    assert risk_state(net, "real", 177) != 0
    assert risk_state(net, "real", 178) == 0
    np.testing.assert_array_equal(scores["real"][178:], scores["cold"][178:])


def test_zero_gradient_has_zero_update_with_supported_feedback():
    net, scores, receipts = run(signal(180), ToyNet(zero_gradient=True))
    assert receipts[0]["support_count"] == 1
    assert receipts[0]["gradnorm"] == 0
    assert receipts[0]["state_norm"]["real"] == 0
    assert receipts[0]["feedback"]["real"] != 0
    np.testing.assert_array_equal(scores["real"], scores["cold"])


def test_reset_and_no_parameter_gradients_between_shots():
    net = ToyNet()
    snapshot = {name: value.clone() for name, value in net.state_dict().items()}
    _, _, first = run(signal(180), net)
    _, second_scores, second = run(signal(160), net)
    _, reference_scores, reference = run(signal(160))
    for arm in second_scores:
        np.testing.assert_array_equal(second_scores[arm], reference_scores[arm])
    assert second == reference
    assert first[0]["e"] == second[0]["e"]
    assert all(p.grad is None for p in net.parameters())
    for name, value in net.state_dict().items():
        torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)


def test_real_helpers_receive_only_arrived_prefix_and_zero_qk_cls(monkeypatch):
    original_windows, original_scores = learning.signal_windows, learning.make_scores
    calls = []

    def windows(net, prefix, ends):
        # The helper sees exactly origin..maturity, with a local zero index.
        assert len(prefix["values"]) == 26 and ends[0] == 0
        assert prefix["values"][0, 2] in (127, 152)
        result = original_windows(net, prefix, ends)
        assert not result[1][0, 10]
        return result

    def make_scores(pair, wrong_pair, ref, wrong_ref, x, target, valid):
        calls.append((x.copy(), target.copy(), valid.copy()))
        return original_scores(pair, wrong_pair, ref, wrong_ref, x, target, valid)

    monkeypatch.setattr(learning, "signal_windows", windows)
    monkeypatch.setattr(learning, "make_scores", make_scores)
    _, _, receipts = run(signal(180), slow=torch.tensor([.25]))
    assert len(calls) == len(receipts) == 2
    for origin, (feature, target, valid) in zip((127, 152), calls):
        np.testing.assert_allclose(feature, np.full((1, 64), origin / 200 + .25), atol=1e-7)
        assert target.shape == (1, 25, 23) and valid.shape == (1, 23)
        expected = np.arcsinh((origin + np.arange(1, 26)) / 200) - np.arcsinh(origin / 200)
        np.testing.assert_allclose(target[0, :, 0], expected, atol=1e-7)


@pytest.mark.parametrize("bad", [
    lambda s: dict(s, outcome=1),
    lambda s: {**s, "available": s["available"].astype(int)},
    lambda s: {**s, "available": s["available"][:-1]},
    lambda s: {**s, "values": s["values"][:, :22], "available": s["available"][:, :22]},
    lambda s: {"values": s["values"][:0], "available": s["available"][:0]},
])
def test_invalid_or_outcome_bearing_signal_is_rejected(bad):
    with pytest.raises(ValueError):
        run(bad(signal(160)))


def test_actual_matrix_attention_cpu_smoke():
    from experiments.risk_aligned_online.attention_basis_20261029.learning import MatrixAttention
    from experiments.risk_aligned_online.simple23 import Config, Simple23

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(41)
        net = MatrixAttention(Simple23(Config(), np.zeros(23), np.ones(23)))
        before = {name: value.clone() for name, value in net.state_dict().items()}
        _, scores, receipts = run(signal(155), net, slow=torch.zeros((64, 64)))
        np.testing.assert_array_equal(scores["identical"], scores["cold"])
        assert all(v.shape == (155,) and np.isfinite(v).all() for v in scores.values())
        assert len(receipts) == 1 and receipts[0]["apply_count"] == 2
        assert receipts[0]["gradnorm"] > 0
        for name, value in net.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    finally:
        torch.set_num_threads(threads)


def test_actual_matrix_attention_exact_parity_and_prefix_across_tail_sizes():
    from experiments.risk_aligned_online.attention_basis_20261029.learning import MatrixAttention
    from experiments.risk_aligned_online.simple23 import Config, Simple23

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(41)
        net = MatrixAttention(Simple23(Config(), np.zeros(23), np.ones(23)))
        slow = torch.zeros((64, 64))
        _, full, _ = run(signal(180), net, slow=slow)
        for length in (1, 5, 31, 127, 156, 176):
            _, prefix, _ = run(signal(length), net, slow=slow)
            for arm in prefix:
                np.testing.assert_array_equal(prefix[arm], full[arm][:length])
            np.testing.assert_array_equal(prefix["cold"], prefix["identical"])
    finally:
        torch.set_num_threads(threads)
