import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import latent_trajectory_adapt_20261105 as latent
import moving_peak_replay_20261105 as original
import moving_peak_interaction_fit_20261105 as fit


def test_archived_receipt_and_initial_checkpoint_are_bound_by_digest():
    root = Path(__file__).resolve().parents[1]
    receipt = root / "probe_outputs/moving_peak_replay360_20261105_v1/fit_receipt.json"
    initial = root / "probe_outputs/latent_trajectory_adapt360_20261105_v1/fit20.npz"
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == fit.FIXED_RECEIPT_SHA256
    assert hashlib.sha256(initial.read_bytes()).hexdigest() == fit.INITIAL_FIT_SHA256
    contents = json.loads(receipt.read_bytes())
    assert contents["initial_fit_sha256"] == fit.INITIAL_FIT_SHA256
    assert len(contents["ids"]) == 20
    assert len(contents["arms"]["moving_peak"]["rounds"]) == 3


def test_positive_interaction_is_penalized_only_for_safe_examples():
    z11 = torch.tensor([2.0, -2.0, 5.0], requires_grad=True)
    z10 = torch.zeros(3, requires_grad=True)
    z01 = torch.zeros(3, requires_grad=True)
    z00 = torch.zeros(3)
    weights = torch.tensor([0.25, 0.25, 0.5])
    safe = torch.tensor([True, True, False])

    penalty = fit._interaction_penalty(z11, z10, z01, z00, weights, safe)
    assert penalty.item() == pytest.approx(1.0)
    penalty.backward()
    torch.testing.assert_close(z11.grad, torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(z10.grad, torch.tensor([-1.0, 0.0, 0.0]))
    torch.testing.assert_close(z01.grad, torch.tensor([-1.0, 0.0, 0.0]))


def test_negative_infinite_interaction_is_not_hidden_by_relu():
    large = torch.finfo(torch.float32).max
    with pytest.raises(FloatingPointError, match="nonfinite"):
        fit._interaction_penalty(
            torch.tensor([-large]),
            torch.tensor([large]),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            torch.tensor([1.0]),
            torch.tensor([True]),
        )


def test_four_exact_routed_calls_retain_layer_actions_and_freeze_z00(monkeypatch):
    calls = []

    def routed(base, x, a, routes, delta):
        calls.append(delta)
        return delta[:, :128].sum(1) + 2 * delta[:, 128:].sum(1)

    monkeypatch.setattr(fit.policy, "routed_pair", routed)
    monkeypatch.setattr(fit.latent, "routes_all", lambda a: a)
    delta = torch.arange(256.0).reshape(1, 256).requires_grad_()
    z11, z10, z01, z00 = fit._routed_logits(None, None, None, delta)

    assert len(calls) == 4
    torch.testing.assert_close(calls[0], delta)
    torch.testing.assert_close(calls[1][:, :128], delta[:, :128])
    torch.testing.assert_close(calls[1][:, 128:], torch.zeros(1, 128))
    torch.testing.assert_close(calls[2][:, :128], torch.zeros(1, 128))
    torch.testing.assert_close(calls[2][:, 128:], delta[:, 128:])
    assert torch.count_nonzero(calls[3]) == 0
    assert not z00.requires_grad
    assert z11.requires_grad and z10.requires_grad and z01.requires_grad
    (g10,) = torch.autograd.grad(z10.sum(), delta, retain_graph=True)
    (g01,) = torch.autograd.grad(z01.sum(), delta)
    assert torch.count_nonzero(g10[:, :128]) == 128
    assert torch.count_nonzero(g10[:, 128:]) == 0
    assert torch.count_nonzero(g01[:, :128]) == 0
    assert torch.count_nonzero(g01[:, 128:]) == 128


def test_real_routed_layers_match_original_full_forward_and_preserve_gradients():
    from experiments.risk_aligned_online.simple23 import Config, Simple23

    was_fast = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            base = (
                Simple23(
                    Config(fast_site="tokens", fast_rule="normalized"),
                    torch.zeros(23),
                    torch.ones(23),
                )
                .eval()
                .requires_grad_(False)
            )
            x = torch.randn(2, 128, 23)
            available = torch.ones_like(x, dtype=torch.bool)
            delta = (torch.randn(2, 256) * 0.02).requires_grad_()
            expected = original.policy.routed_pair(
                base, x, available, latent.routes_all(available), delta
            )
            z11, z10, z01, z00 = fit._routed_logits(base, x, available, delta)
            torch.testing.assert_close(z11, expected, rtol=0, atol=0)
            assert not z00.requires_grad
            for z in (z10, z01):
                (grad,) = torch.autograd.grad(z.sum(), delta, retain_graph=True)
                assert torch.isfinite(grad).all() and torch.count_nonzero(grad)
            assert all(p.grad is None for p in base.parameters())
    finally:
        torch.backends.mha.set_fastpath_enabled(was_fast)


class TinyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("center", torch.zeros(1))
        self.eval().requires_grad_(False)


def toy_routed(base, x, a, routes, delta):
    left = delta[:, :128].mean(1)
    right = delta[:, 128:].mean(1)
    return x[:, 0, 0] * 0.1 + left + 2 * right + 10 * left * right


@pytest.fixture
def fit_case(tmp_path, monkeypatch):
    torch.manual_seed(8)
    rule = latent.Generator(torch.zeros(194), torch.ones(194))
    with torch.no_grad():
        rule.layers[-1].weight.normal_(0, 0.01)
        rule.layers[-1].bias.fill_(0.02)
    initial = tmp_path / "initial.npz"
    np.savez_compressed(
        initial,
        **{"history." + k: v.detach().numpy() for k, v in rule.state_dict().items()},
    )
    initial_sha = hashlib.sha256(initial.read_bytes()).hexdigest()
    monkeypatch.setattr(fit, "INITIAL_FIT_SHA256", initial_sha)

    n = 240
    times = np.arange(n, dtype=np.float64) / 1000
    x = torch.linspace(-1, 1, n).reshape(n, 1, 1).expand(n, 128, 23)
    a = torch.ones(1, 128, 23, dtype=torch.bool).expand(n, -1, -1)
    p = np.zeros(n, dtype=np.float32)
    state = torch.linspace(-0.5, 0.5, 4 * 66).reshape(4, 66)
    history_features = torch.linspace(0.3, -0.4, 4 * 128).reshape(4, 128)
    ctx = (x, a, p, p, state, history_features, history_features)
    history = [
        (100 + i, ctx, times.copy(), i < 8, 0.255 if i < 8 else None) for i in range(20)
    ]
    rounds = []
    curves = []
    for stage in range(3):
        selection, by_shot = {}, {}
        for sid, _, clock, flag, down in history:
            adapted = np.zeros(n, dtype=np.float32)
            if not flag:
                for cell in np.array_split(np.arange(original.policy.FIRST, n), 8):
                    adapted[cell[(sid + stage) % len(cell)]] = 1.0
            q = original.mining_queries(clock, p, adapted, flag, down, "moving_peak")
            selection[str(sid)] = q.tolist()
            by_shot[sid] = adapted
        rounds.append({"round": stage + 1, "steps": 12, "selection": selection})
        curves.append(by_shot)
    receipt = {
        "ids": [row[0] for row in history],
        "initial_fit_sha256": initial_sha,
        "arms": {"moving_peak": {"rounds": rounds, "steps": 36}},
    }
    receipt_path = tmp_path / "fit_receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(
        fit,
        "FIXED_RECEIPT_SHA256",
        hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(fit.policy, "routed_pair", toy_routed)
    return dict(
        history=history,
        receipt=receipt,
        receipt_path=receipt_path,
        initial=initial,
        curves=curves,
        output=tmp_path / "output",
    )


def test_fit_has_identical_fixed_selection_budget_and_original_control(
    fit_case, monkeypatch
):
    case = fit_case
    case["output"].mkdir()
    base = TinyBase()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with monkeypatch.context() as guarded:

            def forbidden(*args, **kwargs):
                raise AssertionError("selection must not be remined")

            guarded.setattr(original, "forecast", forbidden)
            guarded.setattr(original, "mining_queries", forbidden)
            rules, hashes = fit.fit_interaction(
                base,
                case["history"],
                case["output"],
                case["initial"],
                case["receipt_path"],
            )
        assert (
            set(rules) == set(hashes) == {"interaction_control", "interaction_penalty"}
        )
        assert all(
            not p.requires_grad for rule in rules.values() for p in rule.parameters()
        )
        receipt = json.loads((case["output"] / "fit_receipt.json").read_text())
        assert (
            receipt["fixed_receipt_sha256"]
            == hashlib.sha256(case["receipt_path"].read_bytes()).hexdigest()
        )
        assert receipt["ids"] == case["receipt"]["ids"]
        assert [r["selection"] for r in receipt["selections"]] == [
            r["selection"] for r in case["receipt"]["arms"]["moving_peak"]["rounds"]
        ]
        for arm, digest in hashes.items():
            assert receipt["arms"][arm]["steps"] == 36
            assert [r["steps"] for r in receipt["arms"][arm]["rounds"]] == [12] * 3
            assert (
                digest
                == hashlib.sha256(
                    (case["output"] / f"fit_{arm}.npz").read_bytes()
                ).hexdigest()
            )
            for rd in receipt["arms"][arm]["rounds"]:
                assert all(
                    np.isfinite(list(rd[k].values())).all()
                    for k in ("initial", "final")
                )
        control_initial = receipt["arms"]["interaction_control"]["rounds"][0]["initial"]
        penalty_initial = receipt["arms"]["interaction_penalty"]["rounds"][0]["initial"]
        assert control_initial["matched"] == penalty_initial["matched"]
        assert (
            control_initial["interaction_penalty"]
            == penalty_initial["interaction_penalty"]
        )
        assert penalty_initial["interaction_penalty"] > 0
        assert any(
            not torch.equal(rules["interaction_control"].state_dict()[k], v)
            for k, v in rules["interaction_penalty"].state_dict().items()
        )

        # The control has exactly the original objective and update reduction order.
        expected = original.load_rule(
            case["initial"], base.center.device, namespace="history."
        )
        optimizer = torch.optim.Adam(expected.parameters(), lr=latent.LR)
        for stage, order in enumerate(original.shared_orders(160)):
            original.train_round(
                base,
                expected,
                optimizer,
                case["history"],
                case["curves"][stage],
                order,
                "moving_peak",
            )
        for name, value in expected.state_dict().items():
            assert torch.equal(
                value, rules["interaction_control"].state_dict()[name]
            ), name
    finally:
        torch.set_num_threads(threads)


def test_receipt_digest_and_history_order_fail_closed_before_writes(fit_case):
    case = fit_case
    case["output"].mkdir()
    case["receipt_path"].write_text(json.dumps(case["receipt"]) + " ")
    with pytest.raises(ValueError, match="receipt"):
        fit.fit_interaction(
            TinyBase(),
            case["history"],
            case["output"],
            case["initial"],
            case["receipt_path"],
        )
    assert list(case["output"].iterdir()) == []


def test_drifted_threshold_fails_closed_before_writes(fit_case, monkeypatch):
    case = fit_case
    case["output"].mkdir()
    monkeypatch.setattr(original, "THRESHOLD", 0.6)
    with pytest.raises(ValueError, match="constants"):
        fit.fit_interaction(
            TinyBase(),
            case["history"],
            case["output"],
            case["initial"],
            case["receipt_path"],
        )
    assert list(case["output"].iterdir()) == []


def test_nonfinite_routed_logits_fail_closed_before_writes(fit_case, monkeypatch):
    case = fit_case
    case["output"].mkdir()
    monkeypatch.setattr(
        fit.policy,
        "routed_pair",
        lambda base, x, a, routes, delta: delta[:, 0] * float("nan"),
    )
    with pytest.raises(FloatingPointError, match="nonfinite"):
        fit.fit_interaction(
            TinyBase(),
            case["history"],
            case["output"],
            case["initial"],
            case["receipt_path"],
        )
    assert list(case["output"].iterdir()) == []


def test_nonfinite_frozen_source_fails_closed_before_writes(fit_case):
    case = fit_case
    case["output"].mkdir()
    base = TinyBase()
    base.register_buffer("bad", torch.tensor([float("nan")]))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        fit.fit_interaction(
            base, case["history"], case["output"], case["initial"], case["receipt_path"]
        )
    assert list(case["output"].iterdir()) == []


@pytest.mark.parametrize(
    "corruption",
    ["order", "window", "positive_window", "class", "class_swap", "clock", "initial"],
)
def test_malformed_history_or_selection_fails_before_writes(
    fit_case, monkeypatch, corruption
):
    case = fit_case
    case["output"].mkdir()
    if corruption == "order":
        case["history"][0], case["history"][1] = case["history"][1], case["history"][0]
    elif corruption == "window":
        case["receipt"]["arms"]["moving_peak"]["rounds"][0]["selection"]["108"][0] = (
            original.policy.FIRST - 1
        )
    elif corruption == "positive_window":
        case["receipt"]["arms"]["moving_peak"]["rounds"][0]["selection"]["100"][1] += 1
    elif corruption == "class":
        sid, ctx, times, _, _ = case["history"][0]
        case["history"][0] = (sid, ctx, times, False, None)
    elif corruption == "class_swap":
        sid0, ctx0, times0, _, _ = case["history"][0]
        sid1, ctx1, times1, _, _ = case["history"][8]
        case["history"][0] = (sid0, ctx0, times0, False, None)
        case["history"][8] = (sid1, ctx1, times1, True, 0.255)
    elif corruption == "clock":
        case["history"][0][2][150] = np.nan
    elif corruption == "initial":
        case["receipt"]["initial_fit_sha256"] = "0" * 64
    case["receipt_path"].write_text(json.dumps(case["receipt"]))
    monkeypatch.setattr(
        fit,
        "FIXED_RECEIPT_SHA256",
        hashlib.sha256(case["receipt_path"].read_bytes()).hexdigest(),
    )
    with pytest.raises((ValueError, FloatingPointError)):
        fit.fit_interaction(
            TinyBase(),
            case["history"],
            case["output"],
            case["initial"],
            case["receipt_path"],
        )
    assert list(case["output"].iterdir()) == []
