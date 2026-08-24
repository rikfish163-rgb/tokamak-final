from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "experiments" / "transfer_methods_2026" / "jtext_online_adapter.py"


def load_module():
    assert MODULE.is_file(), "J-TEXT online adapter production module is missing"
    spec = importlib.util.spec_from_file_location("jtext_online_adapter", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def make_intra_batch(module, *, offset: float = 0.35):
    return module.JTextIntraBatch(
        source_rebuilt=torch.tensor(
            [[0.0, 0.5, -0.25], [0.2, -0.1, 0.4]], dtype=torch.float32,
        ),
        source_logit=torch.tensor([-0.3, 0.2], dtype=torch.float32),
        source_available=torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool,
        ),
        observed_rebuilt=torch.tensor(
            [[0.2, 0.8, -0.25], [0.4, 0.0, 0.7]], dtype=torch.float32,
        ),
        observed_source_logit=torch.tensor(
            [-0.3 + offset, 0.2 + offset], dtype=torch.float32,
        ),
        observed_available=torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool,
        ),
    )


def make_inter_batch(module, *, label: float = 1.0, nonfinite: bool = False):
    labels = torch.tensor([label, label, label, label], dtype=torch.float32)
    if nonfinite:
        labels[0] = float("nan")
    return module.JTextInterBatch(
        rebuilt=torch.tensor(
            [
                [0.0, 0.5, -0.25],
                [0.2, -0.1, 0.4],
                [0.4, 0.0, 0.7],
                [0.8, -0.2, 0.9],
            ],
            dtype=torch.float32,
        ),
        source_logit=torch.tensor([-0.7, -0.2, 0.1, 0.4], dtype=torch.float32),
        available=torch.tensor(
            [
                [True, True, False],
                [True, True, True],
                [True, True, True],
                [True, True, True],
            ],
            dtype=torch.bool,
        ),
        label=labels,
    )


def make_controller(module, *, seed: int = 7):
    torch.manual_seed(seed)
    encoder = nn.Sequential(nn.Linear(3, 4), nn.GELU())
    head = nn.Linear(4, 1)
    residual = module.JTextOnlineResidual(
        feature_count=3, hidden_size=8, max_logit_shift=0.25,
    )
    controller = module.JTextOnlineController(
        residual,
        source_encoder=encoder,
        source_head=head,
        intra_lr=0.02,
        inter_lr=0.01,
        loss_weights={
            "next": 1.0,
            "teacher": 0.5,
            "risk_anchor": 0.25,
            "physics": 0.0,
            "supervised": 1.0,
            "replay": 0.5,
        },
    )
    return controller, encoder, head


def assert_loss_receipts(receipt: dict[str, Any], expected: set[str]) -> None:
    assert set(receipt["losses"]) == expected
    for term in receipt["losses"].values():
        assert set(term) == {"enabled", "weight", "raw", "weighted"}
        assert isinstance(term["enabled"], bool)
        assert isinstance(term["weight"], float)
        assert term["raw"] is None or isinstance(term["raw"], float)
        assert term["weighted"] is None or isinstance(term["weighted"], float)
        if not term["enabled"]:
            assert term["weighted"] == 0.0


def test_zero_initialized_residual_is_bitwise_source_and_bounded() -> None:
    module = load_module()
    torch.manual_seed(3)
    residual = module.JTextOnlineResidual(
        feature_count=3, hidden_size=6, max_logit_shift=0.125,
    )
    rebuilt = torch.randn(11, 3)
    source_logit = torch.linspace(-2.0, 2.0, 11)
    available = torch.ones_like(rebuilt, dtype=torch.bool)

    adjusted = residual(rebuilt, source_logit, available)

    assert torch.equal(adjusted, source_logit)
    assert residual.online_parameter_count == sum(
        parameter.numel() for parameter in residual.parameters()
        if parameter.requires_grad
    )

    with torch.no_grad():
        residual.residual_head.bias.fill_(100.0)
    shifted = residual(rebuilt, source_logit, available)
    assert torch.all(torch.abs(shifted - source_logit) <= 0.125 + 1e-7)


def test_begin_shot_resets_fast_copy_to_exact_persistent_state() -> None:
    module = load_module()
    controller, _encoder, _head = make_controller(module)
    controller.begin_shot(10)
    receipt = controller.intra_update(
        make_intra_batch(module), source_endpoint=3, observed_endpoint=4,
    )
    assert receipt["updated"] is True
    assert receipt["fast_sha256"] != receipt["persistent_sha256"]
    controller.seal_shot(10)
    controller.discard_unqueried(10)

    begin = controller.begin_shot(11)

    assert begin["fast_sha256"] == begin["persistent_sha256"]


@pytest.mark.parametrize(
    "forbidden",
    ["label", "target", "outcome", "DownTime", "AdvancedTime", "TTD", "horizon"],
)
def test_intra_update_rejects_every_outcome_bearing_field(forbidden: str) -> None:
    module = load_module()
    controller, _encoder, _head = make_controller(module)
    controller.begin_shot(10)
    batch = dict(vars(make_intra_batch(module)))
    batch[forbidden] = torch.tensor([1.0])

    with pytest.raises(ValueError, match="outcome-bearing intra field"):
        controller.intra_update(batch, source_endpoint=3, observed_endpoint=4)


def test_intra_update_is_strictly_later_and_reports_all_loss_terms() -> None:
    module = load_module()
    controller, _encoder, _head = make_controller(module)
    controller.begin_shot(10)

    with pytest.raises(ValueError, match="source_endpoint < observed_endpoint"):
        controller.intra_update(
            make_intra_batch(module), source_endpoint=4, observed_endpoint=4,
        )
    receipt = controller.intra_update(
        make_intra_batch(module), source_endpoint=3, observed_endpoint=4,
    )

    assert receipt["source_endpoint"] == 3
    assert receipt["observed_endpoint"] == 4
    assert_loss_receipts(
        receipt, {"next", "teacher", "risk_anchor", "physics"},
    )
    assert receipt["losses"]["physics"]["enabled"] is False


def test_source_encoder_and_head_are_bitwise_frozen_across_updates() -> None:
    module = load_module()
    controller, encoder, head = make_controller(module)
    source_before = controller.source_hashes
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert all(not parameter.requires_grad for parameter in head.parameters())
    controller.begin_shot(1)
    controller.intra_update(
        make_intra_batch(module), source_endpoint=0, observed_endpoint=1,
    )
    controller.seal_shot(1)
    controller.inter_update(
        1, current=make_inter_batch(module), replay=None,
        guard_batch=make_inter_batch(module, label=0.0), steps=1,
    )

    assert controller.source_hashes == source_before
    controller.assert_source_frozen()


def test_first_selected_shot_allows_no_replay_and_disables_replay_loss() -> None:
    module = load_module()
    controller, _encoder, _head = make_controller(module)
    controller.begin_shot(1)
    controller.seal_shot(1)

    receipt = controller.inter_update(
        1, current=make_inter_batch(module), replay=None,
        guard_batch=make_inter_batch(module, label=0.0), steps=1,
    )

    assert receipt["accepted"] is True
    assert receipt["rollback_reason"] is None
    assert_loss_receipts(
        receipt, {"supervised", "replay", "risk_anchor", "physics"},
    )
    assert receipt["losses"]["replay"]["enabled"] is False


def snapshot_transaction(controller):
    return (
        copy.deepcopy(controller.persistent.state_dict()),
        copy.deepcopy(controller.inter_optimizer.state_dict()),
    )


def assert_transaction_restored(controller, before) -> None:
    model_before, optimizer_before = before
    assert_nested_equal(controller.persistent.state_dict(), model_before)
    assert_nested_equal(controller.inter_optimizer.state_dict(), optimizer_before)


def test_nonfinite_loss_restores_model_and_optimizer() -> None:
    module = load_module()
    controller, _encoder, _head = make_controller(module)
    controller.begin_shot(1)
    controller.seal_shot(1)
    before = snapshot_transaction(controller)

    receipt = controller.inter_update(
        1, current=make_inter_batch(module, nonfinite=True), replay=None,
        guard_batch=make_inter_batch(module, label=0.0), steps=1,
    )

    assert receipt["accepted"] is False
    assert receipt["rollback_reason"] == "nonfinite_loss"
    assert_transaction_restored(controller, before)


def test_nonfinite_gradient_restores_model_and_optimizer() -> None:
    module = load_module()
    controller, _encoder, _head = make_controller(module)
    controller.begin_shot(1)
    controller.seal_shot(1)
    parameter = next(controller.persistent.parameters())
    hook = parameter.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
    before = snapshot_transaction(controller)
    try:
        receipt = controller.inter_update(
            1, current=make_inter_batch(module), replay=None,
            guard_batch=make_inter_batch(module, label=0.0), steps=1,
        )
    finally:
        hook.remove()

    assert receipt["accepted"] is False
    assert receipt["rollback_reason"] == "nonfinite_gradient"
    assert_transaction_restored(controller, before)


def test_nonfinite_parameter_restores_model_and_optimizer() -> None:
    module = load_module()
    controller, _encoder, _head = make_controller(module)
    controller.begin_shot(1)
    controller.seal_shot(1)
    original_step = controller.inter_optimizer.step

    def poison_parameter(*args, **kwargs):
        result = original_step(*args, **kwargs)
        with torch.no_grad():
            next(controller.persistent.parameters()).view(-1)[0] = float("nan")
        return result

    controller.inter_optimizer.step = poison_parameter
    before = snapshot_transaction(controller)
    receipt = controller.inter_update(
        1, current=make_inter_batch(module), replay=None,
        guard_batch=make_inter_batch(module, label=0.0), steps=1,
    )

    assert receipt["accepted"] is False
    assert receipt["rollback_reason"] == "nonfinite_parameter"
    assert_transaction_restored(controller, before)


def warm_adam(controller, module, shot: int = 1) -> None:
    controller.begin_shot(shot)
    controller.seal_shot(shot)
    receipt = controller.inter_update(
        shot, current=make_inter_batch(module), replay=None,
        guard_batch=make_inter_batch(module, label=0.0), steps=1,
    )
    assert receipt["accepted"] is True


def test_nonfinite_optimizer_state_restores_model_and_optimizer() -> None:
    module = load_module()
    controller, _encoder, _head = make_controller(module)
    warm_adam(controller, module)
    controller.begin_shot(2)
    controller.seal_shot(2)
    original_step = controller.inter_optimizer.step

    def poison_optimizer(*args, **kwargs):
        result = original_step(*args, **kwargs)
        for state in controller.inter_optimizer.state.values():
            state["exp_avg"].view(-1)[0] = float("nan")
            break
        return result

    controller.inter_optimizer.step = poison_optimizer
    before = snapshot_transaction(controller)
    receipt = controller.inter_update(
        2, current=make_inter_batch(module, label=0.0),
        replay=make_inter_batch(module),
        guard_batch=make_inter_batch(module, label=0.0), steps=1,
    )

    assert receipt["accepted"] is False
    assert receipt["rollback_reason"] == "nonfinite_optimizer_state"
    assert_transaction_restored(controller, before)


def test_guard_rejection_restores_model_and_optimizer() -> None:
    module = load_module()
    controller, _encoder, _head = make_controller(module)
    warm_adam(controller, module)
    controller.begin_shot(2)
    controller.seal_shot(2)
    before = snapshot_transaction(controller)

    receipt = controller.inter_update(
        2, current=make_inter_batch(module, label=0.0),
        replay=make_inter_batch(module),
        guard_batch=make_inter_batch(module, label=0.0), steps=1,
        guard=lambda _before, _after: False,
    )

    assert receipt["accepted"] is False
    assert receipt["rollback_reason"] == "guard_rejected"
    assert_transaction_restored(controller, before)


def test_rejected_adam_update_cannot_change_the_next_identical_update() -> None:
    module = load_module()
    rejected, _encoder, _head = make_controller(module, seed=13)
    direct, _encoder2, _head2 = make_controller(module, seed=13)
    warm_adam(rejected, module)
    warm_adam(direct, module)

    rejected.begin_shot(2)
    rejected.seal_shot(2)
    rollback = rejected.inter_update(
        2, current=make_inter_batch(module, label=0.0),
        replay=make_inter_batch(module),
        guard_batch=make_inter_batch(module, label=0.0), steps=1,
        guard=lambda _before, _after: False,
    )
    assert rollback["accepted"] is False

    rejected.begin_shot(3)
    rejected.seal_shot(3)
    accepted_after_rollback = rejected.inter_update(
        3, current=make_inter_batch(module, label=0.0),
        replay=make_inter_batch(module),
        guard_batch=make_inter_batch(module, label=0.0), steps=1,
    )
    direct.begin_shot(2)
    direct.seal_shot(2)
    accepted_direct = direct.inter_update(
        2, current=make_inter_batch(module, label=0.0),
        replay=make_inter_batch(module),
        guard_batch=make_inter_batch(module, label=0.0), steps=1,
    )

    assert accepted_after_rollback["accepted"] is True
    assert accepted_direct["accepted"] is True
    assert_nested_equal(
        rejected.persistent.state_dict(), direct.persistent.state_dict(),
    )
    assert_nested_equal(
        rejected.inter_optimizer.state_dict(), direct.inter_optimizer.state_dict(),
    )


def test_resume_checkpoint_is_written_only_at_finalized_shot_boundary(
    tmp_path: Path,
) -> None:
    module = load_module()
    controller, _encoder, _head = make_controller(module)
    target = tmp_path / "resume.pt"
    controller.begin_shot(1)
    with pytest.raises(RuntimeError, match="finalized shot boundary"):
        controller.save_resume_checkpoint(target, extra_state={"cursor": 0})
    controller.seal_shot(1)
    with pytest.raises(RuntimeError, match="finalized shot boundary"):
        controller.save_resume_checkpoint(target, extra_state={"cursor": 0})
    controller.discard_unqueried(1)

    receipt = controller.save_resume_checkpoint(target, extra_state={"cursor": 1})
    payload = torch.load(target, map_location="cpu", weights_only=True)

    assert receipt["last_finalized_shot"] == 1
    assert payload["schema_version"] == "jtext-online-controller-resume-v1"
    assert payload["extra_state"] == {"cursor": 1}
    assert payload["last_finalized_shot"] == 1
