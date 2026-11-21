import numpy as np
import torch

from experiments.risk_aligned_online.joint_lora_20261104.model import JointLoRA
from experiments.risk_aligned_online.joint_lora_20261104.timeline import run_shot
from experiments.risk_aligned_online.simple23 import Config, Simple23
from joint_lora_neutral_motion_20261104 import replay_loss


def test_loss_replay_matches_original_and_never_rewrites_prefix():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = Config(fast_site="tokens", fast_rule="normalized")
    base = (
        Simple23(config, torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    )
    model = JointLoRA(base, torch.zeros(168, 70)).eval().requires_grad_(False)
    with torch.no_grad():
        model.residual_decoder.weight.normal_(0, 0.02)
    rng = np.random.default_rng(19)
    values = rng.normal(size=(176, 23)).astype(np.float32)
    signal = dict(values=values, available=np.ones_like(values, dtype=bool))
    for mode in ("real", "hold", "reverse"):
        original, *_ = run_shot(model, signal, mode, device="cpu")
        actual, updates = replay_loss(model, signal, mode)
        assert updates == 2
        np.testing.assert_array_equal(original, actual)
    motion, n = replay_loss(model, signal, "motion")
    anti, _ = replay_loss(model, signal, "anti")
    assert n == 2
    np.testing.assert_array_equal(motion[:135], anti[:135])
    assert not np.array_equal(motion[135:], anti[135:])
