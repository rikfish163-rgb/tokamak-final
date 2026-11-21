import numpy as np
import pytest
import torch

import run_whole_block_update_20261106 as run
import verify_whole_block_update_20261106 as verify


def test_training_tape_keeps_left_padding_at_each_shot():
    rows = torch.zeros(2 * 129, 23)
    rows[127:129] = 1
    rows[-2:] = 2
    mask = rows != 0
    x, a = run.tape_views(dict(values=rows, available=mask), torch.tensor([127, 256]))
    assert x.shape == a.shape == (2, 128, 23)
    assert not a[:, :-1].any()
    torch.testing.assert_close(x[:, -1, 0], torch.tensor([1.0, 2.0]))


def test_training_batch_draw_is_repeatable_and_exactly_stratified():
    data = dict(shot=torch.tensor([1, 1, 2, 3, 4]), role=torch.tensor([0, 0, 0, 1, 2]))
    mask = torch.tensor([True, False, True, True, True])
    draw = run.draw_batches(data, mask, 317, steps=4)
    assert draw.shape == (4, 16)
    assert np.isin(draw[:, :8], (0, 2)).all()
    assert (draw[:, 8:12] == 3).all()
    assert (draw[:, 12:] == 4).all()
    assert np.array_equal(draw, run.draw_batches(data, mask, 317, steps=4))


def pred_fixture():
    n = 98
    a = dict(
        time=np.arange(n) / 1000,
        logits=np.zeros((n, 3, 6), dtype=np.float32),
        active=np.arange(n) >= 64,
        maturity=np.array([63, 95]),
        common=np.ones(2, dtype=bool),
        norms=np.ones((2, 3, 3)) * 0.1,
        cos=np.zeros((2, 3)),
        offsets=np.full((2, 3), 0.25),
    )
    a["logits"][64:, :, 5] = 0.25
    return a, {k: v.copy() for k, v in a.items()}, np.ones((n, 23), dtype=bool)


def test_full_trace_verifier_reconstructs_causal_and_score_support():
    a, original, mask = pred_fixture()
    assert verify.verify_arrays(a, original, a["time"], mask) == 0


@pytest.mark.parametrize("violation", ["past_write", "offset", "wrong_support", "norm"])
def test_full_trace_verifier_rejects_synthetic_contract_drift(violation):
    a, original, mask = pred_fixture()
    if violation == "past_write":
        a["logits"][63, 0, 3] = 0.9
    elif violation == "offset":
        a["logits"][66, 0, 5] = 0.26
    elif violation == "wrong_support":
        mask[:32, :] = False
    else:
        a["norms"][0, 0, 2] = 0.6
    with pytest.raises(ValueError):
        verify.verify_arrays(a, original, a["time"], mask)
