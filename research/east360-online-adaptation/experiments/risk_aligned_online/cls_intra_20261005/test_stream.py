import numpy as np
import pytest
import torch

from ..ffn_intra_20261005.test_meta import case
from .model import ClsModel
from .stream import predict_shot


def fixture():
    old, shot, _ = case()
    torch.manual_seed(123)
    model = ClsModel(old.base)
    shot = {k: v[:55].copy() if isinstance(v, np.ndarray) else v for k, v in shot.items()}
    shot['device'] = 'EAST'
    return model, shot


@pytest.mark.parametrize('use_task', [False, True])
def test_stream_causal_reset_and_first_effect(use_task):
    model, shot = fixture()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    original, audit = predict_shot(model, shot, use_task=use_task)
    static, _ = predict_shot(model, shot, use_task=use_task, intra=False)
    np.testing.assert_array_equal(original[:6], static[:6])
    assert np.max(np.abs(original[6:] - static[6:])) > 1e-7
    for cutoff in (5, 24, 25, 31):
        changed = dict(shot, values=shot['values'].copy(), available=shot['available'].copy(),
                       is_disrupt=False, down=-100.)
        changed['values'][cutoff + 1:] = 999.
        changed['available'][cutoff + 1:] = False
        future, _ = predict_shot(model, changed, use_task=use_task)
        np.testing.assert_array_equal(future[:cutoff + 1], original[:cutoff + 1])
    again, second = predict_shot(model, shot, use_task=use_task)
    np.testing.assert_array_equal(original, again)
    assert audit['final_delta_sha256'] == second['final_delta_sha256']
    assert [(e['update_index'], e['effective_index']) for e in audit['events']] == [(5, 6), (25, 26), (31, 32), (51, 52)]
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_jtext_requires_explicit_offline_access():
    model, shot = fixture(); shot['device'] = 'J-TEXT'
    with pytest.raises(ValueError):
        predict_shot(model, shot, use_task=True)
