import numpy as np
import torch

from experiments.risk_aligned_online import simple_data
from experiments.risk_aligned_online.simple23 import Simple23, Config
from experiments.risk_aligned_online.ffn_intra_20261005.model import FfnModel
from experiments.risk_aligned_online.ffn_intra_20261005.stream import predict_shot


def case(n=160):
    torch.set_num_threads(2)
    torch.manual_seed(9)
    model = FfnModel(Simple23(Config(fast_site='tokens'), torch.zeros(23), torch.ones(23)))
    rng = np.random.default_rng(3)
    shot = dict(device='EAST', shot=1, values=rng.normal(size=(n,23)).astype('float32'),
        available=np.ones((n,23),bool), times=np.arange(n)/1000.)
    return model, shot


def test_first_update_cannot_rewrite_maturity_prediction():
    model, shot = case()
    static, _ = predict_shot(model, shot, intra=False)
    adaptive, audit = predict_shot(model, shot, intra=True, stride=100, step=.03, radius=.3)
    np.testing.assert_allclose(static[:26], adaptive[:26], atol=2e-7, rtol=0)
    assert np.max(np.abs(static[26:]-adaptive[26:])) > 1e-6
    assert len(audit['events']) == 2
    assert audit['events'][0]['maturity_index'] == 25
    assert audit['events'][0]['effective_index'] == 26
    assert audit['events'][1]['effective_index'] == 126
    assert max(e['delta_norm'] for e in audit['events']) <= .300001


def test_future_signal_changes_never_change_old_outputs_or_reset():
    model, shot = case()
    original, _ = predict_shot(model, shot, intra=True)
    other = {k: v.copy() if isinstance(v,np.ndarray) else v for k,v in shot.items()}
    other['values'][73:] *= 20
    changed, _ = predict_shot(model, other, intra=True)
    np.testing.assert_array_equal(original[:73], changed[:73])
    repeated, _ = predict_shot(model, shot, intra=True)
    np.testing.assert_array_equal(original, repeated)
    assert not model.initial_delta().any()
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())


def test_provider_never_sees_unarrived_signal_suffix():
    model, shot = case(60)
    calls=[]
    class Provider:
        @staticmethod
        def window(signals,end,size):
            assert set(signals) == {'values','available'}
            assert len(signals['values']) == end+1
            calls.append(end)
            return simple_data.window(signals,end,size)
    class NoOutcome(dict):
        def __getitem__(self,key):
            assert key not in ('down','is_disrupt','y','valid','horizon')
            return super().__getitem__(key)
    scores, audit = predict_shot(model,NoOutcome(shot),intra=True,data_api=Provider)
    assert len(scores)==60 and max(calls)==59
    assert audit['accepted_updates']==2


def test_missing_targets_make_no_update():
    model,shot=case(30)
    shot['available'][1:26]=False
    scores,audit=predict_shot(model,shot,intra=True)
    assert np.isfinite(scores).all()
    assert audit['accepted_updates']==0
    assert audit['events'][0]['reason']=='no_valid_targets'
