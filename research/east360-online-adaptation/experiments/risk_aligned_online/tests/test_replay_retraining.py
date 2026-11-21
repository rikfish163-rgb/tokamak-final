from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.replay_retraining import ReplayRetrainer, actionable_weights
from experiments.risk_aligned_online.tests.test_ended_adaptation import model, example
from experiments.risk_aligned_online import simple_online
from experiments.risk_aligned_online.tests.test_simple_online import FakeData, shot, Toy


def test_source_restart_is_invariant_to_previous_trained_weights_and_sampling_is_paired():
    a, b = model(), model()
    left = ReplayRetrainer(a, mode='source_restart', steps=2, batch_size=8)
    right = ReplayRetrainer(b, mode='continue', steps=2, batch_size=8)
    first = example()
    x = left(a, first)
    y = right(b, first)
    assert x['sample_sha256'] == y['sample_sha256']
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    clone = deepcopy(a)
    restarted = ReplayRetrainer(model(), mode='source_restart', steps=2, batch_size=8)
    restarted.history = deepcopy(left.history)
    restarted.seen = set(left.seen)
    restarted.round = left.round
    with torch.no_grad():
        a.risk_head.bias.add_(10.)
    second = example()
    second['shot'] = 8
    r1 = left(a, second)
    r2 = restarted(clone, second)
    r3 = right(b, second)
    assert r1['sample_sha256'] == r2['sample_sha256'] == r3['sample_sha256']
    for pa, pb in zip(a.parameters(), clone.parameters()):
        assert torch.equal(pa, pb)
    assert any(not torch.equal(pa, pb) for pa,pb in zip(a.parameters(), b.parameters()))


def test_single_class_start_history_scope_and_duplicate_rejection():
    net = model()
    before = deepcopy(net.state_dict())
    learner = ReplayRetrainer(net, mode='continue', steps=1, capacity=2, batch_size=8)
    for sid in (1, 2, 3):
        s = example()
        s.update(shot=sid, y=np.zeros(60, np.float32), is_disrupt=False)
        log = learner(net, s)
    assert log['history_ids'] == [2, 3]
    assert log['positive_windows'] == 0
    assert log['negative_windows'] == 8
    for name, value in net.state_dict().items():
        if not name.startswith(('risk_head.', 'encoder.layers.1.')) and name != 'fast_origin':
            assert torch.equal(value, before[name])
    with pytest.raises(ValueError, match='duplicate'):
        learner(net, s)


def test_custom_update_happens_after_seal_and_reveal(tmp_path):
    data = FakeData({1: shot(1)})
    data.output = tmp_path / 'inter'
    def update(net, ended, **kwargs):
        assert data.events[-1] == ('reveal', 1)
        assert (data.output / '0001_1_seal.json').exists()
        assert 'y' in ended
        return dict(accepted=True, reason='tested')
    result = simple_online.run_arm(Toy(), order=[dict(shot=1,proxy=1.)], arm='inter',
                                    threshold=.7,output_dir=data.output,data_api=data,
                                    persistent_updater=update)
    assert result['rows'][0]['persistent']['reason'] == 'tested'


def test_actionable_weights_preserve_positive_mass_and_strict_deadline():
    s = dict(valid=np.ones(5,bool), y=np.array([0.,1.,1.,1.,1.]),
             times=np.array([.8,.9,.969,.970,1.]),down=1.,is_disrupt=True)
    labels=s['y'].copy()
    weights=actionable_weights(s)
    np.testing.assert_array_equal(weights,[1.,2.,2.,0.,0.])
    np.testing.assert_array_equal(s['y'],labels)
    assert weights[labels==1].sum()==4


def test_actionable_weights_handle_safe_and_unobservable_early_positive():
    safe=dict(valid=np.array([True,False]),y=np.zeros(2),is_disrupt=False)
    np.testing.assert_array_equal(actionable_weights(safe),[1.,0.])
    late=dict(valid=np.ones(2,bool),y=np.ones(2),is_disrupt=True,
              down=1.,times=np.array([.99,1.]))
    np.testing.assert_array_equal(actionable_weights(late),[0.,0.])


def test_weighted_update_changes_loss_not_sample_manifest():
    a,b=model(),model()
    s=example()
    s.update(times=np.arange(60)/1000.,down=.08)
    base=ReplayRetrainer(a,mode='continue',steps=2)
    candidate=ReplayRetrainer(b,mode='continue',steps=2,objective='actionable')
    x,y=base(a,s),candidate(b,s)
    assert x['sample_sha256']==y['sample_sha256']
    assert y['excluded_late_windows']>0
    assert x['positive_windows']==y['positive_windows']
    assert x['losses']!=y['losses']
    assert any(not torch.equal(pa,pb) for pa,pb in zip(a.parameters(),b.parameters()))


def test_full_scope_updates_embedding_but_never_forecast_or_scaler():
    a,b=model(),model()
    before=deepcopy(b.state_dict())
    x=ReplayRetrainer(a,mode='continue',steps=2)(a,example())
    y=ReplayRetrainer(b,mode='continue',steps=2,scope='full')(b,example())
    assert x['sample_sha256']==y['sample_sha256']
    assert not torch.equal(b.embedding.weight,before['embedding.weight'])
    assert torch.equal(b.forecast_head.weight,before['forecast_head.weight'])
    assert torch.equal(b.center,before['center'])
    assert torch.equal(b.scale,before['scale'])


def test_safe_negative_objective_keeps_manifest_but_excludes_disruptive_early_windows():
    a,b=model(),model()
    x=ReplayRetrainer(a,mode='continue',steps=2,batch_size=16)(a,example())
    y=ReplayRetrainer(b,mode='continue',steps=2,batch_size=16,
                      objective='safe_negative')(b,example())
    assert x['sample_sha256']==y['sample_sha256']
    assert y['excluded_early_windows']==y['negative_windows']>0
    assert y['effective_windows']==y['positive_windows']
    assert y['optimizer_steps']>0
    assert any(not torch.equal(pa,pb) for pa,pb in zip(a.parameters(),b.parameters()))


def test_safe_negative_safe_shot_matches_bce_exactly():
    a,b=model(),model()
    s=example()
    s.update(is_disrupt=False,y=np.zeros(60,np.float32))
    x=ReplayRetrainer(a,mode='continue',steps=2)(a,s)
    y=ReplayRetrainer(b,mode='continue',steps=2,objective='safe_negative')(b,s)
    assert x['sample_sha256']==y['sample_sha256']
    assert y['excluded_early_windows']==0
    assert y['effective_windows']==128
    for pa,pb in zip(a.parameters(),b.parameters()):
        assert torch.equal(pa,pb)


def test_no_observed_precursor_has_no_invented_negative_update():
    net=model()
    before=deepcopy(net.state_dict())
    s=example()
    s['y']=np.zeros(60,np.float32)
    learner=ReplayRetrainer(net,mode='continue',steps=2,objective='safe_negative')
    out=learner(net,s)
    assert out['optimizer_steps']==0
    assert out['effective_windows']==0
    assert out['excluded_early_windows']==128
    assert out['history_ids']==[s['shot']]
    for name,value in net.state_dict().items():
        assert torch.equal(value,before[name])
