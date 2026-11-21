from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.cross_channel_prefix import predict_shot
from experiments.risk_aligned_online.redesign_20261004.run import run_arm
from experiments.risk_aligned_online.simple_data import window
from experiments.risk_aligned_online.tests.test_cross_channel_prefix import make_model,signals


def test_two_east_shots_do_not_inherit_temporary_state_or_read_labels_before_seal(tmp_path):
    torch.set_num_threads(2);m=make_model();observed=[]
    def read(device,sid):
        assert device=='EAST'
        return dict(signals(65),shot=sid)
    def outcome(shot,device,advanced):
        # The source-only inference does not need or receive this outcome.
        assert len(list((tmp_path/'stream').glob('*_sealed.npz')))==len(observed)+1
        observed.append(shot['shot'])
        return dict(shot,horizon=125.,is_disrupt=False,down=None,valid=np.ones(65,bool),y=np.zeros(65))
    def no_persistent(*args,**kwargs):raise AssertionError('pure-intra may not call inter updater')
    api=SimpleNamespace(read_shot=read,attach_outcome=outcome,window=window)
    result=run_arm(m,order=[dict(shot=1),dict(shot=2)],arm='intra',threshold=.72,
        output_dir=tmp_path/'stream',stride=32,device='cpu',predictor=predict_shot,
        updater=no_persistent,data_api=api)
    assert result['observed_count']==2 and result['accepted_inter_updates']==0
    traces=sorted((tmp_path/'stream').glob('*_sealed.npz'))
    with np.load(traces[0],allow_pickle=False) as a,np.load(traces[1],allow_pickle=False) as b:
        np.testing.assert_array_equal(a['scores'],b['scores'])
    assert not m.memory.any()


def test_counterfactual_state_exactly_matches_real_update_without_labels():
    from experiments.risk_aligned_online.cross_channel_counterfactual import replay_to
    torch.set_num_threads(2);m=make_model();s=signals(100)
    _,audit=predict_shot(m,s,intra=True)
    before,after,loss=replay_to(m,s,71)
    assert m.summarize_fast(before)==audit['attempts'][2]['fast_before']
    assert m.summarize_fast(after)==audit['attempts'][2]['fast_after']
    assert loss['loss_after']==audit['attempts'][2]['loss_after']
    with pytest.raises(ValueError,match='signal-only'):
        replay_to(m,dict(s,is_disrupt=False),71)
