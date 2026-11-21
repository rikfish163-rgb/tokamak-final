import json
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.cross_channel_factorial import read_cap,locked_test_order
from experiments.risk_aligned_online.fixed_test_protocol import validate_test_ids
from experiments.risk_aligned_online.cross_channel_stable import HEAD,predict_shot
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import PARENT_SHA256
from experiments.risk_aligned_online.redesign_20261004.run import run_arm
from experiments.risk_aligned_online.simple_data import window
from experiments.risk_aligned_online.simple_online import _hash
from experiments.risk_aligned_online.tests.test_cross_channel_prefix import signals
from experiments.risk_aligned_online.tests.test_cross_channel_stable import model


def test_future_factorial_runs_use_full_test360_without_label_fields():
    order=locked_test_order()
    assert len(order)==360
    assert all(set(row)=={'shot'} for row in order)
    validate_test_ids('EAST',[row['shot'] for row in order])


def test_target_or_mismatched_calibration_rejected(tmp_path):
    p=tmp_path/'cap.json'
    data=dict(source_outcomes_read=False,east_data_used=False,parent_sha256=PARENT_SHA256,
              head_sha256=_hash(HEAD),encoding_rms_cap=.12)
    p.write_text(json.dumps(data));assert read_cap(p)[0]==.12
    for key,value in [('east_data_used',True),('source_outcomes_read',True),
                      ('parent_sha256','wrong'),('head_sha256','wrong'),('encoding_rms_cap',0.)]:
        p.write_text(json.dumps(dict(data,**{key:value})))
        with pytest.raises(ValueError):read_cap(p)


def test_factorial_causal_reveal_and_independent_shot_reset(tmp_path):
    torch.set_num_threads(2);m=model();seen=[]
    def read(device,sid):return dict(signals(65),shot=sid)
    def reveal(s,device,advanced):
        assert len(list((tmp_path/'stream').glob('*_sealed.npz')))==len(seen)+1
        seen.append(s['shot'])
        return dict(s,horizon=125.,is_disrupt=False,down=None,valid=np.ones(65,bool),y=np.zeros(65))
    def no_inter(*args,**kwargs):raise AssertionError('inter updates forbidden')
    result=run_arm(m,order=[dict(shot=1),dict(shot=2)],arm='intra',threshold=.72,
        output_dir=tmp_path/'stream',stride=32,device='cpu',
        predictor=partial(predict_shot,feedback='joint',cap=.01),updater=no_inter,
        data_api=SimpleNamespace(read_shot=read,attach_outcome=reveal,window=window))
    assert result['accepted_inter_updates']==0 and result['accepted_intra_updates']>0
    files=sorted((tmp_path/'stream').glob('*_sealed.npz'))
    with np.load(files[0]) as a,np.load(files[1]) as b:
        np.testing.assert_array_equal(a['scores'],b['scores'])
