from types import SimpleNamespace

import numpy as np
import pytest

from experiments.risk_aligned_online.cross_channel_shot_screen import evaluate_shot


def test_full_signal_prediction_sealed_before_source_labels(tmp_path):
    calls=[]
    signal=dict(shot=42,device='J-TEXT',times=np.arange(4)*.001,
                values=np.zeros((4,23),np.float32),available=np.ones((4,23),bool))
    def predict(model, shot, **kwargs):
        calls.append('predict')
        assert not {'y','valid','down','is_disrupt'} & set(shot)
        assert len(shot['times'])==4
        return np.array([.1,.2,.3,.9],np.float32),dict(fast_discarded=True)
    def outcome(shot,device,advanced):
        calls.append('outcome')
        with np.load(tmp_path/'42.npz',allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['times'],shot['times'])
            assert len(saved['scores'])==4
        return dict(shot,is_disrupt=False,down=None,valid=np.array([1,1,1,0],bool))
    api=SimpleNamespace(read_shot=lambda device,sid:signal,attach_outcome=outcome)
    row,diag=evaluate_shot(None,42,tmp_path,{},predictor=predict,data_api=api)
    assert calls==['predict','outcome']
    assert row['down'] is None
    np.testing.assert_array_equal(row['scores'],np.array([.1,.2,.3],np.float32))
    assert diag['full_prediction_count']==4 and diag['scored_count']==3


@pytest.mark.parametrize('scores',[[.1],[.1,.2,float('nan'),.4],[.1,.2,1.1,.4]])
def test_invalid_prediction_never_opens_labels(tmp_path,scores):
    signal=dict(shot=42,device='J-TEXT',times=np.arange(4)*.001)
    def forbidden(*args):raise AssertionError('outcome opened before valid seal')
    api=SimpleNamespace(read_shot=lambda device,sid:signal,attach_outcome=forbidden)
    with pytest.raises(ValueError,match='probabilities'):
        evaluate_shot(None,42,tmp_path,{},data_api=api,
                      predictor=lambda *a,**k:(np.asarray(scores),{}))
