import numpy as np
import pytest

from experiments.transfer_methods_2026 import rebuilt_harness as harness


@pytest.mark.parametrize('outcome,listed,down', [(True,False,1.25),(False,True,None)])
def test_legacy_label_map_cannot_override_hdf5_outcome(monkeypatch,outcome,listed,down):
    values = {name:np.ones(4,dtype=np.float32) for name in harness.RAW23}
    monkeypatch.setattr(harness,'load_raw',lambda *_:(values,1000.,0.,down,outcome))
    shots = harness.Shots('EAST',[70320],label_map={70320:listed},
                          scaler=(np.zeros(harness.K),np.ones(harness.K)))
    assert len(shots.shots)==1
    assert shots.shots[0]['is_disrupt'] is outcome
    assert shots.shots[0]['down_time_s']==down
