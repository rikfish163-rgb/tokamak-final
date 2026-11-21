import numpy as np
import pytest

from experiments.risk_aligned_online.mapped_head_factor_20261023.mapping import fit_mapping,map_record


def record(sid,device,scale):
    return dict(shot=sid,device=device,values=np.full((5,23),scale,np.float32),available=np.ones((5,23),bool))


def test_only_authorized_signal_records_fit_a_positive_density_map():
    source=[record(1,'J-TEXT',2.)];target=[record(i,'EAST',10.) for i in range(20,40)]
    mapping=fit_mapping(source,target,source_ids=[1],target_ids=list(range(20,40)))
    assert mapping['gain'][12]==.2 and mapping['physical_calibration_certified'] is False
    assert all(g==1. for i,g in enumerate(mapping['gain']) if i!=12)
    assert mapping['offset']==[0.]*23
    with pytest.raises(ValueError):fit_mapping(source,target,source_ids=[1],target_ids=list(range(21,41)))
    with pytest.raises(ValueError):fit_mapping(source,[dict(s,is_disrupt=False) for s in target],source_ids=[1],target_ids=list(range(20,40)))
    with pytest.raises(ValueError):fit_mapping(source,target[:-1],source_ids=[1],target_ids=list(range(20,40)))


def test_mapping_preserves_labels_timing_masks_and_caller_arrays():
    mapping=dict(gain=[.2 if c==12 else 1. for c in range(23)],offset=[0.]*23)
    s=record(20,'EAST',10.);s.update(y=np.arange(5),valid=np.ones(5,bool),times=np.arange(5)/1000,down=.01)
    s['values'][:,12]=[-2,0,2,4,8];s['available'][3,12]=False
    original=s['values'].copy();out=map_record(s,mapping)
    np.testing.assert_array_equal(s['values'],original)
    np.testing.assert_allclose(out['values'][:,12],[-.4,0,.4,4,1.6])
    np.testing.assert_array_equal(out['values'][:,:12],original[:,:12])
    np.testing.assert_array_equal(out['values'][:,13:],original[:,13:])
    np.testing.assert_array_equal(out['available'],s['available'])
    for k in ('y','valid','times'):np.testing.assert_array_equal(out[k],s[k])
    assert out['down']==s['down'] and out['shot']==20
    assert not np.shares_memory(s['values'],out['values'])
