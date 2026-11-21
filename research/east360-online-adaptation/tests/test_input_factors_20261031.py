import copy

import numpy as np
import pytest

from experiments.risk_aligned_online.input_factors_20261031.core import (
    NAMES, apply_map, chunks, fit_maps,
)


def parent():
    source = dict(device='J-TEXT', median=[2.]*23, record_counts=[100]*23)
    target = dict(device='EAST', median=[4.]*23, record_counts=[20]*23,
                  shot_ids=list(range(20)))
    return dict(policy='density_gain', calibration_ids=list(range(20)),
                source_statistics=source, target_statistics=target)


def test_maps_only_change_declared_coordinate_and_keep_parent():
    p = parent()
    old = copy.deepcopy(p)
    maps = fit_maps(p, list(range(360)))
    assert p == old
    assert tuple(maps) == NAMES
    for name, index in [('n1_gain',11),('CIII_gain',1),('dx_shift',7),('dy_shift',8)]:
        m = maps[name]
        changed = np.flatnonzero((np.asarray(m['gain'])!=1)|(np.asarray(m['offset'])!=0))
        assert changed.tolist()==[index]
        assert m['gain'][index]==(.5 if 'gain' in name else 1.)
        assert m['offset'][index]==(0. if 'gain' in name else -2.)


@pytest.mark.parametrize('kind',['order','duplicate','missing','zero','nan','negative'])
def test_invalid_fitting_evidence_is_rejected(kind):
    p = parent()
    order = list(range(360))
    if kind=='order':
        order[0],order[1] = order[1],order[0]
    elif kind=='duplicate':
        order[1]=order[0]
    elif kind=='missing':
        p['target_statistics']['record_counts'][1]=0
    else:
        p['target_statistics']['median'][11]={'zero':0.,'nan':float('nan'),'negative':-1.}[kind]
    with pytest.raises(ValueError):
        fit_maps(p,order)


def test_pointwise_mapping_preserves_missing_values_and_prefix():
    maps = fit_maps(parent(),list(range(360)))
    x = np.arange(23*11,dtype=np.float32).reshape(11,23)
    mask = np.ones_like(x,dtype=bool)
    mask[2,11]=False
    s = dict(values=x,available=mask)
    out = apply_map(s,maps['n1_gain'])
    assert np.array_equal(s['values'],x)
    assert np.array_equal(out['available'],mask)
    assert out['values'][2,11]==x[2,11]
    assert np.array_equal(out['values'][:,0],x[:,0])
    pre = apply_map({k:v[:6] for k,v in s.items()},maps['n1_gain'])
    assert np.array_equal(pre['values'],out['values'][:6])
    assert np.array_equal(apply_map(s,maps['cold'])['values'],x)


def test_labels_rejected_not_silently_ignored():
    s=dict(values=np.zeros((2,23),np.float32),available=np.ones((2,23),bool),y=[0,1])
    with pytest.raises(ValueError):
        apply_map(s,fit_maps(parent(),list(range(360)))['cold'])


def test_exact_archived_chunk_geometry():
    out=list(chunks(180))
    assert out==[(0,32),(32,64),(64,96),(96,128),(128,153),(153,178),(178,180)]
    for n in (1,25,128,129,152,153,1000):
        windows=list(chunks(n))
        assert [i for lo,hi in windows for i in range(lo,hi)]==list(range(n))
        assert all(0<hi-lo<=32 for lo,hi in windows)


@pytest.mark.parametrize('n',[0,-1,True,1.5])
def test_invalid_length(n):
    with pytest.raises(ValueError):
        list(chunks(n))


def test_forward_uses_causal_windows_fixed_geometry_and_identical_control():
    import torch
    from types import SimpleNamespace
    from experiments.risk_aligned_online.input_factors_20261031.stream import predict

    class Dummy:
        base = SimpleNamespace(center=torch.zeros(23))

        def __call__(self,x,m,state,slow,mode,forecast):
            assert x.shape==(288,128,23) and m.dtype==torch.bool
            assert state.shape==(288,8192) and not state.any()
            assert mode=='qk' and forecast is False
            return dict(logit=x[:,-1].mean(1))

    maps = fit_maps(parent(),list(range(360)))
    signal = dict(values=np.full((180,23),2.,np.float32),available=np.ones((180,23),bool))
    out = predict(Dummy(),signal,None,maps)
    assert tuple(out)==NAMES
    assert np.array_equal(out['cold'],out['identical'])
    assert (out['n1_gain']<out['cold']).all()
    prefix = predict(Dummy(),{k:v[:170] for k,v in signal.items()},None,maps)
    assert all(np.array_equal(prefix[n],out[n][:170]) for n in NAMES)
