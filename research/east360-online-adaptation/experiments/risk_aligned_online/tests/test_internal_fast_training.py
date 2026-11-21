import copy
import numpy as np
import torch

from experiments.risk_aligned_online.internal_fast_window_20261023.learning import (
    training_pools, sample_batch, risk_step, predict)
from experiments.risk_aligned_online.internal_fast_window_20261023.model import WindowAdapter
from experiments.risk_aligned_online.simple23 import Simple23, Config


def model():
    torch.set_num_threads(1);torch.manual_seed(131)
    return WindowAdapter(Simple23(Config(fast_site='tokens'),torch.zeros(23),torch.ones(23)).eval())


def signal(sid,disruptive):
    rng=np.random.default_rng(sid);n=300;times=np.arange(n)*.001
    return dict(shot=sid,is_disrupt=disruptive,down=.299 if disruptive else None,
        times=times,values=rng.normal(size=(n,23)).astype(np.float32),
        available=np.ones((n,23),bool),valid=np.ones(n,bool))


def test_balanced_sampling_obeys_preparation_only_and_timely_window():
    records=[signal(1,False),signal(2,True)]
    pool=training_pools(records);x,m,y,receipt=sample_batch(records,pool,np.random.default_rng(131),'cpu')
    assert x.shape==(32,128,23) and y.sum()==16
    assert {r['shot'] for r in receipt}=={1,2}
    for r in receipt:
        s=records[r['shot']-1];end=r['end']
        assert end>=127
        if r['label']:assert 30<round((s['down']-s['times'][end])*1000,9)<=125
    # A future change cannot modify this already observed training window.
    records[0]['values'][receipt[0]['end']+1:]=999
    from experiments.risk_aligned_online.simple_data import window
    xx,mm=window(records[0],receipt[0]['end'],128)
    np.testing.assert_array_equal(xx,x[0].numpy())


def test_training_changes_only_layer_and_prediction_reads_only_past_windows():
    a=model();b=copy.deepcopy(a);s=signal(5,False)
    values=torch.from_numpy(s['values'][:128]).unsqueeze(0)
    mask=torch.ones_like(values,dtype=torch.bool);original=copy.deepcopy(a.base.state_dict())
    opt=torch.optim.Adam(a.layer.parameters(),lr=.001)
    with torch.no_grad():h,valid=a.tokens(values,mask)
    risk_step(a,opt,h,valid,torch.ones(1),mode='on')
    assert a.layer.output.weight.abs().max()>0
    for k,v in original.items():assert torch.equal(v,a.base.state_dict()[k])
    prefix={k:s[k][:129] for k in ('values','available')}
    full={k:s[k] for k in ('values','available')}
    first,_=predict(a,b,prefix,device='cpu',batch=64)
    second,_=predict(a,b,full,device='cpu',batch=64)
    for arm in first:np.testing.assert_allclose(first[arm],second[arm][:129],atol=1e-6,rtol=1e-6)
