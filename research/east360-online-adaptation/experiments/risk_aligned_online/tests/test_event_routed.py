import numpy as np
import torch


def test_ensemble_distance_is_average_of_three_seed_distances():
    from experiments.risk_aligned_online.event_routed_20261024.learning import embed
    from experiments.risk_aligned_online.event_pairing_20261024.learning import Encoder
    torch.manual_seed(1);models=[Encoder().eval() for _ in range(3)]
    x=np.random.default_rng(1).normal(size=(2,391)).astype('float32')
    joined=embed(x,models,np.zeros(391),np.ones(391),'risk','cpu')
    with torch.no_grad():
        expected=np.mean([float((m(torch.tensor(x))[0]-m(torch.tensor(x))[1]).square().sum()) for m in models])
    np.testing.assert_allclose(((joined[0]-joined[1])**2).sum(),expected,rtol=1e-6)
    assert joined.shape==(2,48)


def test_event_routed_neural_updates_reset_and_are_causal():
    from experiments.risk_aligned_online.tests.test_task_geometry import small_model
    from experiments.risk_aligned_online.trajectory_memory_20261024.learning import make_bank
    from experiments.risk_aligned_online.event_pairing_20261024.learning import Encoder
    from experiments.risk_aligned_online.event_routed_20261024.learning import prepare_bank,predict,ARMS
    model=small_model();rng=np.random.default_rng(18);history=[]
    before={k:v.clone() for k,v in model.state_dict().items()}
    for sid,disruptive in enumerate([False,False,True,True]):
        x=rng.normal(size=(300,23)).astype('float32');times=np.arange(300)/1000
        history.append(dict(shot=sid,values=x,available=np.ones_like(x,bool),times=times,
            valid=np.ones(300,bool),is_disrupt=disruptive,down=.3 if disruptive else None))
    bank=make_bank(model,history,device='cpu');torch.manual_seed(1)
    heads={f'{mode}_{seed}':Encoder().state_dict() for mode in ('risk','mean','shuffle','family') for seed in (131,132,133)}
    prepare_bank(bank,heads,dict(mean=np.zeros(391),scale=np.ones(391)),device='cpu')
    x=rng.normal(size=(300,23)).astype('float32');signal=dict(values=x,available=np.ones_like(x,bool))
    a,ra=predict(model,signal,bank,device='cpu');b,rb=predict(model,signal,bank,device='cpu')
    assert ra==rb
    for arm in a:np.testing.assert_array_equal(a[arm],b[arm])
    x[256:]=1e4;c,rc=predict(model,signal,bank,device='cpu')
    assert ra==rc
    for arm in a:
        np.testing.assert_array_equal(a[arm][:256],c[arm][:256])
        np.testing.assert_array_equal(a[arm][:128],a['off'][:128])
    assert any(not np.array_equal(a[arm],a['off']) for arm in ARMS)
    for r in ra:
        for arm in ARMS:
            assert abs(r[arm]['step_norm']-r['risk']['step_norm'])<2e-6
            assert len(set(bank['shots'][r[arm]['selected']]))==4
    for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)


def test_inter_qualification_requires_full_stream_and_genuine_net_improvement():
    from experiments.risk_aligned_online.event_routed_20261024.run import eligible
    from experiments.risk_aligned_online.event_routed_20261024.learning import ARMS
    rows={a:dict(tp=132,fp=37,n_safe=180,n_disrupt=180) for a in ('off',)+ARMS}
    assert not eligible(rows)
    rows['risk'].update(tp=133,fp=35);assert eligible(rows)
    rows['reverse'].update(tp=134,fp=36);assert not eligible(rows)
    rows['risk'].update(n_safe=0);assert not eligible(rows)
