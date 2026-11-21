import numpy as np
import pytest
import torch
from experiments.risk_aligned_online.simple23 import Config,Simple23
from experiments.risk_aligned_online.local_transition_20261023.model import LocalTransition,fit_local
from experiments.risk_aligned_online.context_separation_20261023.model import mean_shift,forward


def test_mean_shift_equals_constant_key_ridge_and_missing_zero():
    torch.manual_seed(31)
    q=torch.randn(2,1,9,dtype=torch.double);q[:,:,-1]=1
    k=q.expand(-1,14,-1);target=torch.randn(2,14,23,dtype=torch.double)
    mask=torch.rand(2,14,23)>.3;mask[:,:,2]=False
    prior=torch.randn(9,23,dtype=torch.double)
    actual=mean_shift(k,target,mask,prior)
    expected=torch.einsum('br,brc->bc',q[:,0],fit_local(k,target,mask,prior)-prior[None])
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    assert not actual[:,2].any()
    torch.testing.assert_close(actual,mean_shift(k,target.flip(1),mask.flip(1),prior),atol=1e-12,rtol=1e-12)


def test_recent_forward_equals_original_and_reference_controls_update():
    torch.set_num_threads(2);torch.manual_seed(19)
    model=LocalTransition(Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23))).eval()
    with torch.no_grad():model.readout.weight.normal_(0,.05)
    x=torch.randn(3,128,23);mask=torch.ones_like(x,dtype=torch.bool)
    out=model(x,mask,mode='fit');new=forward(model,x,mask,x,mask,kind='ridge')
    torch.testing.assert_close(out['logit'],new['logit'],rtol=0,atol=0)
    torch.testing.assert_close(out['innovation'],new['innovation'],rtol=0,atol=0)
    # Only the specified reference can change the fitted correction.
    now=x.clone();now[:,-8:]+=4
    altered=forward(model,now,mask,x,mask,kind='ridge')
    torch.testing.assert_close(new['fitted'],altered['fitted'],rtol=0,atol=0)
    before={k:v.clone() for k,v in model.state_dict().items()}
    forward(model,x,mask,x+3,mask,kind='mean')
    for k,v in before.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)


def test_stream_future_invariance_reset_and_common_warmup():
    from experiments.risk_aligned_online.context_separation_20261023.run import predict
    from experiments.risk_aligned_online.simple_data import window
    torch.set_num_threads(2);torch.manual_seed(23)
    model=LocalTransition(Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23))).eval()
    with torch.no_grad():model.readout.weight.normal_(0,.05)
    model.requires_grad_(False)
    rng=np.random.default_rng(23)
    signal={'values':rng.normal(size=(320,23)).astype(np.float32),'available':np.ones((320,23),bool)}
    result,_=predict(model,signal,device='cpu',replay_recent=True)
    changed={k:v.copy() for k,v in signal.items()};changed['values'][288:]+=3
    different,_=predict(model,changed,device='cpu',replay_recent=True)
    repeated,_=predict(model,signal,device='cpu',replay_recent=True)
    original=[];prior=[]
    for lo in range(0,320,32):
        x,m=[torch.tensor(np.stack(v)) for v in zip(*[window(signal,t,128) for t in range(lo,lo+32)])]
        original.append(model(x,m,mode='fit')['logit'].sigmoid().detach().numpy())
        prior.append(model(x,m,mode='prior')['logit'].sigmoid().detach().numpy())
    expected=np.concatenate(original);expected[:256]=np.concatenate(prior)[:256]
    np.testing.assert_array_equal(result['recent_ridge'],expected)
    for arm in result:
        np.testing.assert_array_equal(result[arm][:288],different[arm][:288])
        np.testing.assert_array_equal(result[arm],repeated[arm])
        np.testing.assert_array_equal(result[arm][:256],np.concatenate(prior)[:256])
    with pytest.raises(ValueError):predict(model,{**signal,'y':np.zeros(320)},device='cpu')
