"""Supplementary review checks; added after the experiment code was sealed."""
import numpy as np
import torch

from experiments.risk_aligned_online import simple_data
from experiments.risk_aligned_online.simple23 import Simple23, Config
from experiments.risk_aligned_online.paired_feedback_20261022.model import PairedFeedback, initial_encoder
from experiments.risk_aligned_online.lag_bias_feedback_20261022 import learning
from experiments.risk_aligned_online.deepcoral_prefix_20261022.learning import forecast_examples
from experiments.risk_aligned_online.warmstart_alignment_20261022.alignment import transform


def model_signal():
    torch.set_num_threads(2);torch.manual_seed(131)
    base=Simple23(Config(dim=16,heads=2,fast_site='tokens',fast_rule='normalized'),
                  np.linspace(-1,1,23),np.linspace(.2,2.,23))
    model=PairedFeedback(base,np.ones(23));model.set_stage('frozen')
    x=np.random.default_rng(131).normal(size=(550,23)).astype(np.float32)
    return model,dict(values=x,available=np.ones_like(x,bool))


def test_source_coordinates_mapped_once_and_same_bias_used_after_step(monkeypatch):
    model,s=model_signal()
    s['available'][210:215,4]=False
    gain=np.ones(23);gain[12]=.00025946043454283725
    signal=transform(s,dict(gain=gain,offset=np.zeros(23)))
    # The mapping contract correctly rejects nonfinite available input. Inject
    # these afterward to separately exercise the forecast-loss finite mask.
    signal['values'][156,0]=np.nan;signal['values'][230,1]=np.inf
    # The update receives already-mapped signal, and must NOT apply gain again.
    calls=[];original=learning.feedback_loss
    def capture(prediction,truth,mask,bias,valid,**kwargs):
        calls.append((truth.detach().clone(),mask.clone(),bias,bias.clone()))
        return original(prediction,truth,mask,bias,valid,**kwargs)
    monkeypatch.setattr(learning,'feedback_loss',capture)
    learning.adapt_arrived(model,signal,256,initial_encoder(model),device='cpu',correct=True)
    target=torch.as_tensor(signal['values'][156:256].reshape(4,25,23))
    valid=torch.as_tensor(signal['available'][156:256].reshape(4,25,23)) & torch.isfinite(target)
    expected=torch.asinh((torch.where(valid,target,model.base.center)-model.base.center)/model.base.scale)
    for truth,mask,bias,copy in calls:
        torch.testing.assert_close(truth,expected,rtol=0,atol=0)
        torch.testing.assert_close(mask,valid,rtol=0,atol=0)
        assert bias is calls[0][2] and not bias.requires_grad
        torch.testing.assert_close(copy,calls[0][3],rtol=0,atol=0)


def test_forecast_reconstruction_after_nonzero_fast_update():
    model,signal=model_signal()
    fast=initial_encoder(model)
    fast,r=learning.adapt_arrived(model,signal,256,fast,device='cpu',correct=True)
    assert r['parameter_delta_norm']>0
    issued=[]
    for origin in (283,308,333,358):
        # No later point exists in this prefix at forecast issuance.
        prefix={k:v[:origin+1] for k,v in signal.items()}
        x,m=simple_data.window(prefix,origin,128)
        issued.append(model.predict_fast(torch.tensor(x)[None],torch.tensor(m)[None],fast)['forecast'].detach())
    x,m,_,_,ends=forecast_examples(signal,384,model.cfg,'cpu')
    reconstructed=model.predict_fast(x,m,fast)['forecast'].detach()[np.argsort(ends).tolist()]
    # Different GEMM batch sizes have ordinary floating point roundoff; this
    # compares identical origin information/state, not probability thresholds.
    torch.testing.assert_close(reconstructed,torch.cat(issued),rtol=1e-6,atol=1e-6)
