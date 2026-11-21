import numpy as np
import torch

from experiments.risk_aligned_online.prefix_density_control import predict_prefix
from experiments.risk_aligned_online.redesign_20261004.intra import predict_shot
from .test_cross_density import case, threads, D


def shot(x,a,size=61):
    v=x[0,:size].numpy().copy(); v[:,D]=100.
    return dict(device='EAST',shot=1,values=v,available=a[0,:size].numpy().copy(),times=np.arange(size)/1000.)


def test_first26_predictions_are_sealed_before_ten_step_prefix_fit():
    _,model,x,a=case();s=shot(x,a)
    static,_=predict_shot(model,s,False)
    out,audit=predict_prefix(model,s,True)
    np.testing.assert_array_equal(out[:26],static[:26])
    assert audit['accepted_updates']>0 and audit['attempted_updates']<=10
    assert all(r['origin_index']==0 and r['maturity_index']==25 for r in audit['attempts'])
    assert all(r['effective_index']==26 for r in audit['attempts'] if r['accepted'])


def test_later_signals_cannot_change_prefix_bias_or_retroactive_predictions():
    _,model,x,a=case();s=shot(x,a);other=shot(x,a);other['values'][26:,D]=1000.
    left,la=predict_prefix(model,s,True);right,ra=predict_prefix(model,other,True)
    np.testing.assert_array_equal(left[:26],right[:26])
    assert la['final_fast']==ra['final_fast'] and la['accepted_updates']==ra['accepted_updates']
    assert la['updated_signal_origins']==1 and la['fast_frozen_after_prefix']


def test_short_shot_never_fits_unarrived_target_and_disposes_state():
    _,model,x,a=case();s=shot(x,a,25)
    state={k:v.clone() for k,v in model.state_dict().items()}
    out,audit=predict_prefix(model,s,True);static,_=predict_prefix(model,s,False)
    np.testing.assert_array_equal(out,static)
    assert audit['attempted_updates']==0 and audit['fast_discarded']
    for k,v in state.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)


def test_same_shot_restarts_zero_and_original_model_state_is_preserved():
    _,model,x,a=case();s=shot(x,a)
    state={k:v.clone() for k,v in model.state_dict().items()}
    left,la=predict_prefix(model,s,True);right,ra=predict_prefix(model,s,True)
    np.testing.assert_array_equal(left,right)
    assert la['final_fast']==ra['final_fast'] and abs(la['final_fast'][0])<=12.
    for k,v in state.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
