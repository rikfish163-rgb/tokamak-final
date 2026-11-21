import numpy as np
import pytest

from experiments.risk_aligned_online.patch_order_20261023.control import reverse_past_patches, predict_reversed


def windows():
    x=np.random.default_rng(9).normal(size=(4,128,23)).astype(np.float32)
    m=np.ones_like(x,dtype=bool);m[:,:,2]=False
    return x,m


def test_reversal_keeps_latest_patch_masks_and_joint_rows_exact():
    x,m=windows();before=x.copy();mask=m.copy()
    out,eligible=reverse_past_patches(x,m,np.array([128,256,1000,2000]))
    assert eligible.all()
    np.testing.assert_array_equal(out[:,-8:],x[:,-8:])
    expected=np.concatenate([np.arange(8*k,8*k+8) for k in range(14,-1,-1)]+[np.arange(120,128)])
    np.testing.assert_array_equal(out,x[:,expected])
    np.testing.assert_array_equal(np.sort(out,axis=1),np.sort(x,axis=1))
    np.testing.assert_array_equal(m,mask);np.testing.assert_array_equal(x,before)
    restored,_=reverse_past_patches(out,m,np.array([128,256,1000,2000]))
    np.testing.assert_array_equal(restored,x)


def test_padding_variable_masks_and_empty_windows_are_not_intervened():
    x,m=windows();m[1,4,1]=False;m[2]=False
    out,eligible=reverse_past_patches(x,m,np.array([127,128,129,130]))
    assert eligible.tolist()==[False,False,False,True]
    np.testing.assert_array_equal(out[:3],x[:3])


def test_bad_inputs_rejected():
    x,m=windows();n=np.arange(4)+128
    for counts in (n.astype(float),np.array([1,2]),np.array([128,129,-1,130])):
        with pytest.raises(ValueError):reverse_past_patches(x,m,counts)
    with pytest.raises(ValueError):reverse_past_patches(x,m,n,patch=7)
    with pytest.raises(ValueError):reverse_past_patches(x,m.astype(float),n)
    x[0,0,0]=np.nan
    with pytest.raises(ValueError):reverse_past_patches(x,m,n)


def test_real_model_prefix_noop_reference_and_state_are_preserved():
    import torch
    from experiments.risk_aligned_online.simple23 import Simple23,Config
    from experiments.risk_aligned_online.paired_feedback_20261022.model import PairedFeedback
    from experiments.risk_aligned_online.warmstart_alignment_20261022.run import predict_many
    torch.set_num_threads(1);torch.manual_seed(7)
    model=PairedFeedback(Simple23(Config(dim=8,heads=2,fast_site='tokens'),np.zeros(23),np.ones(23)),np.ones(23))
    model.set_stage('frozen');state={k:v.clone() for k,v in model.state_dict().items()}
    x=np.random.default_rng(3).normal(size=(224,23)).astype(np.float32)
    m=np.ones_like(x,dtype=bool);m[:,2]=False;m[180,5]=False
    signal=dict(values=x,available=m)
    result=predict_reversed(model,signal,device='cpu')
    short=predict_reversed(model,{k:v[:192] for k,v in signal.items()},device='cpu')
    for key,value in short.items():np.testing.assert_array_equal(value,result[key][:192])
    original=predict_many(model,dict(original=signal),device='cpu')['original']
    np.testing.assert_array_equal(result['probability'][~result['eligible']],original[~result['eligible']])
    assert result['eligible'].sum()==53 and result['changed'].sum()==53
    for k,v in model.state_dict().items():torch.testing.assert_close(v,state[k],rtol=0,atol=0)
