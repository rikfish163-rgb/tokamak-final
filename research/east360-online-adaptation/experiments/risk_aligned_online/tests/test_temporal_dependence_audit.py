import numpy as np
import pytest

from experiments.risk_aligned_online.temporal_dependence_20261023.audit import eligible_peak, transition_groups


def test_peak_is_strictly_timely_and_uses_actual_times():
    t=np.array([.1,.96,.97,.98,1.01]);p=np.array([.2,.4,.9,.95,1.])
    assert eligible_peak(t,p,True,1.)==1
    assert eligible_peak(t,p,False,None)==4
    assert eligible_peak(np.array([.98]),np.array([.2]),True,1.) is None


def test_no_timely_alarms_has_no_median_not_nan():
    from experiments.risk_aligned_online.temporal_dependence_20261023.audit import lead_median
    assert lead_median([]) is None
    assert lead_median([40.,80.])==60.


def test_transition_groups_require_paired_shots():
    a=[dict(shot=1,is_disrupt=True,timely=True,alarm=True),dict(shot=2,is_disrupt=False,timely=False,alarm=True)]
    b=[dict(a[0],timely=False),dict(a[1],alarm=False)]
    assert transition_groups(a,b)==dict(lost_timely=[1],removed_fp=[2],other=[])
    with pytest.raises(ValueError):transition_groups(a,list(reversed(b)))


def test_model_controls_preserve_reference_state_and_prefix_predictions():
    import torch
    from experiments.risk_aligned_online.simple23 import Simple23,Config
    from experiments.risk_aligned_online.paired_feedback_20261022.model import PairedFeedback
    from experiments.risk_aligned_online.temporal_dependence_20261023.controls import predict_controls
    from experiments.risk_aligned_online.warmstart_alignment_20261022.run import predict_many
    torch.set_num_threads(1);torch.manual_seed(5)
    base=Simple23(Config(dim=8,heads=2,fast_site='tokens'),np.zeros(23),np.ones(23))
    model=PairedFeedback(base,np.ones(23));model.set_stage('frozen')
    state={k:v.clone() for k,v in model.state_dict().items()}
    x=np.random.default_rng(3).normal(size=(224,23)).astype(np.float32);x[:,12]=2.
    signal=dict(values=x,available=np.ones_like(x,dtype=bool))
    controls=predict_controls(model,signal,device='cpu')
    reference=predict_many(model,dict(original=signal),device='cpu')['original']
    np.testing.assert_array_equal(controls['original'],reference)
    np.testing.assert_array_equal(controls['density_flat'],reference)
    short={k:v[:192] for k,v in signal.items()}
    for name,p in predict_controls(model,short,device='cpu').items():
        np.testing.assert_array_equal(p,controls[name][:192])
    for k,v in model.state_dict().items():torch.testing.assert_close(v,state[k],rtol=0,atol=0)
