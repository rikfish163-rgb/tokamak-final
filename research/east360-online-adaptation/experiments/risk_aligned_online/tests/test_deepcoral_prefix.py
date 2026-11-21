import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.deepcoral_prefix_20261022.learning import (
    coral_loss, natural_batch, validate_prefix, forecast_examples, adapt_arrived, eligible_records,
)
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.paired_feedback_20261022.model import PairedFeedback, initial_encoder
from experiments.risk_aligned_online.deepcoral_prefix_20261022.run import predict_pairs


def record(sid=1, n=200):
    return dict(shot=sid, device='EAST', values=np.ones((n,23),np.float32),
                available=np.ones((n,23),bool))


def test_coral_equation_and_gradients():
    s=torch.tensor([[0.,0.],[2.,0.],[0.,2.]],dtype=torch.double,requires_grad=True)
    assert coral_loss(s,s/2).item()==pytest.approx(.15625)
    assert coral_loss(s,s+7).item()==pytest.approx(0.,abs=1e-20)
    t=torch.randn(5,2,dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(coral_loss,(s,t))
    assert coral_loss(s,t).item()==pytest.approx(coral_loss(t,s).item())
    for bad in (s[:1],s[:,0]):
        with pytest.raises(ValueError): coral_loss(bad,t)


def test_sampler_cannot_receive_target_labels_or_future_ids():
    rows=[record(1),record(2)]
    validate_prefix(rows, [1,2,3], 2)
    with pytest.raises(ValueError): validate_prefix([record(1),record(3)],[1,2,3],2)
    with pytest.raises(ValueError): validate_prefix([dict(rows[0],y=1),rows[1]],[1,2,3],2)
    cfg=Config()
    a=natural_batch(rows,np.random.default_rng(42),32,cfg,'cpu')
    b=natural_batch(rows,np.random.default_rng(42),32,cfg,'cpu')
    assert a[-1]==b[-1]
    assert set(r['shot'] for r in a[-1])=={1,2}
    with pytest.raises(ValueError): natural_batch([dict(rows[0],y=0)],np.random.default_rng(1),4,cfg,'cpu')


def test_full_window_eligibility_is_signal_only_and_reports_short_shots():
    rows=[record(1,32),record(2,128),record(3,153),record(4,200)]
    rows[-1]['available'][:]=False
    kept,excluded=eligible_records(rows,Config(),future=False)
    assert [s['shot'] for s in kept]==[2,3]
    assert [s['shot'] for s in excluded]==[1,4]
    kept,excluded=eligible_records(rows,Config(),future=True)
    assert [s['shot'] for s in kept]==[3]
    with pytest.raises(ValueError): eligible_records([dict(record(),valid=True)],Config())


def test_arrived_forecast_never_reads_later_values_and_does_not_mutate_parent():
    torch.manual_seed(8)
    cfg=Config(fast_site='tokens',fast_rule='normalized')
    model=PairedFeedback(Simple23(cfg,np.zeros(23),np.ones(23)),np.ones(23))
    model.set_stage('frozen')
    r=record(n=400); signal={k:r[k] for k in ('values','available')}
    r['values'][:]=np.random.default_rng(2).normal(size=(400,23))
    a=forecast_examples(signal,256,cfg,'cpu')
    r['values'][256:]=999
    b=forecast_examples(signal,256,cfg,'cpu')
    for x,y in zip(a[:-1],b[:-1]): torch.testing.assert_close(x,y)
    assert max(a[-1])==230 and min(a[-1])>=127
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    fast=initial_encoder(model)
    updated,receipt=adapt_arrived(model,signal,256,fast,device='cpu')
    assert receipt['first_affected_prediction']==256
    assert receipt['last_observation']==255
    assert receipt['parameter_delta_norm']>0
    assert set(updated)==set(model.encoder_names())
    for k,v in original.items(): torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
    with pytest.raises(ValueError): forecast_examples(dict(signal,y=0),256,cfg,'cpu')


def test_stream_future_invariance_reset_and_zero_coral_equivalence():
    import copy
    torch.set_num_threads(2); torch.manual_seed(19)
    cfg=Config(window=32,dim=16,heads=2,fast_site='tokens')
    model=PairedFeedback(Simple23(cfg,np.zeros(23),np.ones(23)),np.ones(23))
    model.set_stage('frozen')
    signal={k:v for k,v in record(n=400).items() if k in ('values','available')}
    signal['values'][:]=np.random.default_rng(3).normal(size=(400,23))
    a,ra=predict_pairs({'model':model},signal,device='cpu')
    signal['values'][256:]=77.
    b,rb=predict_pairs({'model':model},signal,device='cpu')
    for k in a: np.testing.assert_array_equal(a[k][:256],b[k][:256])
    c,rc=predict_pairs({'model':model},signal,device='cpu')
    for k in c: np.testing.assert_array_equal(c[k],b[k])
    assert ra['model'][0]['first_affected_prediction']==128
    np.testing.assert_array_equal(a['model_static'][:128],a['model_intra'][:128])
    # BCE alone and BCE+zero*CORAL produce the same optimizer update, not just
    # identical scalar losses. No optimizer state may leak between arms.
    models=[copy.deepcopy(model),copy.deepcopy(model)]
    for mod in models:
        for k,p in mod.named_parameters(): p.requires_grad_(k in mod.encoder_names())
    x=torch.randn(4,32,23); mask=torch.ones_like(x,dtype=torch.bool); y=torch.tensor([0.,1.,0.,1.])
    for j,mod in enumerate(models):
        opt=torch.optim.Adam([p for p in mod.parameters() if p.requires_grad],lr=1e-4)
        h=mod(x,mask,mode='risk_state')['risk_state']
        loss=torch.nn.functional.binary_cross_entropy_with_logits(mod.base.risk_head(h).squeeze(-1),y)
        if j: loss=loss+0*coral_loss(h,mod(x*2,mask,mode='risk_state')['risk_state'])
        loss.backward(); opt.step()
    for k,v in models[0].state_dict().items(): torch.testing.assert_close(v,models[1].state_dict()[k],rtol=0,atol=0)
