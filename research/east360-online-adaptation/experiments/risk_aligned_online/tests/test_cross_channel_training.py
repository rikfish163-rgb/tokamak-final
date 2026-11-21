import pytest
import torch

from experiments.risk_aligned_online.cross_channel_train import train_step, source_screen
from experiments.risk_aligned_online.tests.test_cross_channel_prefix import make_model


@pytest.mark.parametrize('site',['embedding','ffn'])
@pytest.mark.parametrize('mode',['signal','risk'])
def test_source_training_only_changes_auxiliary_decoder(site,mode):
    torch.set_num_threads(2)
    m=make_model(site);m.decoder.requires_grad_(True)
    before={k:v.clone() for k,v in m.base.state_dict().items()};head=m.decoder.weight.clone()
    e=dict(x=torch.randn(2,128,23),a=torch.ones(2,128,23,dtype=torch.bool),
        truth=torch.randn(2,25,23),mask=torch.ones(2,25,23,dtype=torch.bool),
        qx=torch.randn(2,128,23),qa=torch.ones(2,128,23,dtype=torch.bool),y=torch.tensor([0.,1.]))
    opt=torch.optim.AdamW(m.decoder.parameters(),lr=1e-4,weight_decay=0.)
    r=train_step(m,opt,e,mode=mode,phase=2,length=8)
    assert r['gradient_norm']>0 and not torch.equal(head,m.decoder.weight)
    assert not m.memory.any()
    for k,v in m.base.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    assert all(p.grad is None for p in m.base.parameters())


def test_source_screen_requires_both_classes_and_not_only_reconstruction_gain():
    rows=[dict(label=0,before=.2,after=.1),dict(label=1,before=.2,after=.19)]
    assert source_screen(rows)['passed']
    rows[1]['after']=.21
    assert not source_screen(rows)['passed']
    with pytest.raises(ValueError):source_screen(rows[:1])
