import torch
import pytest

from .feedback import feedback_loss, segment_step


class Toy:
    center=torch.zeros(23)
    scale=torch.ones(23)

    def risk_from_source_coordinates(self,z,available):
        return z.masked_fill(~available,0).sum((1,2))/10

    def __call__(self,x,a,delta):
        return {'forecast':delta[0].expand(len(x),25,23)}


def inputs(h):
    return (torch.zeros(1,128,23),torch.ones(1,128,23,dtype=torch.bool),
            torch.zeros(1,h,23),torch.ones(1,h,23,dtype=torch.bool))


def test_exact_trajectory_has_zero_both_losses():
    model=Toy(); x,a,y,m=inputs(25)
    forecast=torch.zeros(1,25,23,requires_grad=True)
    loss,row=feedback_loss(model,forecast,x,a,y,m,'late',use_task=True)
    assert loss.item()==0 and row['task_loss']==0 and row['signal_loss']==0


def test_early_loss_cannot_read_prediction_tail():
    x,a,y,m=inputs(5); f=torch.zeros(1,25,23,requires_grad=True)
    first,_=feedback_loss(Toy(),f,x,a,y,m,'early',use_task=True)
    changed=f.detach().clone(); changed[:,5:]=100; changed.requires_grad_()
    second,_=feedback_loss(Toy(),changed,x,a,y,m,'early',use_task=True)
    assert torch.equal(first,second)
    g,=torch.autograd.grad(second,changed)
    assert not g[:,5:].any()


def test_truth_is_detached_and_forecast_receives_task_gradient():
    x,a,y,m=inputs(5); y.fill_(1); y.requires_grad_()
    f=torch.zeros(1,25,23,requires_grad=True)
    full,_=feedback_loss(Toy(),f,x,a,y,m,'early',use_task=True)
    plain,_=feedback_loss(Toy(),f,x,a,y,m,'early',use_task=False)
    g,gy=torch.autograd.grad(full,(f,y),allow_unused=True,retain_graph=True)
    gp,=torch.autograd.grad(plain,f)
    assert gy is None and not torch.equal(g,gp)


def test_late_requires_full_arrived_prefix_but_excludes_early_levels():
    x,a,y,m=inputs(25); m[:,5:]=False
    delta=torch.tensor([.3],requires_grad=True)
    after,row=segment_step(Toy(),delta,x,a,y,m,'late',use_task=True)
    assert after is delta and not row['state_changed']
    with pytest.raises(ValueError):
        feedback_loss(Toy(),torch.zeros(1,25,23),x,a,y[:,:21],m[:,:21],'late',use_task=True)


def test_missing_values_do_not_create_a_task_gradient():
    x,a,y,m=inputs(5); y.fill_(float('nan')); m[:]=False
    f=torch.zeros(1,25,23,requires_grad=True)
    loss,row=feedback_loss(Toy(),f,x,a,y,m,'early',use_task=True)
    g,=torch.autograd.grad(loss,f)
    assert torch.isfinite(loss) and loss.item()==0 and not g.any()


def test_constant_drift_is_not_risk_smoothing():
    # The target is the arrived trajectory, not an instruction to return to
    # the origin's low risk. A rising trajectory therefore points prediction up.
    x,a,y,m=inputs(5); y.fill_(2)
    f=torch.zeros(1,25,23,requires_grad=True)
    loss,_=feedback_loss(Toy(),f,x,a,y,m,'early',use_task=True)
    g,=torch.autograd.grad(loss,f)
    assert (g[:,:5]<0).all()
