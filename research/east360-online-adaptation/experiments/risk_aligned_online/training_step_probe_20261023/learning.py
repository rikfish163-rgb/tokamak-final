"""One-step diagnostic interventions, retaining real Adam history."""
import copy
import torch


def subtract(full,part):
    if not full or len(full)!=len(part) or any(a.shape!=b.shape for a,b in zip(full,part)):
        raise ValueError('matching nonempty gradient lists required')
    if any(not torch.isfinite(t).all() for t in [*full,*part]):
        raise FloatingPointError('finite gradients required')
    return [a-b for a,b in zip(full,part)]


def probe_steps(model,opt,params,gradients,reader):
    if not isinstance(opt,torch.optim.Adam) or not gradients:
        raise ValueError('Adam and explicit diagnostic gradients required')
    if set(map(id,params))!={id(p) for group in opt.param_groups for p in group['params']}:
        raise ValueError('probe must cover exactly the optimized parameters')
    for values in gradients.values():subtract(params,values)
    state=copy.deepcopy(model.state_dict());optimizer=copy.deepcopy(opt.state_dict())
    original=[p.grad.detach().clone() if p.grad is not None else None for p in params]
    def restore():
        model.load_state_dict(state);opt.load_state_dict(copy.deepcopy(optimizer))
        for p,g in zip(params,original):p.grad=None if g is None else g.clone()
    result={}
    try:
        for name,values in gradients.items():
            restore()
            for p,g in zip(params,values):p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
            with torch.no_grad():result[name]=reader().detach().clone()
    finally:
        restore()
    return result
