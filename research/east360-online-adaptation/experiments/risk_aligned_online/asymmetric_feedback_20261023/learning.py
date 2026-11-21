"""Ablate the direct auxiliary gradient, not the actual intra-shot update."""
import torch

from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..east_joint_feedback_20261023.learning import episode
from ..increment_forecast_20261023.learning import feedback_loss


def decoder_forecast(model,signal,count,*,device):
    if set(signal)!={'values','available'} or type(count) is not int or not 256<=count<=len(signal['values']) or count%128:
        raise ValueError('exact arrived signal-only prefix required')
    prefix={k:v[:count] for k,v in signal.items()}
    x,m,t,tm,_=forecast_examples(prefix,count,model.cfg,device)
    x=x.to(model.base.center.dtype);t=t.to(model.base.center.dtype)
    frozen={k:p.detach() for k,p in model.named_parameters() if k in model.encoder_names()}
    # Explicit derivative intervention: same forward value and head gradient;
    # only the DIRECT outer forecast gradient to the initial encoder is absent.
    # Inner updates and outer risk-through-inner gradients remain unchanged.
    pred=model.predict_fast(x,m,frozen)['forecast']
    return feedback_loss(model,pred,x,m,t,tm)


def routed_episode(model,signal,count,ends,disruptive,*,boundary,track,device):
    result=episode(model,signal,count,ends,disruptive,boundary=boundary,track=track,device=device)
    result['forecast']=decoder_forecast(model,signal,count,device=device)
    return result


def gradient_relation(a,b):
    if a.ndim!=1 or a.shape!=b.shape or not len(a):raise ValueError('matched nonempty gradient vectors required')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise FloatingPointError('nonfinite gradient')
    a,b=a.double(),b.double();an,bn=float(a.norm()),float(b.norm());dot=float(a@b)
    # A genuinely zero derivative has no angle; do not report spurious alignment.
    return dict(dot=dot,norm_a=an,norm_b=bn,cosine=dot/(an*bn) if an and bn else None)
