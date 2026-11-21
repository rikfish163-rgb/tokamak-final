"""Short-prefix, cross-diagnostic feedback in original source-network weights.

No risk residual, target support, target statistics, or cross-shot parameter
state. Hidden channels are reconstructed only from other observed diagnostics.
Compare original patch VALUE weights with the original last Transformer FFN.
The availability columns, positional encoding and classifier head are frozen.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call

from . import simple_data
from .fast_update import normalized_step
from .masked_prefix_train import current_window
from .persistent_ffn import PersistentFfnModel
from .redesign_20261004.intra import _batch_windows


def channel_mask(available, *, phase):
    if (available.ndim != 3 or available.shape[1:] != (128,23)
            or available.dtype != torch.bool or type(phase) is not int or phase not in range(4)):
        raise ValueError('boolean Bx128x23 and phase0..3 required')
    return available & (torch.arange(23,device=available.device)%4 == phase)[None,None,:]


def episode_windows(e, length):
    if type(length) is not int or not 1 <= length <= 128:
        raise ValueError('1..128 observed history points required')
    x,a = current_window(e)
    a=a.clone(); a[:,:128-length]=False
    qa=e['qa'].clone(); qa[:,:128-min(128,length+1)]=False
    return x,a,e['qx'],qa


class CrossChannelModel(nn.Module):
    def __init__(self, base, *, site):
        super().__init__()
        if site not in ('embedding','ffn'):
            raise ValueError('explicit original embedding or FFN site required')
        if (base.cfg.window,base.cfg.patch,base.cfg.dim,base.cfg.horizon,
                base.cfg.fast_site,base.cfg.input_view)!=(128,8,64,25,'tokens','source'):
            raise ValueError('qualified source W128/P8/d64/H25 contract required')
        self.base=base.eval().requires_grad_(False)
        self.site=site
        last=base.cfg.layers-1
        self.parameter_names=tuple(f'encoder.layers.{last}.{layer}.{part}'
            for layer in ('linear1','linear2') for part in ('weight','bias'))
        parameters=dict(base.named_parameters())
        self.temporary_dim=(64*8*23 if site=='embedding' else
                            sum(parameters[n].numel() for n in self.parameter_names))
        self.register_buffer('memory',base.center.new_zeros(self.temporary_dim))
        self.decoder=nn.Linear(64,8*23).to(base.center)
        self.requires_grad_(False)

    @property
    def cfg(self): return self.base.cfg
    @property
    def center(self): return self.base.center
    @property
    def scale(self): return self.base.scale

    def initial_fast(self,batch):
        if type(batch) is not int or batch<1:raise ValueError('positive batch required')
        return self.memory.new_zeros(batch,self.temporary_dim)

    def summarize_fast(self,tensor):
        return PersistentFfnModel.summarize_fast(self,tensor)

    def overrides(self,delta):
        if delta.shape!=(self.temporary_dim,) or not torch.isfinite(delta).all():
            raise ValueError('one finite declared temporary vector required')
        if self.site=='embedding':
            # Patch features are [23 values,23 availability] at each of 8 times.
            # Only original value columns change: no new measurement/mask path.
            expanded=F.pad(delta.reshape(64,8,23),(0,23)).reshape(64,8*46)
            return {'embedding.weight':self.base.embedding.weight+expanded}
        parameters=dict(self.base.named_parameters());out={};offset=0
        for name in self.parameter_names:
            original=parameters[name]
            out[name]=original+delta[offset:offset+original.numel()].view_as(original)
            offset+=original.numel()
        return out

    def forward(self,values,available,fast=None):
        if torch.is_inference_mode_enabled():raise RuntimeError('fixed gradient-path attention required')
        fast=self.initial_fast(len(values)) if fast is None else fast
        if (fast.shape!=(len(values),self.temporary_dim) or fast.device!=self.center.device
                or fast.dtype!=self.center.dtype or not torch.isfinite(fast).all()
                or not torch.equal(fast,fast[:1].expand_as(fast))):
            raise ValueError('same finite temporary state for every endpoint in this batch required')
        if self.memory.count_nonzero():raise ValueError('pure-intra model cannot hold target memory')
        enabled=torch.is_grad_enabled()
        with torch.enable_grad():
            fixed=self.base.initial_fast(len(values)).detach().requires_grad_()
            out=functional_call(self.base,self.overrides(fast[0]),(values,available,fixed),strict=False)
        return out if enabled else {k:v.detach() for k,v in out.items()}

    def reconstruction(self,values,available,fast,*,phase):
        available=available & torch.isfinite(values)
        hidden=channel_mask(available,phase=phase)
        visible=available & ~hidden
        # Without any remaining real observation, no evidence supports an update.
        hidden=hidden & visible.flatten(1).any(1)[:,None,None]
        masked=torch.where(visible,values.detach(),self.center)
        target=torch.asinh((torch.where(available,values.detach(),self.center)-self.center)/self.scale)
        features=[]
        handle=self.base.norm.register_forward_hook(lambda module,inputs,output:features.append(output))
        try:self(masked,visible,fast)
        finally:handle.remove()
        if len(features)!=1:raise RuntimeError('one original normalization activation required')
        prediction=self.decoder(features[0][:,:16]).reshape(-1,128,23)
        return prediction,target.detach(),hidden

    def reconstruction_loss(self,values,available,fast,*,phase):
        prediction,target,hidden=self.reconstruction(values,available,fast,phase=phase)
        error=F.smooth_l1_loss(prediction,target,reduction='none').masked_fill(~hidden,0.)
        return (error.sum((1,2))/hidden.sum((1,2)).clamp_min(1)).mean()


def predict_shot(model,shot,*,intra,device='cpu',stride=32,step=.05,radius=1.,
                 objective='level',first_update=7,data_api=simple_data):
    """Emit t before updating; current observed feedback first affects t+1.

    Offline causal replay, NOT certified real-time execution. Early starts after
    eight real points; the full-window control starts after128. Both update every
    32 points. All initial predictions, short records and failed updates count.
    """
    if set(shot)&{'y','is_disrupt','down','horizon','IsDisrupt','DownTime'}:
        raise ValueError('outcome fields prohibited before seal')
    if first_update not in (7,127) or stride!=32 or step!=.05 or radius!=1. or objective!='level':
        raise ValueError('locked first7/127,stride32,step.05,radius1,level protocol')
    times=np.asarray(shot['times']);values=np.asarray(shot['values']);a=np.asarray(shot['available'])
    if (not len(times) or values.shape!=(len(times),23) or a.shape!=values.shape
            or a.dtype!=bool or not np.isfinite(times).all() or (np.diff(times)<=0).any()):
        raise ValueError('nonempty ordered observed record required')
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    fast=model.initial_fast(1).requires_grad_();origin=fast.detach().clone()
    first=model.summarize_fast(fast)
    signals=dict(values=values,available=a)
    ticks=list(range(first_update,len(times)-1,stride)) if intra else []
    scores=np.empty(len(times),np.float32);attempts=[];start=0
    for phase_index,end in enumerate([*ticks,len(times)-1]):
        for lo in range(start,end+1,128):
            hi=min(lo+128,end+1)
            x,mask=_batch_windows(data_api,signals,range(lo,hi),model,device)
            scores[lo:hi]=model(x,mask,fast.expand(hi-lo,-1))['logit'].detach().sigmoid().cpu().numpy()
        start=end+1
        if end not in ticks:continue
        x,mask=_batch_windows(data_api,signals,[end],model,device)
        phase=phase_index%4
        hidden=channel_mask(mask,phase=phase)
        visible=mask & ~hidden
        count=int(hidden.sum()) if visible.any() else 0
        row=dict(update_index=end,observed_through=end,effective_index=end+1,
            predictions_sealed_through_index=end,phase=phase,targets=count,accepted=False,
            fast_before=model.summarize_fast(fast),fast_after=model.summarize_fast(fast),
            loss_before=None,loss_after=None,reason='no_observed_hidden_targets')
        if count:
            loss=model.reconstruction_loss(x,mask,fast,phase=phase)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite current-signal loss')
            gradient,=torch.autograd.grad(loss,fast)
            candidate=normalized_step(fast.detach(),origin,gradient.detach(),step=step,radius=radius).requires_grad_()
            after=model.reconstruction_loss(x,mask,candidate,phase=phase)
            if not torch.isfinite(after):raise FloatingPointError('nonfinite current-signal candidate')
            accept=bool(after<loss.detach())
            row.update(loss_before=float(loss.detach()),loss_after=float(after.detach()),
                gradient_norm=float(gradient.detach().double().norm()),accepted=accept,
                reason='observed_loss_decreased' if accept else 'observed_loss_not_decreased')
            if accept:
                fast=candidate;row['fast_after']=model.summarize_fast(fast)
        fast=fast.detach().requires_grad_();attempts.append(row)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,initial[k],rtol=0,atol=0)
    return scores,dict(intra=bool(intra),site=model.site,score_count=len(scores),
        first_update=first_update,attempts=attempts,attempted_updates=len(attempts),
        accepted_updates=sum(r['accepted'] for r in attempts),fast_initial=first,
        fast_final=model.summarize_fast(fast),fast_discarded=True,
        model_parameter_change_norm=0.,model_buffer_change_norm=0.,
        auxiliary_decoder_updated=False,outcome_access=False,real_time_latency_validated=False)
