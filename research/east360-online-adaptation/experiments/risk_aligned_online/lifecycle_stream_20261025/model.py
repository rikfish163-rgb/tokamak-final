"""Persistent and shot-local changes both enter between Transformer blocks.

No classifier/threshold residual. The immutable source and risk head remain
untouched. The fast matrix writes matured direct multi-step diagnostic targets.
"""
import torch
from torch import nn

from ..delayed_recurrent_memory_20261025.model import (
    DiagnosticMemory, RecurrentAdapter, patch_signal, endpoint_corrections,
)
from ..progressive_feedback_20261026.model import ProgressiveDiagnosticMemory
from ..ordered_readout_20261025.learning import decompose


class LifecycleAdapter(RecurrentAdapter):
    def __init__(self,base,*,readout='shape',feedback='delayed'):
        if readout not in ('full','shape','level'):
            raise ValueError('readout must be full, shape or level')
        if feedback not in ('delayed','progressive'):
            raise ValueError('feedback must be delayed or progressive')
        super().__init__(base,memory_class=DiagnosticMemory if feedback=='delayed' else ProgressiveDiagnosticMemory)
        self.feedback=feedback
        self.readout=readout
        self.slow=nn.Linear(base.cfg.dim,base.cfg.dim,bias=False)
        nn.init.zeros_(self.slow.weight)
        self.memory.requires_grad_(False)
        self.memory.output.requires_grad_(True)

    def risk(self,h,valid,delta):
        # Linear backward needs its input token unchanged. Construct a new
        # sequence rather than overwrite the view saved by self.slow.
        last=h[:,-2]+self.slow(h[:,-2])*valid[:,-2,None]
        adapted=torch.cat((h[:,:-2],last[:,None],h[:,-1:]),dim=1)
        return super().risk(adapted,valid,decompose(delta,self.readout))

    def forward(self,values,available,*,delta):
        h,valid=self.tokens(values,available)
        return self.risk(h,valid,delta)

    @torch.no_grad()
    def memory_delta(self,signal,mode='memory'):
        means,complete=patch_signal(signal,self.base.center,self.base.scale)
        main=self.memory.trace(means,complete,mode='memory')
        trace=main if mode=='memory' else self.memory.trace(means,complete,mode=mode,
                                                          reference_norm=main['norm'])
        ends=torch.arange(len(signal['values']),device=self.base.center.device)
        delta=endpoint_corrections(trace['delta'],ends).detach()
        # Observability only: these summaries never control learning or alarms.
        # Float64 decomposition makes the orthogonal energy identity auditable.
        energy={k:float(decompose(delta.double(),k).square().sum())
                for k in ('full','shape','level')}
        info=dict(updates=trace['updates'],first_affected_index=40 if self.feedback=='delayed' else 16,
                          max_norm=float(trace['norm'].max()) if len(trace['norm']) else 0.,
                          forecast_loss=float(trace['forecast_loss']),reset_each_shot=True,
                          readout=self.readout,readout_energy=energy)
        if self.feedback=='progressive':
            info.update(feedback=self.feedback,horizon_targets=trace['horizon_targets'],
                        target_scalars=trace['target_scalars'])
        return delta,info

    def adapted_state(self):
        return dict(slow=self.slow.weight.detach().clone(),
                    fast_output=self.memory.output.weight.detach().clone())

    @torch.no_grad()
    def restore(self,state):
        self.slow.weight.copy_(state['slow']);self.memory.output.weight.copy_(state['fast_output'])
