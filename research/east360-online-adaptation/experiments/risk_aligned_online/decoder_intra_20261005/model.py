"""Direct H25 diagnostic CHANGE prediction; the original risk path is unchanged."""
import torch

from ..ffn_intra_20261005.model import FfnModel


def last_observation(values,available,center,scale):
    """Last observed value, source-asinh coordinates, detached from adaptation.

    If this channel has never appeared in the supplied prefix, use the fixed
    source centre (zero in these coordinates) and return seen=False. This is a
    declared source-only prior, not an estimated target scale or false anchor.
    """
    observed=available & torch.isfinite(values)
    positions=torch.arange(values.shape[1],device=values.device)[None,:,None]
    last=positions.masked_fill(~observed,-1).amax(1)
    seen=last>=0
    raw=values.gather(1,last.clamp_min(0)[:,None]).squeeze(1)
    raw=torch.where(seen,raw,center)
    return torch.asinh((raw-center)/scale).detach(),seen


class AnchoredFfnModel(FfnModel):
    def __init__(self,base,*,initialize):
        super().__init__(base)
        if initialize:
            with torch.no_grad():
                self.base.forecast_head.weight.zero_()
                self.base.forecast_head.bias.zero_()

    def forward(self,values,available,delta):
        out=super().forward(values,available,delta)
        anchor,_=last_observation(values,available,self.center,self.scale)
        return dict(logit=out['logit'],forecast=anchor[:,None]+out['forecast'])
