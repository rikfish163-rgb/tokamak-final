"""End-of-shot diagnostic teaching with bounded risk-gradient credit."""

import torch

from risk_feature_intra_20261107 import FeatureIntra
from risk_meta_sequence_20261108 import teacher_sequence


def teach_decoder(slow, decoder, optimizer, episodes, query, query_available, labels, *, credit):
    """Update the persistent decoder once using already available EOS inputs.

    The caller supplies the attention backend, matured signal episodes and
    appropriately timed labels, and owns the persistent optimizer. Risk credit
    is either stopped or restricted to the last temporary signal update. The
    reported losses and mean query logit are measured before the decoder step;
    gradient_norm is the decoder gradient norm before clipping.
    """
    parameters = tuple(decoder.parameters())
    try:
        if credit not in ('none', 'last'):
            raise ValueError('credit must be none or last')
        optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
        if optimized != {id(p) for p in parameters}:
            raise ValueError('optimizer parameters must match decoder parameters')

        optimizer.zero_grad(set_to_none=True)
        model = FeatureIntra(slow, decoder)
        model.decoder = decoder
        model.eval()
        decoder.requires_grad_(True)
        with torch.enable_grad():
            result = teacher_sequence(model, episodes, query, query_available, labels, credit=credit)
            loss = result['signal_loss'] + result['risk_loss']
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            stats = dict(loss=float(loss.detach()),
                         signal_loss=float(result['signal_loss'].detach()),
                         risk_loss=float(result['risk_loss'].detach()),
                         gradient_norm=float(norm.detach()),
                         logit=float(result['logit'].detach().mean()))
            optimizer.step()
        return stats
    finally:
        decoder.requires_grad_(False)
        for parameter in parameters:
            parameter.grad = None
