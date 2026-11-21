"""Post-hoc diagnostics, not a training or model-selection entry point."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ..direction_teaching_20261026.learning import objective
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import KINDS


def reference_only_step(label):
    """Counterexample in independent logit coordinates, not a real model update.

    The shared network cannot in general change these coordinates independently.
    This demonstrates an objective loophole, not its measured effect on EAST.
    """
    y = torch.tensor([label], dtype=torch.double)
    after = torch.tensor([.2 if label else -.2], dtype=torch.double)
    before = torch.zeros(1, dtype=torch.double, requires_grad=True)
    e = dict(flag=bool(label), labels=y, local_mask=torch.ones(1, dtype=torch.bool),
             direction_mask=torch.ones(1, dtype=torch.bool))
    _, parts = objective(after, before, e, .5, 'direction')
    gradient, = torch.autograd.grad(parts['direction'], before)
    changed = before.detach()-.1*gradient
    _, new = objective(after, changed, e, .5, 'direction')
    reference_loss_change = F.binary_cross_entropy_with_logits(changed, y)-F.binary_cross_entropy_with_logits(before, y)
    return dict(label=label, reference_logit_before=float(before.detach()),
                reference_logit_after=float(changed), reference_gradient=float(gradient),
                reference_loss_change=float(reference_loss_change.detach()),
                direction_loss_change=float((new['direction']-parts['direction']).detach()),
                after_loss_change=0., deployed_model_modified=False,
                scope='INDEPENDENT_LOGIT_COUNTEREXAMPLE_NOT_EAST_CAUSAL_ATTRIBUTION')


def projection_summary(weight):
    w = weight.detach().double()
    if w.ndim != 2 or w.shape[1] != 92 or not torch.isfinite(w).all():
        raise ValueError('finite output projection with four23-channel blocks required')
    spectrum = torch.linalg.svdvals(w).square()
    energy = spectrum.sum()
    if energy <= 0:
        raise ValueError('nonzero learned projection required')
    shares = spectrum/energy
    blocks = [w[:, 23*h:23*(h+1)].reshape(-1) for h in range(4)]
    cosines = [float(F.cosine_similarity(a, b, dim=0))
               for i, a in enumerate(blocks) for b in blocks[i+1:]]
    return dict(largest_energy_fraction=float(shares[0]),
                top4_energy_fraction=float(shares[:4].sum()),
                participation_rank=float(1/shares.square().sum()),
                horizon_cosines=cosines, frobenius_norm=float(w.norm()))


def audit(root):
    target = root/'source_mechanism_audit.json'
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    done = json.loads((root/'complete.json').read_text())
    verified = json.loads((root/'audit.json').read_text())
    if not done['complete'] or verified['complete_sha256'] != sha(root/'complete.json'):
        raise ValueError('completed audited source training required')
    projections = {}
    for kind in KINDS:
        path = root/f'{kind}_epoch2.pt'
        if sha(path) != done['checkpoints'][kind]:
            raise ValueError('changed source memory checkpoint')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        projections[kind] = projection_summary(checkpoint['memory']['output.weight'])
    result = dict(scope='POST_HOC_SOURCE_CHECKPOINT_AND_OBJECTIVE_ONLY',
        source_audit_sha256=sha(root/'audit.json'),
        source_event_probe_sha256=sha(root/'source_event_fast_ablation.json'),
        projections=projections, reference_only=[reference_only_step(y) for y in (0., 1.)],
        interpretation='Relative loss can decrease by degrading its trainable reference. '
                       'Projection similarity is descriptive, not proof of causal failure.',
        model_parameters_modified=False, east_outcomes_read=0,
        code_sha256=sha(Path(__file__)))
    _durable_json(target, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
