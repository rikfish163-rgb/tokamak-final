"""Fit two auxiliary targets on the same already-ended twenty discharges."""
import numpy as np
import torch
from torch.nn import functional as F

from ..anchored_feedback_20261030.learning import AnchoredTarget
from ..anchored_feedback_20261030.run import state_for
from ..conditional_weights_20261030.run import draw_balanced, subset, training_pools
from ..context_vector_20261030.scaling import fit_context_scale, transform_context
from ..maturity_jacobian_20261030.run import ORIGIN, load_inputs
from ..simple_online import _durable_json
from .core import VARIANTS


def fit_policies(net, slow, ids, output):
    _, _, data, mature, records, _ = load_inputs(ORIGIN)
    shot_ids = [r['shot'] for r in records]
    if set(shot_ids) != set(ids) or len(ids) != 20:
        raise ValueError('frozen cold20 observations required')
    scale = fit_context_scale(data['context'], shot_ids)
    scaled = transform_context(data['context'], scale)
    mean = torch.stack([scaled[np.array(shot_ids) == sid].mean(0) for sid in ids]).mean(0)
    pools = training_pools(records, ids)
    policies, paired_draws = {}, None
    for variant in VARIANTS:
        inputs = data if variant == 'origin' else mature
        torch.manual_seed(131)
        controller = AnchoredTarget().cuda()
        optimizer = torch.optim.Adam(controller.parameters(), lr=.001)
        rng, history, draws = np.random.default_rng(131), [], []
        for iteration in range(256):
            take = draw_balanced(pools, rng)
            d = subset(inputs, take)
            state = state_for(net, controller, 'anchored', d, scaled[take].cuda(), mean.cuda(), outer=True)
            logits = net(d['later_x'], d['later_mask'], state, slow, 'qk', forecast=False)['logit']
            y = torch.tensor([float(records[i]['group'] == 'danger') for i in take], device='cuda')
            loss = F.binary_cross_entropy_with_logits(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite auxiliary teaching loss')
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            draws.append(take.tolist())
            history.append(dict(step=iteration+1, loss=float(loss.detach()), gradient_norm=float(norm)))
            if (iteration+1) % 64 == 0:
                print(dict(variant=variant, **history[-1]), flush=True)
        if paired_draws is not None and paired_draws != draws:
            raise AssertionError('training samples must be paired')
        paired_draws = draws
        _durable_json(output/f'{variant}_training.json', dict(draws=draws, history=history))
        saved = dict(state={k: v.detach().cpu() for k, v in controller.state_dict().items()},
                     scale=scale, mean_context=mean, donor_origins=scaled[:, :128],
                     donor_feedback=scaled[:, 128:], fit_ids=ids, radius=.1,
                     variant=variant, record_count=len(records))
        torch.save(saved, output/f'{variant}.pt')
        policies[variant] = unpack(saved)
    return policies


def unpack(saved):
    controller = AnchoredTarget().cuda()
    controller.load_state_dict(saved['state'], strict=True)
    return dict(controller=controller.eval().requires_grad_(False), scale=saved['scale'],
                mean_context=saved['mean_context'].cuda(), donor_origins=saved['donor_origins'].cuda(),
                donor_feedback=saved['donor_feedback'].cuda(), radius=saved['radius'])
