"""Learn only 23 auxiliary-loss weights on source TRAIN; evaluate on source VAL.

No classifier, forecast head, threshold or feature scaler is trained. The
outer risk label chooses weights offline; deployment uses signal errors only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import numpy as np
from torch.nn import functional as F

from . import simple_data, simple_train
from .channel_loss import weighted_loss
from .sequence_fast import unroll
from .sequence_train import pools, sample
from .simple23 import Config, Simple23
from .simple_online import predict_shot


class Weighted23(Simple23):
    def __init__(self, checkpoint, logits):
        super().__init__(Config(**checkpoint['model_config']), checkpoint['source_center'], checkpoint['source_scale'])
        self.load_state_dict(checkpoint['model'])
        self.register_buffer('channel_logits', logits.detach().clone())

    def forecast_loss(self, prediction, truth, available):
        mask = available & torch.isfinite(truth)
        truth = torch.where(mask, truth.detach(), self.center)
        target = torch.asinh((truth-self.center)/self.scale)
        losses = F.smooth_l1_loss(prediction, target, reduction='none')
        return weighted_loss(losses, mask, self.channel_logits).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=512)
    args = parser.parse_args()
    if args.steps < 1 or args.output.exists():
        raise ValueError('positive steps and new output required')
    torch.set_num_threads(4)
    torch.manual_seed(131)
    rng = np.random.default_rng(131)
    parent = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = Weighted23(parent, torch.zeros(23)).cuda().eval()
    model.requires_grad_(False)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    logits = torch.nn.Parameter(torch.zeros(23, device='cuda'))
    optimizer = torch.optim.Adam([logits], lr=.01)
    roles = simple_data.source_roles()
    advanced = simple_data.advanced_times()
    shots = simple_train.load_shots(roles['train'], advanced)
    pool = pools(shots, model.cfg, 4)
    args.output.mkdir(parents=True)
    protocol = dict(parent_sha256=simple_train.sha(args.checkpoint), train_ids=roles['train'],
                    val_ids=roles['val'], steps=args.steps, depth=4, batch=8, lr=.01, seed=131,
                    objective='final post-update BCE only; 23 channel logits only',
                    threshold=parent['threshold'], evidence='DEVELOPMENT_ONLY',
                    code_sha256={n:simple_train.sha(Path(__file__).with_name(n)) for n in
                                 ('channel_meta.py', 'channel_loss.py', 'sequence_fast.py', 'sequence_train.py',
                                  'simple_online.py', 'simple23.py', 'fast_direction.py', 'fast_update.py')})
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    digest = hashlib.sha256()
    history = []
    for step in range(args.steps):
        episodes, manifest = sample(shots, pool, model.cfg, rng, 4, 8, 'cuda')
        digest.update(json.dumps(manifest).encode())
        optimizer.zero_grad(set_to_none=True)
        result = unroll(model, episodes, create_graph=True, channel_logits=logits)
        risk = F.binary_cross_entropy_with_logits(result['logits'][-1], episodes[-1]['y'])
        risk.backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            raise FloatingPointError('invalid channel meta-gradient')
        optimizer.step()
        history.append(dict(risk=float(risk.detach()), grad_norm=float(logits.grad.norm())))
        if (step+1) % 64 == 0:
            print(json.dumps(dict(step=step+1, **history[-1])), flush=True)
    assert all(torch.equal(v, before[k]) for k,v in model.state_dict().items())
    protocol['sample_sha256'] = digest.hexdigest()
    weights = (23*logits.softmax(0)).detach().cpu().tolist()
    torch.save(dict(channel_logits=logits.detach().cpu(), parent_sha256=protocol['parent_sha256'],
                    protocol=protocol), args.output/'loss_weights.pt')
    (args.output/'training.json').write_text(json.dumps(dict(protocol=protocol, weights=weights, history=history), indent=2))
    val = simple_train.load_shots(roles['val'], advanced)
    results = {}
    for arm in ('static', 'uniform', 'learned'):
        model.channel_logits.copy_(logits.detach() if arm == 'learned' else torch.zeros_like(logits))
        rows = []
        for shot in val:
            scores, _ = predict_shot(model, shot, intra=arm != 'static', device='cuda')
            rows.append(dict(shot=shot['shot'], is_disrupt=shot['is_disrupt'], down=shot['down'],
                             times=shot['times'][shot['valid']], scores=scores[shot['valid']]))
        results[arm] = simple_train.metrics(rows, parent['threshold'])
        print(json.dumps(dict(arm=arm, **{k:v for k,v in results[arm].items() if k != 'predictions'})), flush=True)
    (args.output/'validation.json').write_text(json.dumps(dict(protocol=protocol, weights=weights, arms=results), indent=2))


if __name__ == '__main__':
    main()
