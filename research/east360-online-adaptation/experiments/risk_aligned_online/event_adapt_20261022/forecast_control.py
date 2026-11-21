"""Source-only diagnostic: can the auxiliary task be learned without risk loss?

The resulting weights are NOT qualified source models, and must not be promoted
on auxiliary loss alone. No TEST or EAST data are read. Compare decoder-only
and full-network training from the same source candidate for a fixed budget.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23
from ..event_prefix_probe import DEFAULT_LABELS, read_events
from ..simple_train import pools, sha
from .model import Config, EventForecastModel, arrived_loss
from .data import attach_training_events
from .train import load_source, event_pool, sample_batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scope', choices=['head', 'full'], required=True)
    p.add_argument('--steps', type=int, default=1024)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError('never overwrite a diagnostic experiment')
    if args.steps < 1: p.error('positive diagnostic budget required')
    torch.set_num_threads(4); torch.manual_seed(271)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(271); start = time.monotonic()
    source_sha = sha(args.source)
    ckpt = torch.load(args.source, weights_only=False, map_location='cuda')
    model = EventForecastModel(Config(**ckpt['model_config']), **ckpt['coordinates']).to('cuda')
    model.load_state_dict(ckpt['model'])
    roles = io23.source_roles()
    shots = load_source(roles['train'], io23.advanced_times())
    events, _ = read_events(DEFAULT_LABELS)
    shots = attach_training_events(shots, events, allowed_train_ids=roles['train'])
    groups, pool = pools(shots), event_pool(shots)
    args.output.mkdir(parents=True)
    config = dict(scope=args.scope, steps=args.steps, source_sha256=source_sha,
                  training_split='canonical J-TEXT TRAIN1353', validation_split='VAL160',
                  test_access=False, east_access=False, seed=271, lr=3e-4, batch_size=32,
                  purpose='diagnostic auxiliary learnability, NOT risk qualification',
                  code_sha256=sha(__file__))
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    # Measure competing task gradients at the common initialization, on TRAIN.
    diagnostic = []
    params = [p for n, p in model.named_parameters() if n.startswith(('embedding.', 'blocks.', 'fast_', 'output_norm.'))]
    for _ in range(16):
        x, m, y, _, target, tm = sample_batch(shots, groups, pool, rng, model.cfg, 32, 'cuda')
        out = model(x, m)
        risk = F.binary_cross_entropy_with_logits(out['logit'], y.float())
        fore, n = arrived_loss(model, out['forecast'], x[:, -1], m[:, -1], target, tm)
        fore = fore[n > 0].mean()
        g = torch.autograd.grad(risk, params, retain_graph=True)
        f = torch.autograd.grad(fore, params)
        g, f = torch.cat([v.flatten() for v in g]), torch.cat([v.flatten() for v in f])
        diagnostic.append(dict(risk_norm=float(g.norm()), forecast_norm=float(f.norm()),
                               cosine=float(F.cosine_similarity(g[None], f[None]))))
    (args.output/'initial_gradient_audit.json').write_text(json.dumps(diagnostic, indent=2))
    for name, param in model.named_parameters():
        param.requires_grad_(args.scope == 'full' or name.startswith('forecast_head.'))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=.01)
    history = []
    for step in range(1, args.steps+1):
        x, m, _, _, target, tm = sample_batch(shots, groups, pool, rng, model.cfg, 32, 'cuda')
        optimizer.zero_grad(set_to_none=True)
        out = model(x, m)
        loss, n = arrived_loss(model, out['forecast'], x[:, -1], m[:, -1], target, tm)
        loss = loss[n > 0].mean()
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite forecast-only loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step(); history.append(float(loss.detach()))
        if step % 256 == 0:
            print(json.dumps(dict(step=step, scope=args.scope, loss=float(np.mean(history[-256:])))), flush=True)
    ckpt['model'] = model.state_dict()
    ckpt['diagnostic_only'] = True
    torch.save(ckpt, args.output/'diagnostic.pt')
    if sha(args.source) != source_sha: raise AssertionError('source file changed')
    subprocess.run([sys.executable, '-m', 'experiments.risk_aligned_online.event_adapt_20261022.diagnose',
                    '--source', str(args.output/'diagnostic.pt'),
                    '--output', str(args.output/'forecast_val160.json')], check=True)
    report = dict(status='COMPLETED', scope=args.scope, steps=args.steps,
                  elapsed_seconds=time.monotonic()-start, last_256_loss=float(np.mean(history[-256:])),
                  source_unchanged=True, test_access=False, east_access=False)
    (args.output/'completed.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
