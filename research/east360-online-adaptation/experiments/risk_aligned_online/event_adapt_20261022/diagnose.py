"""Held-out source VAL audit: are predicted diagnostic changes meaningful?

No optimizer, no EAST access, no TEST access. Compare direct forecasts against
the explicit persistence prediction (zero change); report every channel.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23
from ..event_prefix_probe import DEFAULT_LABELS, read_events, assign_events
from ..simple_train import sha
from .model import Config, EventForecastModel
from .train import load_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError('do not overwrite a diagnostic')
    torch.set_num_threads(4)
    ckpt = torch.load(args.source, weights_only=False, map_location='cuda')
    model = EventForecastModel(Config(**ckpt['model_config']), **ckpt['coordinates']).to('cuda').eval()
    model.load_state_dict(ckpt['model'])
    shots = load_source(io23.source_roles()['val'], io23.advanced_times())
    events, _ = read_events(DEFAULT_LABELS)
    counts, error2, zero2, huber, zero_huber = [np.zeros(23, 'float64') for _ in range(5)]
    per_shot = []
    event_count = correct = 0
    with torch.no_grad():
        for s in shots:
            x = torch.as_tensor(s['values'], device='cuda')[None]
            mask = torch.as_tensor(s['available'], device='cuda')[None]
            out = model.sequence(x, mask)
            origins = np.arange(0, len(s['values'])-model.cfg.horizon, 8)
            if not len(origins): continue
            target_indices = origins[:, None]+np.arange(1, model.cfg.horizon+1)[None]
            z, observed = model.coordinates(x, mask)
            target = (z[0, target_indices]-z[0, origins, None])/model.change_scale
            use = observed[0, target_indices] & observed[0, origins, None] & model.forecast_active
            legal = s['valid'][target_indices] & s['valid'][origins, None]
            use &= torch.as_tensor(legal, device='cuda')[..., None]
            prediction = out['forecast'][0, origins]
            diff = (prediction-target).masked_fill(~use, 0.)
            zero = target.masked_fill(~use, 0.)
            loss = F.smooth_l1_loss(prediction, target, reduction='none').masked_fill(~use, 0.)
            base = F.smooth_l1_loss(torch.zeros_like(target), target, reduction='none').masked_fill(~use, 0.)
            counts += use.sum((0, 1)).cpu().numpy()
            error2 += diff.square().sum((0, 1)).cpu().numpy()
            zero2 += zero.square().sum((0, 1)).cpu().numpy()
            huber += loss.sum((0, 1)).cpu().numpy()
            zero_huber += base.sum((0, 1)).cpu().numpy()
            per_shot.append(dict(shot=s['shot'], prediction_huber=float(loss.sum()/use.sum().clamp_min(1)),
                                 persistence_huber=float(base.sum()/use.sum().clamp_min(1))))
            e, _ = assign_events(s['times'], events.get(s['shot'], []))
            known = (e >= 0) & s['valid']
            pred = out['event'][0].argmax(-1).cpu().numpy()
            event_count += int(known.sum()); correct += int((pred[known] == e[known]).sum())
    channels = {}
    for c, name in enumerate(io23.RAW23):
        channels[name] = dict(n=int(counts[c]), mse=float(error2[c]/counts[c]) if counts[c] else None,
            persistence_mse=float(zero2[c]/counts[c]) if counts[c] else None,
            mse_skill_vs_persistence=float(1-error2[c]/zero2[c]) if zero2[c] else None,
            huber_skill_vs_persistence=float(1-huber[c]/zero_huber[c]) if zero_huber[c] else None)
    report = dict(source_sha256=sha(args.source), split='J-TEXT VAL160 only; no parameter update',
        forecast_semantics='change from current observed value; source asinh/delta-RMS coordinates',
        evaluation='origin stride8; only complete 32-point record horizon; valid preterminal targets',
        counts=int(counts.sum()), mse_skill=float(1-error2.sum()/zero2.sum()),
        huber_skill=float(1-huber.sum()/zero_huber.sum()),
        event_conditional_endpoint_accuracy=correct/event_count if event_count else None,
        event_points=event_count, channels=channels, per_shot=per_shot,
        independent_confirmation=False, source_meta_prepared=False)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('channels', 'per_shot')}), flush=True)


if __name__ == '__main__': main()
