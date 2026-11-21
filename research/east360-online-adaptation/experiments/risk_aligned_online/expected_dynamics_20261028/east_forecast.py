"""Complete360 component audit; no risk prediction, optimization or model selection."""
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..delayed_recurrent_memory_20261025.model import patch_signal
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import fit_prefix_map, transform
from .forecast import DirectForecast, observed_features


@torch.no_grad()
def run(output, forecast_root, stream):
    if output.absolute().parent != io23.ROOT / 'probe_outputs' or output.exists():
        raise ValueError('new output under this worktree only')
    torch.set_num_threads(2)
    source = read(forecast_root, 'protocol.json'); done = read(forecast_root, 'complete.json')
    source_checkpoint = forecast_root / 'forecast_epoch10.pt'
    if not done['complete'] or sha(source_checkpoint) != done['checkpoint_sha256']:
        raise ValueError('completed frozen source forecast required')
    parent, _ = load_parent(DEFAULT_PARENT)
    f = DirectForecast().eval().requires_grad_(False)
    f.load_state_dict(torch.load(source_checkpoint, map_location='cpu', weights_only=True)['model'])
    weights = torch.tensor(read(Path(source['reference']), 'forecast_weights.json')['weights'])
    p = read(stream, 'protocol.json'); completion = read(stream, 'completed.json')
    order = read(stream, 'order.json'); mapping = read(stream, 'input_map.json')
    hashes = read(stream, 'input_hashes.json')
    ids = [r['shot'] for r in order]
    if (completion['status'] != 'COMPLETED' or len(ids) != 360 or len(set(ids)) != 360
            or p['warmup'] != 20 or p['threshold'] <= 0
            or p['source_sha256'] != PARENT_SHA256 or sha(stream/'order.json') != p['order_sha256']
            or mapping['calibration_ids'] != ids[:20] or mapping['policy'] != 'density_gain'):
        raise ValueError('complete locked commissioning360/cold20 reference required')
    output.mkdir(parents=True); (output/'seals').mkdir()
    plan = io23.ROOT/'EAST_FORECAST_TRANSFER_PLAN.md'
    protocol = dict(scope='EAST_COMPLETE360_FORECAST_COMPONENT_ONLY_NOT_RISK', stream=str(stream),
        order=ids, prefix=20, reference_hashes={n: sha(stream/n) for n in
            ('order.json', 'input_map.json', 'input_hashes.json', 'protocol.json', 'completed.json')},
        source_sha256=PARENT_SHA256, forecast_sha256=sha(source_checkpoint),
        plan_sha256=sha(plan), code_sha256=sha(Path(__file__)), optimizer_steps=0,
        predicted_quantity='source-normalized diagnostic patch increments; not disruption probability',
        scoring='all complete horizons ending no later than disruption after sealed forecasts; safe full record')
    _durable_json(output/'protocol.json', protocol)
    rows = []; prefix = []; started = time.monotonic()
    for index, sid in enumerate(ids):
        if sha(io23.path_for('EAST', sid)) != hashes[str(sid)]:
            raise ValueError('raw EAST record changed')
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        signal = {k: raw[k] for k in ('values', 'available')}
        if index >= 20:
            signal = transform(signal, mapping)
        x, a = patch_signal(signal, parent.center, parent.scale)
        if len(x) <= 4:
            raise ValueError('no complete forecast targets; do not silently omit a stream shot')
        pred = f(observed_features(x, a))
        _durable_json(output/'seals'/f'{index+1:04d}_{sid}.json', dict(shot=sid, index=index+1,
            predictions_sha256=hashlib.sha256(pred.numpy().tobytes()).hexdigest(),
            outcome_used=False, map_applied=index >= 20))
        flag, down = io23.read_outcome('EAST', sid)
        y = torch.stack([x[h:len(x)-4+h]-x[:-4] for h in range(1,5)],1)
        valid = torch.stack([a[h:len(x)-4+h]&a[:-4] for h in range(1,5)],1)
        endpoint_times = np.stack([raw['times'][(np.arange(len(x)-4)+h+1)*8-1] for h in range(1,5)],1)
        within = endpoint_times <= down if flag else np.ones_like(endpoint_times,dtype=bool)
        w = valid * torch.tensor(within)[...,None] * weights
        if w.sum() <= 0:
            raise ValueError('no eligible observed scored target')
        prediction = float(((pred[:-4]-y).square()*w).sum()/w.sum())
        persistence = float((y.square()*w).sum()/w.sum())
        rows.append(dict(shot=sid, index=index+1, disruptive=flag, prediction=prediction,
                         persistence=persistence, excluded_post_endpoint_horizons=int((~within).sum())))
        if index < 20:
            prefix.append({k:raw[k] for k in ('shot','device','values','available')})
        if index == 19:
            rebuilt = fit_prefix_map(mapping['source_statistics'],prefix,order=ids,count=20,policy='density_gain')
            if rebuilt != mapping:
                raise ValueError('ended-prefix alignment does not reproduce sealed map')
            prefix = []
        if (index+1) % 40 == 0:
            print(dict(shots=index+1,seconds=time.monotonic()-started),flush=True)
    if sum(r['disruptive'] for r in rows) != 180:
        raise ValueError('EAST class balance changed')
    summary = {}
    for section, subset in (('full360', rows), ('cold20', rows[:20]), ('suffix340', rows[20:])):
        summary[section] = {}
        for group in ('all', 'D', 'S'):
            part = [r for r in subset if group == 'all' or r['disruptive'] == (group == 'D')]
            pred_mean = float(np.mean([r['prediction'] for r in part]))
            held_mean = float(np.mean([r['persistence'] for r in part]))
            summary[section][group] = dict(shots=len(part), prediction=pred_mean, persistence=held_mean,
                relative_reduction=1-pred_mean/held_mean, improved=sum(r['prediction']<r['persistence'] for r in part))
    if sha(DEFAULT_PARENT) != PARENT_SHA256 or sha(plan) != protocol['plan_sha256']:
        raise ValueError('source or plan changed')
    _durable_json(output/'complete.json', dict(complete=True, summary=summary, rows=rows,
        seconds=time.monotonic()-started, source_unchanged=True, optimizer_steps=0,
        risk_predictions_computed=False, phase_labels_read=0, standalone_forecast_not_intra_gain=True))
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--forecast-root', type=Path, required=True)
    parser.add_argument('--stream', type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.forecast_root, args.stream)
