"""Fit a source diagnostic expectation; no classifier update or EAST access."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from .. import io23, simple_data
from ..context_memory_20261028.checkpoint import read
from ..context_memory_20261028.scaled import validate_weights
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..source_task_memory_20261026.learning import validate_roles
from .forecast import DirectForecast, direct_examples, forecast_screen


def run(output, reference):
    if output.absolute().parent != io23.ROOT / 'probe_outputs' or output.exists():
        raise ValueError('new output within this worktree/probe_outputs required')
    if not torch.cuda.is_available():
        raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2); torch.manual_seed(131)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    roles = simple_data.source_roles(); validate_roles(roles)
    previous = read(reference, 'protocol.json')
    cache = Path(previous['cache_root'])
    if (previous['train_ids'] != roles['train'] or previous['val_ids'] != roles['val']
            or sha(cache/'inventory.json') != previous['cache_inventory_sha256']
            or sha(reference/'forecast_weights.json') != previous['forecast_weights_sha256']
            or sha(DEFAULT_PARENT) != PARENT_SHA256):
        raise ValueError('locked source identities/cache/scales/parent required')
    weights = torch.tensor(read(reference, 'forecast_weights.json')['weights'])
    validate_weights(weights)
    inventory = read(cache, 'inventory.json')
    examples, exclusions = {}, []
    for record in inventory:
        sid, role = record['shot'], record['role']
        path = cache/'cache'/f'{sid}.pt'
        if role not in ('train', 'val') or sid not in roles[role] or sha(path) != record['cache_sha256']:
            raise ValueError('canonical TRAIN/VAL cache required')
        e = torch.load(path, map_location='cpu', weights_only=True)
        disruptive, down = io23.read_outcome('J-TEXT', sid)
        with h5py.File(io23.path_for('J-TEXT', sid), 'r') as handle:
            ds = handle['data/ip']
            rate = float(np.asarray(ds.attrs['SampleRate']).reshape(-1)[0])
            start = float(np.asarray(ds.attrs['StartTime']).reshape(-1)[0])
            count = len(ds)//8
        if count != len(e['means']) or bool(e['flag']) != disruptive:
            raise ValueError('source cache identity or patch clock changed')
        last_time = start+(count*8-1)/rate
        if disruptive and last_time > down + 1e-12:
            raise ValueError('source forecast crosses observed disruption endpoint; explicit new contract required')
        if count <= 4:
            # Audited eligibility, not imputation or an exception bypass. These
            # four TRAIN shots have no complete 4-horizon target and are logged.
            exclusions.append(dict(shot=sid, role=role, patches=count, reason='no_complete_four_horizon_origin'))
            continue
        x, y, mask = direct_examples(e['means'], e['complete'])
        w = mask * weights
        if w.sum() <= 0:
            raise ValueError('shot has no observed diagnostic target')
        examples[sid] = dict(x=x, y=y, w=w, role=role, disruptive=disruptive)
    expected_excluded = {1052590, 1053119, 1053639, 1057940}
    if {r['shot'] for r in exclusions} != expected_excluded or any(r['role'] != 'train' for r in exclusions):
        raise ValueError('eligibility differs from pre-run census')
    train_ids = [sid for sid in roles['train'] if sid in examples]
    if len(train_ids) != 1349 or any(sid not in examples for sid in roles['val']):
        raise ValueError('declared 1349 eligible TRAIN and complete VAL160 required')
    counts = {flag: sum(examples[sid]['disruptive'] == flag for sid in train_ids) for flag in (False, True)}
    pieces = {key: [] for key in ('x', 'y', 'w')}
    for sid in train_ids:
        e = examples[sid]
        pieces['x'].append(e['x']); pieces['y'].append(e['y'])
        pieces['w'].append(e['w'] / e['w'].sum() * (.5/counts[e['disruptive']]))
    dataset = {k: torch.cat(v).cuda() for k, v in pieces.items()}
    if abs(float(dataset['w'].sum()) - 1.) > 1e-5:
        raise ValueError('class/shot/observed-target weights do not sum to one')
    n = len(dataset['x'])
    model = DirectForecast().cuda().train()
    output.mkdir(parents=True)
    plan = io23.ROOT/'EXPECTED_DYNAMICS_PLAN.md'
    files = [Path(__file__), Path(__file__).parent/'forecast.py']
    protocol = dict(profile='expected_diagnostic_forecast_v1_NOT_A_RISK_MODEL', seed=131,
        train_ids=train_ids, canonical_train_ids=roles['train'], val_ids=roles['val'],
        excluded=exclusions, epochs=10, selected_epoch=10, learning_rate=.0003, batch_size=512,
        gradient_clip=1., source_sha256=PARENT_SHA256, parent_loaded=False, forecast_origins=n,
        architecture='92-64-SiLU-92 direct four increments', parameters=sum(q.numel() for q in model.parameters()),
        weights='class equal, then shot equal, then observed horizon/channel weighted by fixed TRAIN scale',
        train_class_counts={str(k):v for k,v in counts.items()}, cache=str(cache),
        cache_inventory_sha256=sha(cache/'inventory.json'), reference=str(reference),
        forecast_weights_sha256=sha(reference/'forecast_weights.json'), plan_sha256=sha(plan),
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        east_records_read=0, test220_records_read=0, source_event_labels_read=0,
        original_source_classifier_modified=False, torch=torch.__version__, gpu=torch.cuda.get_device_name())
    _durable_json(output/'protocol.json', protocol)
    start = time.monotonic()

    @torch.no_grad()
    def evaluate(ids):
        rows = []
        for sid in ids:
            e = examples[sid]
            x, y, w = (e[k].cuda() for k in ('x', 'y', 'w'))
            prediction = torch.cat([model(x[i:i+512]) for i in range(0, len(x), 512)])
            error = float(((prediction-y).square()*w).sum()/w.sum())
            persistence = float((y.square()*w).sum()/w.sum())
            rows.append(dict(shot=sid, disruptive=e['disruptive'], origins=len(x),
                             prediction=error, persistence=persistence))
        summary = {}
        for name in ('all', 'D', 'S'):
            subset = [r for r in rows if name == 'all' or r['disruptive'] == (name == 'D')]
            summary[name] = dict(shots=len(subset),
                prediction=float(np.mean([r['prediction'] for r in subset])),
                persistence=float(np.mean([r['persistence'] for r in subset])),
                improved=sum(r['prediction'] < r['persistence'] for r in subset))
        return dict(rows=rows, summary=summary)

    initial = evaluate(roles['val'])
    if any(r['prediction'] != r['persistence'] for r in initial['rows']):
        raise ValueError('zero-initialized forecast does not equal persistence')
    _durable_json(output/'val_epoch0.json', initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    rng = np.random.default_rng(131); history = []
    for epoch in range(1, 11):
        order_cpu = rng.permutation(n)
        order = torch.tensor(order_cpu, device='cuda')
        totals = []; gradients = []
        for lo in range(0, n, 512):
            ids = order[lo:lo+512]
            prediction = model(dataset['x'][ids])
            loss = ((prediction-dataset['y'][ids]).square()*dataset['w'][ids]).sum()*n/len(ids)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite source forecast objective')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
            optimizer.step(); totals.append(float(loss.detach())); gradients.append(norm)
        result = evaluate(roles['val'])
        _durable_json(output/f'val_epoch{epoch}.json', result)
        # Interim validation is reported, never used for checkpoint selection.
        history.append(dict(epoch=epoch, optimizer_steps=len(totals), train_batch_mean=float(np.mean(totals)),
                            nonzero_gradient_steps=sum(g>0 for g in gradients),
                            order_sha256=hashlib.sha256(order_cpu.tobytes()).hexdigest()))
        _durable_json(output/'training.json', history)
        print(json.dumps(dict(epoch=epoch, seconds=time.monotonic()-start, summary=result['summary'])),flush=True)
    torch.save(dict(profile=protocol['profile'], protocol_sha256=sha(output/'protocol.json'),
                    model={k:v.detach().cpu() for k,v in model.state_dict().items()}), output/'forecast_epoch10.pt')
    if sha(DEFAULT_PARENT) != PARENT_SHA256 or sha(plan) != protocol['plan_sha256']:
        raise ValueError('original classifier or preregistered plan changed')
    for f,h in protocol['code_sha256'].items():
        if sha(io23.ROOT/f) != h:
            raise ValueError('source forecast code changed during training')
    done = dict(complete=True, final_epoch=10, seconds=time.monotonic()-start, source_classifier_unchanged=True,
        gate=forecast_screen(result['rows']), summary=result['summary'], east_records_read=0, test220_records_read=0,
        checkpoint_sha256=sha(output/'forecast_epoch10.pt'))
    _durable_json(output/'complete.json',done)
    print(json.dumps(done,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args()
    run(args.output,args.reference)
