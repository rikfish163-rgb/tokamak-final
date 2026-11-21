"""Posthoc direction audit of every post-startup shot; no model training."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..delayed_recurrent_memory_20261025.learning import batch
from ..delayed_recurrent_memory_20261025.model import patch_signal, endpoint_corrections
from ..lifecycle_stream_20261025.model import LifecycleAdapter
from ..redesign_20261004.prepare_prior import load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..source_task_memory_20261026.checkpoint import load_source_memory
from ..warmstart_alignment_20261022.alignment import transform
from .analysis import select_endpoints, selected_states, channel_read_ablation, direction_summary


def read(path):
    return json.loads(path.read_text())


def scalar(value):
    return float(value.detach())


def loss(logit, outcome, threshold):
    target = torch.full_like(logit, float(outcome))
    return F.binary_cross_entropy_with_logits(logit-math.log(threshold/(1-threshold)), target)


def measure(model, h, valid, deltas, offset, state, outcome, threshold):
    """Only this posthoc measurement receives outcome; reconstruction does not."""
    before = state['before'].detach().clone().requires_grad_(True)

    def evaluate(w):
        changed = deltas.clone()
        changed[offset] = state['key'] @ (w-model.memory.prior)
        return model.risk(h, valid, changed)[offset]

    z_before = evaluate(before)
    l_before = loss(z_before, outcome, threshold)
    gradient, = torch.autograd.grad(l_before, before)
    with torch.no_grad():
        z_after = evaluate(state['after'])
        l_after = loss(z_after, outcome, threshold)
        # Same32 risk batch size, with one diagnostic readout removed per row.
        full = deltas[offset:offset+1]
        ablated = torch.cat((channel_read_ablation(full), full.expand(9, -1)))
        all_z = model.risk(h[offset:offset+1].expand(32, -1, -1),
                           valid[offset:offset+1].expand(32, -1), ablated)
        signed_channel = (all_z[23].sigmoid()-all_z[:23].sigmoid()).cpu().tolist()
    result = direction_summary(gradient, state['after']-state['before'])
    result.update(probability_before_last_write=scalar(z_before.sigmoid()),
        probability_after_last_write=scalar(z_after.sigmoid()),
        operational_loss_before_last_write=scalar(l_before),
        operational_loss_after_last_write=scalar(l_after),
        actual_loss_change=scalar(l_after-l_before),
        signal_residual=state['signal_residual'],
        observed_target_scalars=state['observed_target_scalars'],
        channel_removal_probability_effect=signed_channel,
        channel_batch_reference_error=abs(scalar(all_z[23].sigmoid()-z_after.sigmoid())))
    return result


def summarize(rows):
    result = {}
    for role in ('primary', 'late125'):
        for flag in (False, True):
            selected = [r for r in rows if role in r['roles'] and r['disruptive'] == flag]
            if not selected:
                continue
            measured = [r for r in selected if 'actual_loss_change' in r]
            change = [(1 if flag else -1)*(r['intra_probability']-r['cold_probability'])
                      for r in selected]
            key = role+('_D' if flag else '_S')
            result[key] = dict(shots=len(selected), measured=len(measured),
                fast_probability_improved=sum(d > 1e-7 for d in change),
                fast_probability_worsened=sum(d < -1e-7 for d in change),
                mean_signed_probability_change=float(np.mean(change)),
                last_write_helped=sum(r['actual_loss_change'] < -1e-7 for r in measured),
                last_write_harmed=sum(r['actual_loss_change'] > 1e-7 for r in measured),
                signal_fit_improved_but_risk_harmed=sum(
                    r['signal_residual']['before'] is not None
                    and r['signal_residual']['after'] < r['signal_residual']['before']
                    and r['actual_loss_change'] > 1e-7 for r in measured),
                first_order_helped=sum(r['first_order_loss_change'] < -1e-7 for r in measured),
                first_order_harmed=sum(r['first_order_loss_change'] > 1e-7 for r in measured),
                mean_actual_loss_change=float(np.mean([r['actual_loss_change'] for r in measured]))
                if measured else None,
                mean_channel_removal_probability_effect=dict(zip(io23.RAW23,
                    np.mean([r['channel_removal_probability_effect'] for r in measured], axis=0).tolist(),
                    strict=True)) if measured else None)
    return result


def main(root, output):
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError('local rl_env CUDA required')
    p = read(root/'protocol.json')
    if (read(root/'completed.json')['status'] != 'COMPLETED'
            or not read(root/'independent_audit.json')['complete360']
            or p.get('feedback') != 'progressive' or p['readout'] != 'full'):
        raise ValueError('completed/audited progressive full-readout360 required')
    for name, digest in p['code_sha256'].items():
        if sha(io23.ROOT/name) != digest:
            raise ValueError('production code changed since run: '+name)
    for path, digest in ((Path(p['source_checkpoint']), p['source_sha256']),
                         (Path(p['source_memory']['path']), p['source_memory']['sha256'])):
        if sha(path) != digest:
            raise ValueError('changed source or memory checkpoint')
    order = read(root/'order.json')
    if (len(order) != 360 or len({r['shot'] for r in order}) != 360
            or sha(root/'order.json') != p['order_sha256'] or p['warmup'] != 20):
        raise ValueError('exact complete360 order with20 startup shots required')
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    base, _ = load_parent(Path(p['source_checkpoint']))
    model = LifecycleAdapter(base.cuda(), readout='full', feedback='progressive').cuda().eval()
    load_source_memory(model, Path(p['source_memory']['path']))
    model.restore(torch.load(root/'models'/'cold.pt', map_location='cpu', weights_only=True))
    model.requires_grad_(False)
    unchanged = {k: v.clone() for k, v in model.state_dict().items()}
    input_hash = read(root/'input_hashes.json')
    mapping = read(root/'input_map.json')
    if mapping['calibration_ids'] != [r['shot'] for r in order[:20]]:
        raise ValueError('mapping not from exact first20')
    output.mkdir(parents=True)
    plan = io23.ROOT/'docs/research/source_event_rebuild_20261023/炮内更新方向_事后诊断约定_20261026.md'
    fixed_files = ['protocol.json', 'completed.json', 'independent_audit.json', 'input_hashes.json',
                   'order.json', 'input_map.json', 'models/cold.pt']
    receipt = dict(scope='POSTHOC_ORACLE_DIAGNOSTIC_NOT_AN_ONLINE_POLICY',
        root=str(root.resolve()), input_sha256={f: sha(root/f) for f in fixed_files},
        plan_sha256=sha(plan), code_sha256={str(f.relative_to(io23.ROOT)): sha(f)
            for f in Path(__file__).parent.glob('*.py')},
        selection='cold peak: D lead>30ms; S all; secondary D 30<lead<=125ms',
        threshold=p['threshold'], probability_absolute_tolerance=1e-6,
        oracle_gradient_not_used_for_learning=True, source_and_cold_unchanged=True,
        cold_start20_remain_in_original360_score=True,
        physics_causality_established=False, independent_validation=False)
    _durable_json(output/'protocol.json', receipt)
    start = time.monotonic()
    rows, missing, audited = [], [], []
    max_error = 0.
    for index, item in enumerate(order, 1):
        if time.monotonic()-start > 1800:
            raise TimeoutError('declared30-minute diagnostic timeout')
        sid = item['shot']
        path = root/'shots'/f'{index:04d}_{sid}.npz'
        seal = read(root/'shots'/f'{index:04d}_{sid}_seal.json')
        if sha(path) != seal['sha256'] or sha(io23.path_for('EAST', sid)) != input_hash[str(sid)]:
            raise ValueError('changed raw input or sealed curve')
        if index <= 20:
            continue  # no fast updates existed; retained in original primary360 score
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        signal = transform({k: raw[k] for k in ('values', 'available')}, mapping)
        disruptive, down = io23.read_outcome('EAST', sid)  # explicitly posthoc only
        with np.load(path) as saved:
            if not np.array_equal(saved['times'], raw['times']):
                raise ValueError('raw and sealed time grid differs')
            selected = select_endpoints(saved['times'], saved['cold'], disruptive, down)
            roles = {role for s in selected for role in s['roles']}
            expected = {'primary', 'late125'} if disruptive else {'primary'}
            if roles != expected:
                missing.append(dict(shot=sid, missing_roles=sorted(expected-roles)))
            with torch.no_grad():
                means, complete = patch_signal(signal, base.center, base.scale)
                trace = model.memory.trace(means, complete, mode='memory')
                patches = {int(s['end']//8-1) for s in selected if s['end'] >= 8}
                states = selected_states(model.memory, means, complete, patches)
                for j, state in states.items():
                    if not torch.equal(state['key'] @ (state['after']-model.memory.prior), trace['delta'][j]):
                        raise ValueError(f'state reconstruction differs: {sid}/{j}')
            for s in selected:
                end = s['end']
                lo = end//32*32
                ends = np.arange(lo, min(lo+32, len(raw['times'])))
                offset = end-lo
                with torch.no_grad():
                    x, m = batch(signal, ends, base.center.device)
                    h, valid = model.tokens(x, m)
                    deltas = endpoint_corrections(trace['delta'], torch.as_tensor(ends, device=h.device))
                    cold = model.risk(h, valid, torch.zeros_like(deltas)).sigmoid().cpu().numpy()
                    intra = model.risk(h, valid, deltas).sigmoid().cpu().numpy()
                error = max(float(np.max(np.abs(cold-saved['cold'][ends]))),
                            float(np.max(np.abs(intra-saved['intra'][ends]))))
                max_error = max(max_error, error)
                if error > receipt['probability_absolute_tolerance']:
                    raise ValueError(f'sealed curve reproduction error {sid}: {error}')
                row = dict(shot=sid, stream_index=index, disruptive=disruptive,
                    end=end, roles=s['roles'], time=float(raw['times'][end]),
                    lead_ms=float(round((down-raw['times'][end])*1000, 9)) if disruptive else None,
                    cold_probability=float(saved['cold'][end]), intra_probability=float(saved['intra'][end]),
                    reproduction_error=error, selected_patch=end//8-1)
                if end >= 8:
                    row.update(measure(model, h, valid, deltas, offset, states[end//8-1], disruptive, p['threshold']))
                else:
                    row['not_measured_reason'] = 'before_first_complete_patch'
                rows.append(row)
        audited.append(sid)
        if len(audited) % 10 == 0:
            progress = dict(shots=len(audited), total=340, endpoints=len(rows),
                            seconds=round(time.monotonic()-start, 2), max_reproduction_error=max_error)
            _durable_json(output/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    if len(audited) != 340 or any(not torch.equal(v, model.state_dict()[k]) for k, v in unchanged.items()):
        raise ValueError('incomplete audit or model changed')
    for f, digest in receipt['input_sha256'].items():
        if sha(root/f) != digest:
            raise ValueError('parent artifact changed during audit')
    _durable_json(output/'per_endpoint.json', rows)
    done = dict(status='COMPLETED', audited_post_startup_shots=340, endpoints=len(rows),
        missing=missing, max_reproduction_error=max_error, summary=summarize(rows),
        seconds=time.monotonic()-start, original_run_unmodified=True,
        no_new_online_performance_claim=True, scientific_goal_achieved=False)
    _durable_json(output/'completed.json', done)
    print(json.dumps({k: v for k, v in done.items() if k != 'summary'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    main(args.run, args.output)
