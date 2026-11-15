"""Frozen cold20 component screen; not a full-stream alarm evaluation."""
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..attention_basis_20261029.learning import MatrixAttention
from ..causal_input import apply_input_contract
from ..context_vector_20261030.teaching import stratified_folds
from ..delayed_recurrent_memory_20261025.learning import batch
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform

ARMS = ('real', 'reverse', 'unconditioned', 'mean_target', 'mean_direction')
PREVIOUS = io23.ROOT / 'probe_outputs/operational_transport_cold20_v2/protocol.json'


def schedule(length):
    """Fixed-clock opportunities; never selected from outcomes or risk peaks."""
    return [(o, o + 25, o + 50) for o in range(127, length - 50, 25)]


def classify_queries(times, flag, down):
    if not flag:
        return np.full(len(times), 'safe', dtype='<U16')
    if down is None or not np.isfinite(down):
        raise ValueError('valid postshot outcome required')
    lead = np.round((down - np.asarray(times)) * 1000, 9)
    return np.where(lead > 125, 'early', np.where(lead > 30, 'danger',
                    np.where(lead >= 0, 'late', 'postterminal')))


def shot_summary(records, pred, arms=ARMS):
    if any(len(pred[a]) != len(records) or not np.isfinite(pred[a]).all()
           for a in ('cold',) + arms):
        raise ValueError('finite aligned predictions required')
    rows, groups = [], {a: {} for a in arms}
    for sid in sorted({r['shot'] for r in records}):
        for group in ('safe', 'danger', 'early', 'late', 'postterminal'):
            use = np.array([r['shot'] == sid and r['group'] == group for r in records])
            if not use.any():
                continue
            before = np.asarray(pred['cold'][use], dtype=float)
            label = float(group in ('danger', 'late'))
            scored = group != 'postterminal'
            old_loss = np.logaddexp(0., (1 - 2 * label) * before)
            for arm in arms:
                after = np.asarray(pred[arm][use], dtype=float)
                new_loss = np.logaddexp(0., (1 - 2 * label) * after)
                rows.append(dict(shot=sid, group=group, arm=arm, windows=int(use.sum()),
                                 supported=int(pred['support'][use].sum()),
                                 before_bce=float(old_loss.mean()) if scored else None,
                                 delta_bce=float((new_loss-old_loss).mean()) if scored else None,
                                 mean_logit_change=float((after-before).mean())))
    for arm in arms:
        for group in ('safe', 'danger', 'early', 'late'):
            selected = [r for r in rows if r['arm'] == arm and r['group'] == group]
            delta = np.array([r['delta_bce'] for r in selected])
            groups[arm][group] = dict(shots=len(selected),
                supported_shots=sum(r['supported'] > 0 for r in selected),
                mean_delta_bce=float(delta.mean()) if len(delta) else None,
                min_delta_bce=float(delta.min()) if len(delta) else None,
                max_delta_bce=float(delta.max()) if len(delta) else None,
                improved=int((delta < -1e-6).sum()), worsened=int((delta > 1e-6).sum()))
    return rows, groups


def windows(signal, points):
    x, m = batch(signal, points, 'cpu')
    return dict(x=x, mask=m)


def chunks(data, device):
    for lo in range(0, len(data['x']), 32):
        yield slice(lo, lo+32), {k: v[lo:lo+32].to(device) for k, v in data.items()}


def targets(net, d, ref, mean=None):
    from .learning import normalized, relation_target, restrict_target
    z, mask = normalized(net, d['x'], d['mask'])
    pairs = {a: relation_target(z, mask, ref, mode=a) for a in ARMS[:3]}
    common = pairs['real'][1] & pairs['reverse'][1] & pairs['unconditioned'][1]
    if mean is not None:
        common = common & (mean > 0)[None]
    common = common & (common.sum(-1, keepdim=True) >= 4)
    out = {a: restrict_target(pairs[a][0], common) for a in ARMS[:3]}
    if mean is not None:
        out['mean_target'] = restrict_target(mean[None].expand(len(z), -1, -1), common)
    # restrict_target's contract is a target tensor; support is common for all.
    return out, common


def gradients(net, d, target, edge, slow, risk=False):
    from .learning import attention_kl
    state = d['x'].new_zeros(len(d['x']), 8192, requires_grad=True)
    _, packed, _ = net.first(d['x'], d['mask'], state, 'qk')
    loss = attention_kl(packed['query'], packed['key'], target, edge)
    g, = torch.autograd.grad(loss.sum(), state)
    if not torch.isfinite(g).all():
        raise FloatingPointError('nonfinite relational gradient')
    if not risk:
        return g.detach(), loss.detach()
    state2 = torch.zeros_like(state, requires_grad=True)
    score = net(d['x'], d['mask'], state2, slow, 'qk', forecast=False)['logit']
    rg, = torch.autograd.grad(score.sum(), state2)
    return g.detach(), loss.detach(), rg.detach()


def fixed_controls(net, data, ids, ref, slow):
    """No held-shot arrays or labels enter either fixed control."""
    device = net.base.center.device
    sums, counts = [], []
    for sid in ids:
        total = torch.zeros(4, 16, device=device)
        count = torch.zeros_like(total)
        for _, d in chunks(data[sid], device):
            ts, edge = targets(net, d, ref)
            total += ts['real'].sum(0)
            count += edge.sum(0)
        sums.append(torch.where(count > 0, total/count.clamp_min(1), 0.))
        counts.append(count > 0)
    count = torch.stack(counts).sum(0)
    mean = torch.where(count > 0, torch.stack(sums).sum(0)/count.clamp_min(1), 0.)
    gs = []
    for sid in ids:
        total = torch.zeros(8192, device=device)
        n = 0
        for _, d in chunks(data[sid], device):
            ts, edge = targets(net, d, ref, mean)
            g, _ = gradients(net, d, ts['real'], edge, slow)
            total += g.sum(0)
            n += len(g)
        if not n:
            raise ValueError('no training opportunities for fixed direction')
        gs.append(total/n)
    return mean.detach(), torch.stack(gs).mean(0).detach()


def evaluate(net, d, later, ref, mean, mean_gradient, slow):
    from .learning import attention_kl, bounded_step
    ts, edge = targets(net, d, ref, mean)
    support = edge.any(-1).any(-1)
    states, result = {}, {'support': support}
    real_g, _, rg = gradients(net, d, ts['real'], edge, slow, risk=True)
    for arm in ARMS[:4]:
        g, before = gradients(net, d, ts[arm], edge, slow)
        states[arm] = bounded_step(g)
        result[arm+'_loss_before'] = before
    states['mean_direction'] = bounded_step(mean_gradient[None]).expand(len(d['x']), -1) * support[:, None]
    with torch.no_grad():
        zero = torch.zeros_like(states['real'])
        result['cold'] = net(later['x'], later['mask'], zero, slow, 'qk', forecast=False)['logit']
        for arm, state in states.items():
            result[arm] = net(later['x'], later['mask'], state, slow, 'qk', forecast=False)['logit']
            result[arm+'_norm'] = state.norm(dim=1)
            if arm != 'mean_direction':
                _, packed, _ = net.first(d['x'], d['mask'], state, 'qk')
                result[arm+'_loss_after'] = attention_kl(packed['query'], packed['key'], ts[arm], edge)
        nonzero = (states['real'].norm(dim=1) > 0) & (states['reverse'].norm(dim=1) > 0)
        result['direction_nonzero'] = nonzero
        result['real_reverse_cosine'] = F.cosine_similarity(states['real'], states['reverse'], dim=1)
        result['real_risk_cosine'] = F.cosine_similarity(states['real'], rg, dim=1)
        result['real_gradient_norm'] = real_g.norm(dim=1)
        result['real_target'] = ts['real']
        result['reverse_target'] = ts['reverse']
        result['edge_mask'] = edge
    if any(not torch.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('nonfinite evaluated component')
    return {k: v.detach().cpu().numpy() for k, v in result.items()}


def execute(output):
    from .learning import fit_reference, normalized
    if output.exists() or output.parent != io23.ROOT/'probe_outputs':
        raise ValueError('new worktree probe output required')
    previous = json.loads(PREVIOUS.read_text())
    reference = io23.ROOT/previous['reference']
    for name, digest in previous['reference_hashes'].items():
        if sha(reference/name) != digest:
            raise ValueError('cold parent dependency changed: '+name)
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('rl_env CUDA required')
    base, _ = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda())
    frozen = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    slow = torch.load(reference/'models/cold.pt', map_location='cuda', weights_only=True)['slow']
    mapping = json.loads((reference/'input_map.json').read_text())
    hashes = json.loads((reference/'input_hashes.json').read_text())
    ids = previous['warmup_ids']
    output.mkdir()
    files = {io23.ROOT/'RELATIONAL_ATTENTION_PLAN_20261030.md',
             io23.ROOT/'tests/test_relational_attention_run.py',
             io23.ROOT/'tests/test_relational_attention.py', *Path(__file__).parent.glob('*.py')}
    files.update(Path(m.__file__) for n, m in list(sys.modules.items())
                 if n.startswith('experiments.risk_aligned_online.') and getattr(m, '__file__', None))
    protocol = dict(scope='ENDED_COLD20_CONDITIONAL_PARENT_COMPONENT_SCREEN',
        source_sha256=PARENT_SHA256, reference=str(reference),
        reference_hashes=previous['reference_hashes'], warmup_ids=ids,
        conditional_parent_exposed_to_all20=True, suffix_labels_read=0,
        independent_confirmation=False, full_stream_run=False,
        schedule='o=127+25k, m=o+25, query=m+25<N; no outcome-based endpoint selection',
        radius=.05, head=0, seed=131, arms=['cold', *ARMS],
        torch_version=torch.__version__, cuda_device=torch.cuda.get_device_name(),
        code_sha256={str(p.relative_to(io23.ROOT)): sha(p) for p in sorted(files)})
    _durable_json(output/'protocol.json', protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.write(p, str(p.relative_to(io23.ROOT)))
    started = time.monotonic()
    data, later, first, outcomes, records, offsets = {}, {}, {}, {}, [], {}
    for sid in ids:
        if sha(io23.path_for('EAST', sid)) != hashes[str(sid)]:
            raise ValueError('raw cold20 data changed')
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        signal = transform({k: raw[k] for k in ('values', 'available')}, mapping)
        fixed = schedule(len(signal['values']))
        if len(signal['values']) < 256 or not fixed:
            raise ValueError('insufficient reference/schedule support')
        data[sid] = windows(signal, [m for _, m, _ in fixed])
        later[sid] = windows(signal, [q for _, _, q in fixed])
        with torch.no_grad():
            z, m = normalized(net, torch.as_tensor(signal['values'][:256], device='cuda'),
                              torch.as_tensor(signal['available'][:256], device='cuda'))
        first[sid] = (z.cpu(), m.cpu())
        # These are ended developer cold shots. Labels select folds and later
        # metric groups only, never signal features, schedules or KL targets.
        outcomes[sid] = io23.read_outcome('EAST', sid)
        offsets[sid] = list(range(len(records), len(records)+len(fixed)))
        for o, m, q in fixed:
            records.append(dict(shot=sid, origin=o, maturity=m, query=q,
                                query_time=float(raw['times'][q])))
        print(dict(stage='collect', shot=sid, opportunities=len(fixed)), flush=True)
    specs = stratified_folds(ids, {s: v[0] for s, v in outcomes.items()})
    _durable_json(output/'folds.json', specs)
    all_pred = {}
    for spec in specs:
        ref_cpu = fit_reference([first[s] for s in spec['fit']])
        ref = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in ref_cpu.items()}
        mean, mean_g = fixed_controls(net, data, spec['fit'], ref, slow)
        torch.save(dict(reference=ref_cpu, mean_target=mean.cpu(), mean_gradient=mean_g.cpu(),
                        fit_ids=spec['fit'], held_ids=spec['held']), output/f"fold{spec['fold']}.pt")
        for sid in spec['held']:
            parts = {}
            for sl, d in chunks(data[sid], 'cuda'):
                q = {k: v[sl].to('cuda') for k, v in later[sid].items()}
                pred = evaluate(net, d, q, ref, mean, mean_g, slow)
                for k, v in pred.items():
                    parts.setdefault(k, []).append(v)
            merged = {k: np.concatenate(v) for k, v in parts.items()}
            np.savez_compressed(output/f'{sid}_predictions.npz', **merged)
            _durable_json(output/f'{sid}_seal.json', dict(sha256=sha(output/f'{sid}_predictions.npz')))
            all_pred[sid] = merged
            flag, down = outcomes[sid]
            groups = classify_queries(np.array([records[i]['query_time'] for i in offsets[sid]]), flag, down)
            for i, group in zip(offsets[sid], groups):
                records[i]['group'] = str(group)
            print(dict(stage='held', fold=spec['fold'], shot=sid,
                       supported=int(merged['support'].sum()), total=len(merged['cold'])), flush=True)
    pred = {k: np.concatenate([all_pred[s][k] for s in ids]) for k in all_pred[ids[0]]}
    rows, groups = shot_summary(records, pred)
    reasons = []
    for group, minimum, better in [('safe', 3, 2), ('danger', 10, None)]:
        g = groups['real'][group]
        if g['supported_shots'] < minimum:
            reasons.append(group+': insufficient supported shots')
        if g['mean_delta_bce'] is None or g['mean_delta_bce'] >= 0:
            reasons.append(group+': real did not improve cold')
        if g['improved'] < (better if better else (g['shots']+1)//2):
            reasons.append(group+': insufficient improved shots')
        for arm in ARMS[1:]:
            if (g['mean_delta_bce'] is None or groups[arm][group]['mean_delta_bce'] is None
                    or g['mean_delta_bce'] >= groups[arm][group]['mean_delta_bce']):
                reasons.append(group+': real did not beat '+arm)
    nonzero = pred['direction_nonzero']
    cosine = float(np.median(pred['real_reverse_cosine'][nonzero])) if nonzero.any() else None
    different = float((np.abs(pred['real'][nonzero]-pred['reverse'][nonzero]) > 1e-4).mean()) if nonzero.any() else None
    if cosine is None or cosine >= .99 or different < .5:
        reasons.append('directional/function specificity gate failed')
    _durable_json(output/'records.json', records)
    _durable_json(output/'per_shot.json', rows)
    if sha(DEFAULT_PARENT) != PARENT_SHA256 or any(not torch.equal(v.cpu(), frozen[k]) for k, v in net.state_dict().items()):
        raise AssertionError('source network changed')
    if any(p.grad is not None for p in net.parameters()):
        raise AssertionError('frozen weights accumulated gradients')
    for name, digest in protocol['code_sha256'].items():
        if sha(io23.ROOT/name) != digest:
            raise AssertionError('code changed during run: '+name)
    for name, digest in protocol['reference_hashes'].items():
        if sha(reference/name) != digest:
            raise AssertionError('parent changed during run: '+name)
    if any(sha(io23.path_for('EAST', sid)) != hashes[str(sid)] for sid in ids):
        raise AssertionError('cold20 raw data changed during run')
    result = dict(complete=True, full_stream_run=False, source_unchanged=True,
        suffix_labels_read=0, opportunities=len(records), groups=groups,
        real_reverse_median_cosine=cosine, distinct_future_logit_fraction=different,
        supported=int(pred['support'].sum()), matched_nonzero=int(nonzero.sum()),
        full_stream_gate=not reasons, reasons=reasons, seconds=time.monotonic()-started,
        artifact_sha256={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    _durable_json(output/'complete.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    execute(p.parse_args().output.absolute())
