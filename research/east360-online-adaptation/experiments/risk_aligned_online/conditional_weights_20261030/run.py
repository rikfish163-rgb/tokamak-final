"""Paired internal parameterization screen, not a full stream or online SGD."""
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
from ..context_vector_20261030.learning import ContextTarget, ordered_context, representation_step
from ..context_vector_20261030.scaling import fit_context_scale, transform_context
from ..context_vector_20261030.teaching import context_controls
from ..delayed_recurrent_memory_20261025.learning import batch
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..relational_attention_20261030.run import shot_summary
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform

PREVIOUS = io23.ROOT/'probe_outputs/relational_attention_cold20_v1'
ARMS = ('real', 'reverse', 'mean_feedback', 'constant')
KINDS = ('direct', 'representation')


def training_pools(records, fit_ids):
    pools = {0: {}, 1: {}}
    for i, row in enumerate(records):
        if row['shot'] in fit_ids and row['group'] in ('safe', 'danger'):
            label = int(row['group'] == 'danger')
            pools[label].setdefault(row['shot'], []).append(i)
    if any(not p for p in pools.values()):
        raise ValueError('both classes must have fitting shots')
    return pools


def draw_balanced(pools, rng):
    draws = []
    for label in (0, 1):
        ids = sorted(pools[label])
        for _ in range(12):
            sid = int(rng.choice(ids))
            draws.append(int(rng.choice(pools[label][sid])))
    return np.array(draws, dtype=np.int64)


def screen(groups, cosine, distinct):
    reasons = []
    for group, minimum in (('safe', 2), ('danger', 9)):
        g = groups['real'][group]
        if g['mean_delta_bce'] is None or g['mean_delta_bce'] >= -1e-6:
            reasons.append(group+': no mean improvement')
        if g['improved'] < minimum:
            reasons.append(group+': too few improved shots')
        for arm in ARMS[1:]:
            if g['mean_delta_bce'] is None or groups[arm][group]['mean_delta_bce'] is None or (
                    g['mean_delta_bce'] >= groups[arm][group]['mean_delta_bce']-1e-6):
                reasons.append(group+': no advantage over '+arm)
    if cosine is None or cosine >= .99 or distinct <= .5:
        reasons.append('feedback-specificity not demonstrated')
    return dict(passed=not reasons, reasons=reasons,
                real_reverse_median_cosine=cosine, different_later_fraction=distinct)


def state_for(net, controller, kind, data, context, *, outer=False):
    if kind == 'direct':
        return controller(context, data['support'])
    if kind != 'representation':
        raise ValueError('undeclared update family')
    state = data['x'].new_zeros(len(context), 8192, requires_grad=True)
    h, _, _ = net.first(data['x'], data['mask'], state, 'qk')
    target = controller(context, data['support'])
    return representation_step(h[:, -1], h[:, -1].detach(), target, state,
                               radius=.1, create_graph=outer)


def subset(data, index, device='cuda'):
    return {k: v[index].to(device) for k, v in data.items()}


def collect(net, signal, records):
    """Each row uses its own origin and matured target, never later input."""
    parts = {}
    for lo in range(0, len(records), 32):
        rows = records[lo:lo+32]
        origins = [r['origin'] for r in rows]
        if any(r['maturity'] != r['origin']+25 or r['query'] != r['origin']+50 for r in rows):
            raise ValueError('fixed clock changed')
        x, mask = batch(signal, origins, net.base.center.device)
        later_x, later_mask = batch(signal, [r['query'] for r in rows], x.device)
        truth = torch.tensor(np.stack([signal['values'][o+1:o+26] for o in origins]),
                             device=x.device, dtype=x.dtype)
        valid = torch.tensor(np.stack([signal['available'][o+1:o+26] for o in origins]),
                             device=x.device, dtype=torch.bool)
        valid &= torch.isfinite(truth) & mask[:, -1:] & torch.isfinite(x[:, -1:])
        valid[:, :, 10] = False
        with torch.no_grad():
            h, _, _ = net.first(x, mask, x.new_zeros(len(x), 8192), 'qk')
            center, scale = net.base.center, net.base.scale
            arrived = torch.asinh((torch.where(valid, truth, center)-center)/scale)
            anchor = torch.asinh((torch.where(valid, x[:, -1:], center)-center)/scale)
            change = torch.where(valid, arrived-anchor, 0.)
            context, support = ordered_context(h[:, -1], h[:, -2], change, valid)
        for k, v in dict(x=x, mask=mask, later_x=later_x, later_mask=later_mask,
                         context=context, support=support).items():
            parts.setdefault(k, []).append(v.detach().cpu())
    return {k: torch.cat(v) for k, v in parts.items()}


def fit(net, slow, data, scaled, records, spec, kind, output):
    from .learning import ConditionalWeights
    torch.manual_seed(131)
    controller = (ConditionalWeights(radius=.1) if kind == 'direct' else ContextTarget()).cuda()
    optimizer = torch.optim.Adam(controller.parameters(), lr=.001)
    rng = np.random.default_rng(131)
    pools = training_pools(records, spec['fit'])
    labels = torch.tensor([float(r['group'] == 'danger') for r in records])
    history, draws = [], []
    for iteration in range(256):
        take = draw_balanced(pools, rng)
        d = subset(data, take)
        step = state_for(net, controller, kind, d, scaled[take].cuda(), outer=True)
        z = net(d['later_x'], d['later_mask'], step, slow, 'qk', forecast=False)['logit']
        loss = F.binary_cross_entropy_with_logits(z, labels[take].cuda())
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        history.append(dict(step=iteration+1, loss=float(loss.detach()),
                            gradient_norm=float(grad_norm), mean_matrix_norm=float(step.detach().norm(dim=1).mean())))
        draws.append(take.tolist())
        if (iteration+1) % 64 == 0:
            print(dict(fold=spec['fold'], kind=kind, **history[-1]), flush=True)
    _durable_json(output/f"{kind}_fold{spec['fold']}_training.json", dict(history=history, draws=draws))
    return controller.eval()


def evaluate(net, slow, controller, kind, data, scaled, mean, index):
    parts = {}
    for lo in range(0, len(index), 32):
        take = index[lo:lo+32]
        d = subset(data, take)
        controls = context_controls(scaled[take].cuda(), mean.cuda())
        states = {a: state_for(net, controller, kind, d, c).detach() for a, c in controls.items()}
        raw_norms = {a: s.norm(dim=1) for a, s in states.items()}
        norm = raw_norms['real'][:, None]
        for arm in ARMS[1:]:
            magnitude = raw_norms[arm][:, None]
            states[arm] = torch.where(magnitude > 0, states[arm]/magnitude.clamp_min(1e-30)*norm, 0.)
        with torch.no_grad():
            result = dict(support=d['support'])
            zero = torch.zeros_like(states['real'])
            result['cold'] = net(d['later_x'], d['later_mask'], zero, slow, 'qk', forecast=False)['logit']
            for arm, state in states.items():
                result[arm] = net(d['later_x'], d['later_mask'], state, slow, 'qk', forecast=False)['logit']
                result[arm+'_norm'] = state.norm(dim=1)
                result[arm+'_raw_norm'] = raw_norms[arm]
                result[arm+'_cos_real'] = F.cosine_similarity(state, states['real'], dim=1)
            result['direction_nonzero'] = (raw_norms['real'] > 0) & (raw_norms['reverse'] > 0)
            if any(not torch.isfinite(v).all() for v in result.values()):
                raise FloatingPointError('nonfinite evaluation')
            for k, v in result.items():
                parts.setdefault(k, []).append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in parts.items()}


def execute(output):
    from .learning import ConditionalWeights
    if output.exists() or output.parent != io23.ROOT/'probe_outputs':
        raise ValueError('new worktree output directory required')
    original = json.loads((PREVIOUS/'protocol.json').read_text())
    completed = json.loads((PREVIOUS/'complete.json').read_text())
    for name in ('protocol.json', 'records.json', 'folds.json'):
        if sha(PREVIOUS/name) != completed['artifact_sha256'][name]:
            raise ValueError('fixed input record changed: '+name)
    records = json.loads((PREVIOUS/'records.json').read_text())
    folds = json.loads((PREVIOUS/'folds.json').read_text())
    reference = Path(original['reference'])
    for name, digest in original['reference_hashes'].items():
        if sha(reference/name) != digest:
            raise ValueError('parent dependency changed: '+name)
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('rl_env CUDA is required')
    base, _ = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda())
    frozen = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    slow = torch.load(reference/'models/cold.pt', map_location='cuda', weights_only=True)['slow']
    mapping = json.loads((reference/'input_map.json').read_text())
    hashes = json.loads((reference/'input_hashes.json').read_text())
    ids = original['warmup_ids']
    output.mkdir()
    files = {io23.ROOT/'CONDITIONAL_WEIGHTS_PLAN_20261030.md',
             io23.ROOT/'tests/test_conditional_weights.py', io23.ROOT/'tests/test_conditional_weights_run.py',
             *Path(__file__).parent.glob('*.py')}
    files.update(Path(m.__file__) for n, m in list(sys.modules.items())
                 if n.startswith('experiments.risk_aligned_online.') and getattr(m, '__file__', None))
    protocol = dict(scope='ENDED_COLD20_CONDITIONAL_PARENT_PARAMETERIZATION_SCREEN',
        reference=str(reference), reference_hashes=original['reference_hashes'],
        source_sha256=PARENT_SHA256, previous=str(PREVIOUS),
        input_record_hashes={n: sha(PREVIOUS/n) for n in ('protocol.json', 'records.json', 'folds.json')},
        warmup_ids=ids, conditional_parent_exposed_to_all20=True, independent_confirmation=False,
        full_stream_run=False, suffix_labels_read=0, hdf5_outcome_reads=0,
        steps=256, batch_size=24, radius=.1, seed=131, learning_rate=.001,
        direct_online_backward=False, matrix_site='layer0 Q/K all heads',
        parameter_counts=dict(direct=sum(p.numel() for p in ConditionalWeights().parameters()),
                              representation=sum(p.numel() for p in ContextTarget().parameters())),
        code_sha256={str(p.relative_to(io23.ROOT)): sha(p) for p in sorted(files)},
        torch_version=torch.__version__, cuda_device=torch.cuda.get_device_name())
    _durable_json(output/'protocol.json', protocol)
    _durable_json(output/'records.json', records)
    _durable_json(output/'folds.json', folds)
    with zipfile.ZipFile(output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.write(p, str(p.relative_to(io23.ROOT)))
    started = time.monotonic()
    shot_data = []
    for sid in ids:
        if sha(io23.path_for('EAST', sid)) != hashes[str(sid)]:
            raise ValueError('raw shot changed')
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        signal = transform({k: raw[k] for k in ('values', 'available')}, mapping)
        shot_data.append(collect(net, signal, [r for r in records if r['shot'] == sid]))
    if [r['shot'] for r in records] != [s for s in ids for r in records if r['shot'] == s]:
        raise ValueError('records no longer shot grouped')
    data = {k: torch.cat([s[k] for s in shot_data]) for k in shot_data[0]}
    torch.save(data, output/'observations.pt')
    all_pred = {kind: {} for kind in KINDS}
    train_summary, models = [], []
    for spec in folds:
        fit_index = [i for i, r in enumerate(records) if r['shot'] in spec['fit']]
        scale = fit_context_scale(data['context'][fit_index], [records[i]['shot'] for i in fit_index])
        scaled = transform_context(data['context'], scale)
        mean = torch.stack([scaled[[i for i in fit_index if records[i]['shot'] == sid]].mean(0)
                            for sid in spec['fit']]).mean(0)
        for kind in KINDS:
            controller = fit(net, slow, data, scaled, records, spec, kind, output)
            artifact = dict(state_dict={k: v.cpu() for k, v in controller.state_dict().items()},
                            scale=scale, mean=mean, spec=spec, kind=kind)
            model_name = f"{kind}_fold{spec['fold']}.pt"
            torch.save(artifact, output/model_name)
            models.append(model_name)
            fit_parts, fit_records = [], []
            for sid in spec['fit']+spec['held']:
                index = np.array([i for i, r in enumerate(records) if r['shot'] == sid])
                pred = evaluate(net, slow, controller, kind, data, scaled, mean, index)
                if sid in spec['held']:
                    all_pred[kind][sid] = pred
                    np.savez_compressed(output/f'{kind}_{sid}.npz', **pred)
                else:
                    fit_parts.append(pred)
                    fit_records.extend(records[i] for i in index)
            fit_pred = {k: np.concatenate([p[k] for p in fit_parts]) for k in fit_parts[0]}
            _, groups = shot_summary(fit_records, fit_pred, ARMS)
            train_summary.append(dict(fold=spec['fold'], kind=kind, groups=groups))
    summaries = {}
    for kind in KINDS:
        pred = {k: np.concatenate([all_pred[kind][sid][k] for sid in ids])
                for k in all_pred[kind][ids[0]]}
        rows, groups = shot_summary(records, pred, ARMS)
        valid = pred['direction_nonzero']
        cosine = float(np.median(pred['reverse_cos_real'][valid])) if valid.any() else None
        distinct = float((np.abs(pred['real'][valid]-pred['reverse'][valid]) > 1e-4).mean()) if valid.any() else 0.
        summaries[kind] = dict(groups=groups, screen=screen(groups, cosine, distinct))
        _durable_json(output/f'{kind}_per_shot.json', rows)
    if any(not torch.equal(frozen[k], v.cpu()) for k, v in net.state_dict().items()):
        raise AssertionError('parent changed')
    for name, digest in protocol['code_sha256'].items():
        if sha(io23.ROOT/name) != digest:
            raise AssertionError('code changed during run')
    if sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError('source checkpoint changed')
    _durable_json(output/'train_groups.json', train_summary)
    complete = dict(complete=True, opportunities=len(records), source_unchanged=True,
                    full_stream_run=False, summaries=summaries, elapsed_seconds=time.monotonic()-started,
                    artifacts={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    _durable_json(output/'complete.json', complete)
    print(json.dumps({k: v for k, v in complete.items() if k != 'artifacts'}, allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    execute(parser.parse_args().output)
