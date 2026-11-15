"""Anchored auxiliary target, with historical donor and reversed-feedback controls."""
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23
from ..attention_basis_20261029.learning import MatrixAttention
from ..conditional_weights_20261030.run import draw_balanced, subset, training_pools
from ..context_vector_20261030.learning import ContextTarget, representation_step
from ..context_vector_20261030.scaling import transform_context
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..relational_attention_20261030.run import shot_summary
from ..simple_online import _durable_json
from ..simple_train import sha

PREVIOUS = io23.ROOT/'probe_outputs/conditional_weights_cold20_v1'
ARMS = ('real', 'reverse', 'mean_feedback', 'donor')
KINDS = ('ordinary', 'anchored')


def feedback_controls(context, mean, donor):
    if (context.ndim != 2 or context.shape[1] != 358 or mean.shape != (358,)
            or donor.shape != context.shape):
        raise ValueError('context/donor Bx358 and mean 358 required')
    origin, feedback = context[:, :128], context[:, 128:]
    return dict(real=context.clone(),
        reverse=torch.cat((origin, feedback.reshape(-1, 5, 46).flip(1).flatten(1)), 1),
        mean_feedback=torch.cat((origin, mean[None, 128:].expand(len(context), -1)), 1),
        donor=torch.cat((origin, donor[:, 128:]), 1))


def candidate_gate(groups, cosine, distinct):
    reasons = []
    for group, minimum in (('safe', 2), ('danger', 9)):
        row = groups['real'][group]
        if row['mean_delta_bce'] is None or row['mean_delta_bce'] >= -1e-6:
            reasons.append(group+': no mean improvement')
        if row['improved'] < minimum:
            reasons.append(group+': too few improved shots')
        for arm in ('reverse', 'donor'):
            rival = groups[arm][group]['mean_delta_bce']
            if row['mean_delta_bce'] is None or rival is None or row['mean_delta_bce'] >= rival-1e-6:
                reasons.append(group+': no advantage over '+arm)
    if cosine is None or cosine >= .99 or distinct <= .5:
        reasons.append('feedback specificity failed')
    return dict(passed=not reasons, reasons=reasons,
                real_reverse_median_cosine=cosine, different_later_fraction=distinct,
                mean_feedback_not_used_as_utility_gate=True)


def state_for(net, controller, kind, data, context, mean, *, outer=False):
    state = data['x'].new_zeros(len(context), 8192, requires_grad=True)
    h, _, _ = net.first(data['x'], data['mask'], state, 'qk')
    if kind == 'anchored':
        target = controller(context, data['support'], mean)
    elif kind == 'ordinary':
        target = controller(context, data['support'])
    else:
        raise ValueError('undeclared family')
    return representation_step(h[:, -1], h[:, -1].detach(), target, state,
                               radius=.1, create_graph=outer)


def fit(net, slow, data, scaled, mean, records, spec, output):
    from .learning import AnchoredTarget
    torch.manual_seed(131)
    controller = AnchoredTarget().cuda()
    optimizer = torch.optim.Adam(controller.parameters(), lr=.001)
    pools, rng = training_pools(records, spec['fit']), np.random.default_rng(131)
    history, draws = [], []
    for iteration in range(256):
        take = draw_balanced(pools, rng)
        d = subset(data, take)
        state = state_for(net, controller, 'anchored', d, scaled[take].cuda(), mean.cuda(), outer=True)
        z = net(d['later_x'], d['later_mask'], state, slow, 'qk', forecast=False)['logit']
        labels = torch.tensor([float(records[i]['group'] == 'danger') for i in take], device='cuda')
        loss = F.binary_cross_entropy_with_logits(z, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite teaching loss')
        optimizer.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        draws.append(take.tolist())
        history.append(dict(step=iteration+1, loss=float(loss.detach()), gradient_norm=float(gn),
                            mean_matrix_norm=float(state.detach().norm(dim=1).mean())))
        if (iteration+1) % 64 == 0:
            print(dict(fold=spec['fold'], **history[-1]), flush=True)
    previous = json.loads((PREVIOUS/f"representation_fold{spec['fold']}_training.json").read_text())
    if draws != previous['draws']:
        raise AssertionError('not paired teaching draws')
    _durable_json(output/f"anchored_fold{spec['fold']}_training.json", dict(history=history, draws=draws))
    return controller.eval()


def evaluate(net, slow, controller, kind, data, scaled, mean, donor_indices, index):
    parts = {}
    for lo in range(0, len(index), 32):
        take = index[lo:lo+32]
        d = subset(data, take)
        contexts = feedback_controls(scaled[take].cuda(), mean.cuda(), scaled[donor_indices[take]].cuda())
        states = {arm: state_for(net, controller, kind, d, c, mean.cuda()).detach()
                  for arm, c in contexts.items()}
        raw = {arm: s.norm(dim=1) for arm, s in states.items()}
        for arm in ARMS[1:]:
            n = raw[arm][:, None]
            states[arm] = torch.where(n > 0, states[arm]/n.clamp_min(1e-30)*raw['real'][:, None], 0.)
        with torch.no_grad():
            zero = torch.zeros_like(states['real'])
            result = dict(cold=net(d['later_x'], d['later_mask'], zero, slow, 'qk', forecast=False)['logit'],
                          support=d['support'])
            for arm, state in states.items():
                result[arm] = net(d['later_x'], d['later_mask'], state, slow, 'qk', forecast=False)['logit']
                result[arm+'_norm'] = state.norm(dim=1)
                result[arm+'_raw_norm'] = raw[arm]
                result[arm+'_cos_real'] = F.cosine_similarity(state, states['real'], dim=1)
            result['direction_nonzero'] = (raw['real'] > 0) & (raw['reverse'] > 0)
            if kind == 'anchored' and (raw['mean_feedback'].any()
                    or not torch.equal(result['mean_feedback'], result['cold'])):
                raise AssertionError('reference feedback must leave model exactly unchanged')
            if any(not torch.isfinite(v).all() for v in result.values()):
                raise FloatingPointError('nonfinite evaluation')
            for key, value in result.items():
                parts.setdefault(key, []).append(value.cpu().numpy())
    return {k: np.concatenate(v) for k, v in parts.items()}


def execute(output):
    from .learning import AnchoredTarget, nearest_donor_indices
    if output.exists() or output.parent != io23.ROOT/'probe_outputs':
        raise ValueError('new output under worktree probe_outputs required')
    previous = json.loads((PREVIOUS/'complete.json').read_text())
    old_protocol = json.loads((PREVIOUS/'protocol.json').read_text())
    for name, digest in previous['artifacts'].items():
        if sha(PREVIOUS/name) != digest:
            raise ValueError('prior artifact changed: '+name)
    reference = Path(old_protocol['reference'])
    for name, digest in old_protocol['reference_hashes'].items():
        if sha(reference/name) != digest:
            raise ValueError('parent changed: '+name)
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
    data = torch.load(PREVIOUS/'observations.pt', map_location='cpu', weights_only=True)
    records = json.loads((PREVIOUS/'records.json').read_text())
    folds = json.loads((PREVIOUS/'folds.json').read_text())
    ids = old_protocol['warmup_ids']
    output.mkdir()
    files = {io23.ROOT/'ANCHORED_FEEDBACK_PLAN_20261030.md',
             io23.ROOT/'tests/test_anchored_feedback.py', io23.ROOT/'tests/test_anchored_feedback_run.py',
             *Path(__file__).parent.glob('*.py')}
    files.update(Path(m.__file__) for n, m in list(sys.modules.items())
                 if n.startswith('experiments.risk_aligned_online.') and getattr(m, '__file__', None))
    protocol = dict(scope='COLD20_ANCHORED_TARGET_CONDITIONAL_PARENT_DIAGNOSTIC', previous=str(PREVIOUS),
        parent=str(reference), parent_hashes=old_protocol['reference_hashes'], source_sha256=PARENT_SHA256,
        input_hashes={n: sha(PREVIOUS/n) for n in previous['artifacts']},
        fit_parent_exposed_to_held=True, independent_validation=False, full_stream=False,
        raw_signals_read=0, hdf5_outcomes_read=0, suffix_labels_read=0,
        steps=256, learning_rate=.001, batch_size=24, seed=131, radius=.1,
        controller_parameters=sum(p.numel() for p in AnchoredTarget().parameters()),
        actual_inner_gradient=True, update_site='layer0 Q/K',
        code_sha256={str(p.relative_to(io23.ROOT)): sha(p) for p in sorted(files)},
        torch=str(torch.__version__), cuda_device=torch.cuda.get_device_name())
    _durable_json(output/'protocol.json', protocol)
    _durable_json(output/'records.json', records)
    _durable_json(output/'folds.json', folds)
    with zipfile.ZipFile(output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            archive.write(p, str(p.relative_to(io23.ROOT)))
    started = time.monotonic()
    held = {kind: {} for kind in KINDS}
    fit_summary = []
    for spec in folds:
        prior = torch.load(PREVIOUS/f"representation_fold{spec['fold']}.pt", map_location='cpu', weights_only=True)
        if prior['spec'] != spec:
            raise ValueError('fold identity changed')
        mean, scale = prior['mean'], prior['scale']
        scaled = transform_context(data['context'], scale)
        donors = nearest_donor_indices(scaled, [r['shot'] for r in records], spec['fit'])
        _durable_json(output/f"fold{spec['fold']}_donors.json", donors.tolist())
        controllers = dict(anchored=fit(net, slow, data, scaled, mean, records, spec, output),
                           ordinary=ContextTarget().cuda())
        controllers['ordinary'].load_state_dict(prior['state_dict'], strict=True)
        controllers['ordinary'].eval()
        torch.save(dict(state_dict={k: v.detach().cpu() for k, v in controllers['anchored'].state_dict().items()},
                        mean=mean, scale=scale, spec=spec), output/f"anchored_fold{spec['fold']}.pt")
        for kind, controller in controllers.items():
            fit_preds, fit_records = [], []
            for sid in spec['fit']+spec['held']:
                index = np.array([i for i, r in enumerate(records) if r['shot'] == sid])
                pred = evaluate(net, slow, controller, kind, data, scaled, mean, donors, index)
                np.savez_compressed(output/f"{kind}_fold{spec['fold']}_{sid}.npz", **pred)
                if sid in spec['held']:
                    held[kind][sid] = pred
                    if kind == 'ordinary':
                        with np.load(PREVIOUS/f'representation_{sid}.npz', allow_pickle=False) as old:
                            for name in ('cold', 'real', 'reverse', 'mean_feedback', 'real_norm'):
                                if not np.array_equal(pred[name], old[name]):
                                    raise AssertionError('ordinary reference not exactly reproduced: '+name)
                else:
                    fit_preds.append(pred)
                    fit_records.extend(records[i] for i in index)
            packed = {k: np.concatenate([p[k] for p in fit_preds]) for k in fit_preds[0]}
            _, groups = shot_summary(fit_records, packed, ARMS)
            fit_summary.append(dict(fold=spec['fold'], kind=kind, groups=groups))
    summaries = {}
    for kind in KINDS:
        packed = {k: np.concatenate([held[kind][s][k] for s in ids]) for k in held[kind][ids[0]]}
        rows, groups = shot_summary(records, packed, ARMS)
        valid = packed['direction_nonzero']
        cosine = float(np.median(packed['reverse_cos_real'][valid])) if valid.any() else None
        distinct = float((np.abs(packed['real'][valid]-packed['reverse'][valid]) > 1e-4).mean()) if valid.any() else 0.
        summaries[kind] = dict(groups=groups, gate=candidate_gate(groups, cosine, distinct))
        _durable_json(output/f'{kind}_held_per_shot.json', rows)
    if any(not torch.equal(frozen[k], v.cpu()) for k, v in net.state_dict().items()):
        raise AssertionError('frozen parent changed')
    if sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError('source changed')
    for name, digest in protocol['code_sha256'].items():
        if sha(io23.ROOT/name) != digest:
            raise AssertionError('code changed during execution')
    _durable_json(output/'fit_groups.json', fit_summary)
    complete = dict(complete=True, full_stream=False, source_unchanged=True,
        opportunities=len(records), summaries=summaries, elapsed_seconds=time.monotonic()-started,
        artifacts={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    _durable_json(output/'complete.json', complete)
    print(json.dumps({k: v for k, v in complete.items() if k != 'artifacts'}, allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    execute(parser.parse_args().output)
