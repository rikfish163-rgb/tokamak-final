"""Teach and screen representation targets using ended cold20 only."""

import argparse
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
from ..conditional_forecast_20261023.learning import shot_weights
from ..conditional_likelihood_20261029.run import read
from ..delayed_recurrent_memory_20261025.learning import batch
from ..operational_transport_20261030.gradients import cpu_tree
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import ContextTarget, representation_step
from .observation import observation
from .teaching import context_controls, stratified_folds, teaching_mask

PREPARED = io23.ROOT/'probe_outputs/operational_transport_cold20_v2'
ARMS = ('cold', 'real', 'reverse', 'mean_feedback', 'constant', 'risk_axis')


def adapted_state(net, controller, data, *, context=None, outer=False):
    state = data['x'].new_zeros(len(data['x']), 8192, requires_grad=True)
    h, _, _ = net.first(data['x'], data['available'], state, 'qk')
    delta = controller(data['context'] if context is None else context, data['support'])
    return representation_step(h[:, -1], h[:, -1].detach(), delta, state, create_graph=outer)


def subset(data, index):
    return {k: v[index] for k, v in data.items()}


def fit(net, slow, data, index, weights, label, output, name):
    torch.manual_seed(131)
    controller = ContextTarget().to(net.base.center.device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=.001)
    rng = np.random.default_rng(131)
    history = []
    for epoch in range(30):
        order = rng.permutation(len(index))
        total, norms = 0., []
        for start in range(0, len(order), 16):
            local = order[start:start+16]
            take = index[local]
            sampled = subset(data, take)
            step = adapted_state(net, controller, sampled, outer=True)
            z = net(sampled['later_x'], sampled['later_mask'], step, slow, 'qk', forecast=False)['logit']
            loss = F.binary_cross_entropy_with_logits(z, label[take], reduction='none')
            weighted = (loss*weights[local]).sum()
            objective = weighted*len(index)/len(local)
            optimizer.zero_grad()
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(weighted.detach())
            norms.append(float(norm))
        row = dict(epoch=epoch+1, weighted_loss_during_training=total,
                   max_controller_gradient_norm=max(norms))
        history.append(row)
        if epoch in (0, 9, 19, 29):
            print(dict(stage='teach', fit=name, **row), flush=True)
    _durable_json(output/(name+'_training.json'), history)
    return controller.eval()


def evaluate(net, slow, controller, data, mean):
    results = {key: [] for key in ['baseline'] + [a+'_'+k for a in ARMS[1:]
                                                 for k in ('logit', 'norm', 'cos_real', 'cos_risk')]}
    for start in range(0, len(data['x']), 16):
        d = subset(data, slice(start, start+16))
        controls = context_controls(d['context'], mean)
        real = adapted_state(net, controller, d).detach()
        risk_state = d['x'].new_zeros(len(d['x']), 8192, requires_grad=True)
        origin_risk = net(d['x'], d['available'], risk_state, slow, 'qk', forecast=False)['logit']
        risk, = torch.autograd.grad(origin_risk.sum(), risk_state)
        norm = real.norm(dim=1, keepdim=True)
        states = {'real': real}
        for name in ('reverse', 'mean_feedback', 'constant'):
            step = adapted_state(net, controller, d, context=controls[name]).detach()
            magnitude = step.norm(dim=1, keepdim=True)
            states[name] = torch.where(magnitude > 0, step/magnitude.clamp_min(1e-30)*norm, 0.)
        sign = (real*risk).sum(1, keepdim=True).sign()
        states['risk_axis'] = sign*risk/risk.norm(dim=1, keepdim=True).clamp_min(1e-30)*norm
        with torch.no_grad():
            base = net(d['later_x'], d['later_mask'], torch.zeros_like(real), slow, 'qk', forecast=False)['logit']
            results['baseline'].append(base.cpu().numpy())
            for name, step in states.items():
                z = net(d['later_x'], d['later_mask'], step, slow, 'qk', forecast=False)['logit']
                for key, value in dict(logit=z, norm=step.norm(dim=1),
                                       cos_real=F.cosine_similarity(real, step, dim=1),
                                       cos_risk=F.cosine_similarity(risk, step, dim=1)).items():
                    results[name+'_'+key].append(value.cpu().numpy())
    return {k: np.concatenate(v) for k, v in results.items()}


def summarize(records, held_results):
    rows = []
    for fold, index, pred in held_results:
        for sid in sorted({records[i]['shot'] for i in index}):
            for group in ('safe', 'danger_timely', 'early_disruptive'):
                use = np.array([records[i]['shot'] == sid and
                                ('danger_timely' if records[i]['point_label'] else 'early_disruptive'
                                 if records[i]['disruptive'] else 'safe') == group for i in index])
                if not use.any():
                    continue
                before = pred['baseline'][use].astype(float)
                label = float(group == 'danger_timely')
                for arm in ARMS[1:]:
                    after = pred[arm+'_logit'][use].astype(float)
                    delta = np.logaddexp(0., after)-label*after - (np.logaddexp(0., before)-label*before)
                    rows.append(dict(fold=fold, shot=sid, group=group, arm=arm, windows=int(use.sum()),
                                     mean_logit_change=float((after-before).mean()),
                                     before_bce=None if group == 'early_disruptive' else float((np.logaddexp(0., before)-label*before).mean()),
                                     delta_bce=None if group == 'early_disruptive' else float(delta.mean())))
    groups = {}
    for arm in ARMS[1:]:
        groups[arm] = {}
        for group in ('safe', 'danger_timely'):
            selected = [r for r in rows if r['arm'] == arm and r['group'] == group]
            delta = np.array([r['delta_bce'] for r in selected])
            groups[arm][group] = dict(shots=len(selected), mean_delta_bce=float(delta.mean()),
                                      improved=int((delta < 0).sum()), worsened=int((delta > 0).sum()))
    reasons = []
    for group in ('safe', 'danger_timely'):
        real = groups['real'][group]
        if (real['mean_delta_bce'] >= 0 or 2*real['improved'] < real['shots']
                or any(real['mean_delta_bce'] >= groups[arm][group]['mean_delta_bce']
                       for arm in ('reverse', 'mean_feedback', 'constant'))):
            reasons.append(group+' did not pass the prespecified conditional screen')
    return rows, dict(groups=groups, full_stream_gate=not reasons, reasons=reasons)


def execute(output):
    if output.exists() or output.parent != io23.ROOT/'probe_outputs':
        raise ValueError('new worktree probe output required')
    previous = read(PREPARED/'protocol.json')
    completion = read(PREPARED/'complete.json')
    policy = torch.load(PREPARED/'policy.pt', map_location='cpu', weights_only=True)
    if sha(PREPARED/'policy.pt') != completion['policy_sha256'] or sha(PREPARED/'pairs.pt') != policy['pairs_sha256']:
        raise ValueError('previous cold20 teaching records changed')
    records = torch.load(PREPARED/'pairs.pt', map_location='cpu', weights_only=True)['records']
    ids = previous['warmup_ids']
    if {r['shot'] for r in records} != set(ids):
        raise ValueError('non-cold20 shot in teaching records')
    reference = Path(previous['reference'])
    for name, digest in previous['reference_hashes'].items():
        if sha(reference/name) != digest:
            raise ValueError('reference changed: '+name)
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
    mapping, hashes = read(reference/'input_map.json'), read(reference/'input_hashes.json')
    output.mkdir()
    plan = io23.ROOT/'CONTEXT_VECTOR_PLAN_20261030.md'
    files = {plan, *Path(__file__).parent.glob('*.py')}
    files.update(Path(m.__file__) for n, m in list(sys.modules.items())
                 if n.startswith('experiments.risk_aligned_online.') and getattr(m, '__file__', None))
    protocol = dict(scope='ENDED_COLD20_CONDITIONAL_PARENT_COMPONENT_SCREEN', source_sha256=PARENT_SHA256,
                    previous=str(PREPARED), pairs_sha256=sha(PREPARED/'pairs.pt'), reference=str(reference),
                    reference_hashes=previous['reference_hashes'], warmup_ids=ids,
                    conditional_parent_exposed_to_all20=True, suffix_labels_read=0,
                    epochs=30, batch_size=16, seed=131, optimizer='Adam', learning_rate=.001,
                    inner_learning_rate=.1, qk_radius=.1, final_epoch_only=True,
                    independent_confirmation=False, full_stream_run=False,
                    code_sha256={str(p.relative_to(io23.ROOT)): sha(p) for p in sorted(files)})
    _durable_json(output/'protocol.json', protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            archive.write(p, str(p.relative_to(io23.ROOT)))
    started = time.monotonic()
    observations, later_x, later_a, outcomes = [], [], [], {}
    for sid in ids:
        if sha(io23.path_for('EAST', sid)) != hashes[str(sid)]:
            raise ValueError('cold data changed')
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        flag, down = io23.read_outcome('EAST', sid)
        outcomes[sid] = bool(flag)
        sig = transform({k: raw[k] for k in ('values', 'available')}, mapping)
        for r in (row for row in records if row['shot'] == sid):
            if not (r['maturity'] == r['origin']+25 < r['query'] <= r['origin']+50):
                raise AssertionError('teaching time order changed')
            lead = round(float((down-raw['times'][r['query']])*1000), 9) if flag else None
            if flag and (lead <= 30 or r['point_label'] != float(lead <= 125)):
                raise AssertionError('historical target label mismatch')
            observations.append(observation(net, sig, r['origin']))
            x, mask = batch(sig, [r['query']], base.center.device)
            later_x.append(x)
            later_a.append(mask)
        print(dict(stage='collect', shot=sid, examples=len(observations)), flush=True)
    if [r['shot'] for r in records] != [s for s in ids for r in records if r['shot'] == s]:
        raise AssertionError('records must be in original grouped shot order')
    data = {k: torch.cat([o[k] for o in observations]) for k in observations[0]}
    data.update(later_x=torch.cat(later_x), later_mask=torch.cat(later_a))
    labels = base.center.new_tensor([r['point_label'] for r in records])
    sid_array = np.array([r['shot'] for r in records])
    flags = np.array([r['disruptive'] for r in records], int)
    eligible = teaching_mask(records)
    specs = stratified_folds(ids, outcomes)
    _durable_json(output/'folds.json', specs)
    held_results = []
    for spec in specs + [dict(fold='all', fit=ids, held=[])]:
        fit_rows = np.isin(sid_array, spec['fit'])
        selected = np.flatnonzero(fit_rows & eligible)
        weights = labels.new_tensor(shot_weights(sid_array[selected], flags[selected]))
        name = 'fold'+str(spec['fold'])
        controller = fit(net, slow, data, selected, weights, labels, output, name)
        mean = torch.stack([data['context'][sid_array == sid].mean(0) for sid in spec['fit']]).mean(0)
        torch.save(cpu_tree(dict(state=controller.state_dict(), mean_context=mean,
                                 fit_ids=spec['fit'], protocol=protocol)), output/(name+'.pt'))
        if spec['held']:
            index = np.flatnonzero(np.isin(sid_array, spec['held']))
            pred = evaluate(net, slow, controller, subset(data, index), mean)
            np.savez_compressed(output/(name+'_held.npz'), index=index, **pred)
            _durable_json(output/(name+'_seal.json'), dict(sha256=sha(output/(name+'_held.npz'))))
            held_results.append((spec['fold'], index, pred))
    rows, result = summarize(records, held_results)
    _durable_json(output/'per_shot.json', rows)
    _durable_json(output/'records.json', records)
    if sha(DEFAULT_PARENT) != PARENT_SHA256 or any(not torch.equal(v.cpu(), frozen[k]) for k, v in net.state_dict().items()):
        raise AssertionError('source parameters changed')
    if any(p.grad is not None for p in net.parameters()):
        raise AssertionError('frozen network received a parameter gradient')
    for name, digest in protocol['code_sha256'].items():
        if sha(io23.ROOT/name) != digest:
            raise AssertionError('code changed during experiment: '+name)
    result.update(complete=True, full_stream_run=False, source_unchanged=True, suffix_labels_read=0,
                  examples=len(records), teaching_examples=int(eligible.sum()),
                  safe_teaching=int((eligible & (flags == 0)).sum()),
                  danger_teaching=int((eligible & (flags == 1)).sum()),
                  policy_sha256=sha(output/'foldall.pt'), seconds=time.monotonic()-started)
    _durable_json(output/'complete.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    execute(parser.parse_args().output.absolute())
