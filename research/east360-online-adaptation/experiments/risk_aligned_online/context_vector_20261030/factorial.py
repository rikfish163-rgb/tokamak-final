"""Frozen 2x2 context-scale/radius screen; not a new full-stream result."""

import argparse
from pathlib import Path
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
from .prepare import ARMS, subset, summarize
from .scaling import calibrate_radius, fit_context_scale, transform_context
from .teaching import context_controls, teaching_mask

PREVIOUS = io23.ROOT/'probe_outputs/context_vector_cold20_v1'
VARIANTS = ('raw_fixed', 'scaled_fixed', 'raw_calibrated', 'scaled_calibrated')


def make_step(net, controller, data, radius, *, context=None, outer=False):
    state = data['x'].new_zeros(len(data['x']), 8192, requires_grad=True)
    h, _, _ = net.first(data['x'], data['available'], state, 'qk')
    delta = controller(data['context'] if context is None else context, data['support'])
    return representation_step(h[:, -1], h[:, -1].detach(), delta, state,
                               radius=radius, create_graph=outer)


def train(net, slow, data, indices, weights, labels, radius):
    torch.manual_seed(131)
    controller = ContextTarget().to(net.base.center.device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=.001)
    rng, history = np.random.default_rng(131), []
    for epoch in range(30):
        order, total = rng.permutation(len(indices)), 0.
        for start in range(0, len(order), 16):
            local = order[start:start+16]
            take = indices[local]
            d = subset(data, take)
            state = make_step(net, controller, d, radius, outer=True)
            z = net(d['later_x'], d['later_mask'], state, slow, 'qk', forecast=False)['logit']
            weighted = (F.binary_cross_entropy_with_logits(z, labels[take], reduction='none')*weights[local]).sum()
            optimizer.zero_grad()
            (weighted*len(indices)/len(local)).backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(weighted.detach())
        history.append(dict(epoch=epoch+1, weighted_loss_during_training=total))
    return controller.eval(), history


def evaluate(net, slow, controller, data, mean, radius):
    result = {key: [] for key in ['baseline'] + [a+'_'+k for a in ARMS[1:]
                                                for k in ('logit', 'norm', 'cos_real', 'cos_risk')]}
    for lo in range(0, len(data['x']), 16):
        d = subset(data, slice(lo, lo+16))
        contexts = context_controls(d['context'], mean)
        real = make_step(net, controller, d, radius).detach()
        state = d['x'].new_zeros(len(d['x']), 8192, requires_grad=True)
        z = net(d['x'], d['available'], state, slow, 'qk', forecast=False)['logit']
        risk, = torch.autograd.grad(z.sum(), state)
        magnitude = real.norm(dim=1, keepdim=True)
        states = {'real': real}
        for arm in ('reverse', 'mean_feedback', 'constant'):
            step = make_step(net, controller, d, radius, context=contexts[arm]).detach()
            norm = step.norm(dim=1, keepdim=True)
            states[arm] = torch.where(norm > 0, step/norm.clamp_min(1e-30)*magnitude, 0.)
        states['risk_axis'] = (real*risk).sum(1, keepdim=True).sign()*risk/risk.norm(dim=1, keepdim=True).clamp_min(1e-30)*magnitude
        with torch.no_grad():
            base = net(d['later_x'], d['later_mask'], torch.zeros_like(real), slow, 'qk', forecast=False)['logit']
            result['baseline'].append(base.cpu().numpy())
            for arm, step in states.items():
                z = net(d['later_x'], d['later_mask'], step, slow, 'qk', forecast=False)['logit']
                values = dict(logit=z, norm=step.norm(dim=1),
                              cos_real=F.cosine_similarity(real, step, dim=1),
                              cos_risk=F.cosine_similarity(risk, step, dim=1))
                for key, value in values.items():
                    result[arm+'_'+key].append(value.cpu().numpy())
    return {k: np.concatenate(v) for k, v in result.items()}


def execute(output):
    if output.exists() or output.parent != io23.ROOT/'probe_outputs':
        raise ValueError('new worktree output required')
    old = read(PREVIOUS/'protocol.json')
    records, folds = read(PREVIOUS/'records.json'), read(PREVIOUS/'folds.json')
    for name, digest in old['code_sha256'].items():
        if sha(io23.ROOT/name) != digest:
            raise AssertionError('v1 captured code changed')
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
    reference = Path(old['reference'])
    for name, digest in old['reference_hashes'].items():
        if sha(reference/name) != digest:
            raise AssertionError('cold reference changed')
    slow = torch.load(reference/'models/cold.pt', map_location='cuda', weights_only=True)['slow']
    mapping, hashes = read(reference/'input_map.json'), read(reference/'input_hashes.json')
    ids = old['warmup_ids']
    output.mkdir()
    files = {io23.ROOT/'CONTEXT_VECTOR_FACTORIAL_PLAN_20261030.md', *Path(__file__).parent.glob('*.py')}
    files.update(io23.ROOT/p for p in old['code_sha256'])
    protocol = dict(scope='ENDED_COLD20_SCALE_RADIUS_FACTORIAL', previous=str(PREVIOUS),
                    previous_complete_sha256=sha(PREVIOUS/'complete.json'), variants=list(VARIANTS),
                    reference=str(reference), reference_hashes=old['reference_hashes'],
                    warmup_ids=ids, seed=131, epochs=30, learning_rate=.001, batch_size=16,
                    conditional_parent_exposed_to_all20=True, independent_confirmation=False,
                    suffix_labels_read=0, full_stream_run=False,
                    code_sha256={str(p.relative_to(io23.ROOT)): sha(p) for p in sorted(files)})
    _durable_json(output/'protocol.json', protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            archive.write(p, str(p.relative_to(io23.ROOT)))
    observations, xx, aa = [], [], []
    started = time.monotonic()
    for sid in ids:
        if sha(io23.path_for('EAST', sid)) != hashes[str(sid)]:
            raise AssertionError('cold data changed')
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        sig = transform({k: raw[k] for k in ('values', 'available')}, mapping)
        for r in (r for r in records if r['shot'] == sid):
            observations.append(observation(net, sig, r['origin']))
            x, a = batch(sig, [r['query']], base.center.device)
            xx.append(x)
            aa.append(a)
    if [r['shot'] for r in records] != [s for s in ids for r in records if r['shot'] == s]:
        raise AssertionError('record grouping changed')
    data = {k: torch.cat([o[k] for o in observations]) for k in observations[0]}
    data.update(later_x=torch.cat(xx), later_mask=torch.cat(aa))
    context_cpu = data['context'].detach().cpu()
    sid_array = np.array([r['shot'] for r in records])
    flags = np.array([r['disruptive'] for r in records], int)
    eligible = teaching_mask(records)
    labels = base.center.new_tensor([r['point_label'] for r in records])
    diagnostic = read(PREVIOUS/'query_capacity_diagnostic.json')
    distances, wrong = np.ones(len(records)), np.zeros(len(records), bool)
    source = read(Path(old['previous'])/'protocol.json')
    threshold = float(source['threshold'])
    boundary = np.log(threshold/(1-threshold))
    for r in diagnostic['rows']:
        i = r['index']
        if r['shot'] != records[i]['shot'] or r['query'] != records[i]['query']:
            raise AssertionError('capacity record mismatch')
        distances[i] = r['local_linear_boundary_distance']
        wrong[i] = (r['cold_logit'] >= boundary) != bool(r['y'])
    source_norm = float(base.encoder.layers[0].self_attn.in_proj_weight[:128].norm())
    _durable_json(output/'inputs.json', dict(query_diagnostic_sha256=sha(PREVIOUS/'query_capacity_diagnostic.json'),
                                          records_sha256=sha(PREVIOUS/'records.json'), source_qk_norm=source_norm))
    results, radius_records = {}, []
    for variant in VARIANTS:
        directory = output/variant
        directory.mkdir()
        held_results = []
        for spec in folds + [dict(fold='all', fit=ids, held=[])]:
            fit_rows = np.isin(sid_array, spec['fit'])
            selected = np.flatnonzero(fit_rows & eligible)
            radius_info = calibrate_radius(distances, sid_array, fit_rows & eligible & wrong, source_norm)
            radius = radius_info['radius'] if variant.endswith('calibrated') else .1
            radius_records.append(dict(variant=variant, fold=spec['fold'], actual_radius=radius, **radius_info))
            scale = fit_context_scale(context_cpu[fit_rows], sid_array[fit_rows])
            d = dict(data)
            if variant.startswith('scaled'):
                d['context'] = transform_context(context_cpu, scale).to(base.center.device)
            weights = labels.new_tensor(shot_weights(sid_array[selected], flags[selected]))
            controller, history = train(net, slow, d, selected, weights, labels, radius)
            mean = torch.stack([d['context'][sid_array == sid].mean(0) for sid in spec['fit']]).mean(0)
            name = 'fold'+str(spec['fold'])
            model = dict(state=controller.state_dict(), scale=scale, mean_context=mean, radius=radius,
                         fit_ids=spec['fit'], protocol=protocol)
            torch.save(cpu_tree(model), directory/(name+'.pt'))
            _durable_json(directory/(name+'_training.json'), history)
            if variant == 'raw_fixed':
                previous_model = torch.load(PREVIOUS/(name+'.pt'), map_location='cuda', weights_only=True)
                if any(not torch.equal(v, previous_model['state'][k]) for k, v in controller.state_dict().items()):
                    raise AssertionError('raw/fixed model failed exact v1 replay')
            if spec['held']:
                index = np.flatnonzero(np.isin(sid_array, spec['held']))
                pred = evaluate(net, slow, controller, subset(d, index), mean, radius)
                if variant == 'raw_fixed':
                    with np.load(PREVIOUS/(name+'_held.npz'), allow_pickle=False) as archive:
                        if any(not np.array_equal(v, archive[k]) for k, v in pred.items()):
                            raise AssertionError('raw/fixed held predictions failed exact v1 replay')
                np.savez_compressed(directory/(name+'_held.npz'), index=index, **pred)
                _durable_json(directory/(name+'_seal.json'), dict(sha256=sha(directory/(name+'_held.npz'))))
                held_results.append((spec['fold'], index, pred))
            print(dict(variant=variant, fold=spec['fold'], radius=radius,
                       final_training_loss=history[-1]['weighted_loss_during_training']), flush=True)
        rows, result = summarize(records, held_results)
        _durable_json(directory/'per_shot.json', rows)
        _durable_json(directory/'complete.json', dict(result, policy_sha256=sha(directory/'foldall.pt')))
        results[variant] = result
    _durable_json(output/'radius_calibration.json', radius_records)
    if sha(DEFAULT_PARENT) != PARENT_SHA256 or any(not torch.equal(v.cpu(), frozen[k]) for k, v in net.state_dict().items()):
        raise AssertionError('source changed')
    for name, digest in protocol['code_sha256'].items():
        if sha(io23.ROOT/name) != digest:
            raise AssertionError('captured code changed during screen')
    promoted = next((v for v in VARIANTS[1:] if results[v]['full_stream_gate']), None)
    complete = dict(complete=True, raw_fixed_exact_replay=True, variants=results,
                    selected_for_full_stream=promoted, source_unchanged=True,
                    suffix_labels_read=0, full_stream_run=False, seconds=time.monotonic()-started)
    _durable_json(output/'complete.json', complete)
    print(complete, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    execute(parser.parse_args().output.absolute())
