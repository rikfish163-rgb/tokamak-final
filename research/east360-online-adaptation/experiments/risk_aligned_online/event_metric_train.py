"""SOURCE-only learned feedback metric; full VAL160 before EAST promotion.

No source classifier re-training and no J-TEXT online deployment. A real
prefix burn-in precedes up to four differentiable updates. TRAIN outcomes
select outer timely/safety targets only; inner updates read arrived signals.
"""
import argparse
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from . import simple_data, simple_train
from .event_dynamics_experiment import load_coordinates, PROBE
from .event_metric import EventMetricModel, source_step
from .fast_direction import episode
from .ffn_subspace_sequence import operational_loss
from .ffn_subspace_sequence_train import materialize
from .redesign_20261004.intra import predict_source_diagnostic
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent, load_source_train
from .simple_online import _durable_json, _hash


SPEC_ROOT = Path(__file__).parent / 'runs/ffn_subspace_sequence128_20261009'


def train_metric(model, optimizer, batch, *, mode, threshold_logit):
    if mode not in ('risk', 'signal') or len(batch) != 2:
        raise ValueError('balanced two-shot batch and declared objective required')
    if {bool(shot['is_disrupt']) for shot, _ in batch} != {False, True}:
        raise ValueError('one real safe and one disruptive source shot required')
    optimized = [p for group in optimizer.param_groups for p in group['params']]
    if len(optimized) != 1 or optimized[0] is not model.metric:
        raise ValueError('only the 25 loss-metric entries may be optimized')
    model.zero_grad(set_to_none=True)
    rows = []
    for shot, spec in batch:
        if shot['device'] != 'J-TEXT' or shot['shot'] != spec['shot']:
            raise ValueError('matching offline SOURCE spec required')
        fast = model.initial_fast(1).requires_grad_()
        # Preserve all earlier actual accepted steps; truncate only their
        # meta-gradient, not their effect on the model presented to training.
        for origin in range(0, spec['origins'][0], 100):
            e = episode(shot, origin, model.cfg, 'cuda')
            fast, _, _ = source_step(model, e, fast, create_graph=False)
            fast = fast.detach().requires_grad_()
        chain = materialize(shot, spec, model.cfg)
        after, before, labels, leads, losses, accepted = [], [], [], [], [], 0
        for e in chain['steps']:
            fast, receipt, signal = source_step(model, e, fast, create_graph=True)
            accepted += receipt['accepted']; losses.append(signal)
            n = len(e['segment_x'])
            after.append(model(e['segment_x'], e['segment_a'], fast.expand(n, -1))['logit'])
            before.append(model(e['segment_x'], e['segment_a'], model.initial_fast(n))['logit'].detach())
            labels.append(e['segment_y']); leads.append(e['segment_lead'])
        y, lead = torch.cat(labels), torch.cat(leads)
        risk = operational_loss(torch.cat(after), shot['is_disrupt'], y, lead, threshold_logit=threshold_logit)
        original_risk = operational_loss(torch.cat(before), shot['is_disrupt'], y, lead, threshold_logit=threshold_logit)
        state = torch.stack(losses).mean()
        loss = risk if mode == 'risk' else state
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite meta-objective')
        if loss.requires_grad:
            (loss / len(batch)).backward()
        rows.append(dict(shot=shot['shot'], disruptive=bool(shot['is_disrupt']), accepted=accepted,
            risk_before=float(original_risk), risk_after=float(risk.detach()),
            state_after=float(state.detach()), differentiable=loss.requires_grad))
    gradient = model.metric.grad
    if gradient is None:
        # All hard-acceptance branches can reject. That is a measured zero
        # learning step, not permission to bypass the runtime acceptance rule.
        return dict(rows=rows, optimizer_step=False, gradient_norm=0.)
    norm = torch.nn.utils.clip_grad_norm_([model.metric], 1., error_if_nonfinite=True)
    optimizer.step(); model.project_metric()
    return dict(rows=rows, optimizer_step=True, gradient_norm=float(norm))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=128)
    args = p.parse_args()
    if args.output.exists() or args.steps < 1: p.error('new output and positive fixed budget required')
    if not torch.cuda.is_available(): raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(4); torch.manual_seed(131)
    base, parent = load_parent(DEFAULT_PARENT)
    roles = simple_data.source_roles()
    old = json.loads((SPEC_ROOT / 'protocol.json').read_text())
    spec_path = SPEC_ROOT / 'source_chain_specs.json'
    if (old['parent_sha256'] != PARENT_SHA256 or old['train_ids'] != roles['train']
            or _hash(spec_path) != old['source_chain_specs_sha256']):
        raise ValueError('same-parent TRAIN-only fixed anchor manifest required')
    specs = json.loads(spec_path.read_text())
    if len(specs) != len(roles['train']): raise ValueError('source spec count mismatch')
    models = {name: EventMetricModel(base, *load_coordinates('event')).cuda().eval()
              for name in ('identity', 'signal', 'risk')}
    initial = {k: v.detach().cpu().clone() for k, v in models['identity'].state_dict().items() if k != 'metric'}
    optimizers = {k: torch.optim.Adam([models[k].metric], lr=.001) for k in ('signal', 'risk')}
    args.output.mkdir(parents=True)
    root = Path(__file__).parent
    files = [root / n for n in ('event_metric.py', 'event_metric_train.py', 'event_dynamics.py',
        'event_dynamics_experiment.py', 'event_projection_experiment.py', 'event_gradient_projection.py',
        'ended_decoder.py', 'persistent_ffn.py', 'ffn_intra_20261005/model.py',
        'fast_direction.py', 'ffn_subspace_sequence_train.py', 'ffn_subspace_sequence.py',
        'simple_data.py', 'simple_train.py', 'simple23.py', 'io23.py', 'fast_update.py',
        'redesign_20261004/intra.py', 'redesign_20261004/prepare_prior.py')]
    with zipfile.ZipFile(args.output / 'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path, str(path.relative_to(root)))
    protocol = dict(scope='DEVELOPMENT_ONLY_SOURCE_FEEDBACK_METRIC', parent_sha256=PARENT_SHA256,
        source_classifier_changed=False, east_access=False, test_access=False, jtext_online=False,
        train_ids=roles['train'], val_ids=roles['val'], steps=args.steps, seed=131, batch=2,
        optimized_parameters=25, original_forecast_head_changed=False, state_teacher_changed=False,
        source_probe_sha256=_hash(PROBE / 'probe_heads.pt'), annotation_teacher_split_overlap=True,
        metric_constraint='Frobenius norm sqrt(5); orientation and relative direction weights learned',
        inner=dict(stride=100, horizon=25, step=.05, radius=1., acceptance='same strict loss improvement'),
        outer_risk='max safe anchor / max original-positive lead>10 source anchor; fixed threshold margin',
        source_label_window='original max(25ms,AdvancedTime), unchanged',
        source_sequence='actual prefix burn-in; last up to four steps differentiated',
        outer_signal='same-budget decrease of transformed state error, not risk supervision',
        anchors_sha256=_hash(spec_path), threshold=float(parent['threshold']),
        optimizer='Adam lr=.001 clip=1; fixed last step, no validation checkpoint selection',
        source_gate='whole VAL160 TP>=static and FP<=static; strict J improvement over identity and signal',
        code_snapshot_sha256=_hash(args.output / 'code_snapshot.zip'))
    _durable_json(args.output / 'protocol.json', protocol)
    print(json.dumps(dict(stage='load_source_train', shots=1353)), flush=True)
    shots, ids = load_source_train()
    if ids != roles['train']: raise ValueError('source data order changed')
    pools = {False: [], True: []}
    for i, spec in enumerate(specs):
        if spec is not None:
            if spec['shot'] != ids[i]: raise ValueError('cached source identity changed')
            pools[bool(spec['is_disrupt'])].append(i)
    rng = np.random.default_rng(131); history = []; manifest = []
    boundary = math.log(parent['threshold'] / (1. - parent['threshold']))
    started = time.monotonic()
    for step in range(args.steps):
        chosen = [int(rng.choice(pools[label])) for label in (False, True)]
        batch = [(shots[i], specs[i]) for i in chosen]
        manifest.append([specs[i] for i in chosen])
        row = {name: train_metric(models[name], optimizers[name], batch,
                                 mode=name, threshold_logit=boundary) for name in ('signal', 'risk')}
        history.append(row)
        if (step + 1) % 16 == 0 or step + 1 == args.steps:
            _durable_json(args.output / 'training.json', dict(history=history, source_manifest=manifest))
            print(json.dumps(dict(stage='meta_metric', step=step+1, elapsed_s=time.monotonic()-started,
                **{n: dict(gradient_norm=r['gradient_norm'], optimizer_step=r['optimizer_step'],
                           risk_after=sum(s['risk_after'] for s in r['rows'])/2) for n, r in row.items()})), flush=True)
    for name, model in models.items():
        for k, v in initial.items(): torch.testing.assert_close(model.state_dict()[k].cpu(), v, rtol=0, atol=0)
        torch.save(dict(metric=model.metric.detach().cpu(), arm=name, parent_sha256=PARENT_SHA256),
                   args.output / f'{name}.pt')
    validation = simple_train.load_shots(roles['val'], simple_data.advanced_times())
    results = {}
    for name in ('static', 'identity', 'signal', 'risk'):
        model = models['identity' if name == 'static' else name].requires_grad_(False)
        folder = args.output / name; folder.mkdir()
        rows = []
        for i, shot in enumerate(validation):
            scores, audit = predict_source_diagnostic(model, shot, intra=name != 'static', device='cuda')
            path = folder / f'{shot["shot"]}.npz'
            np.savez_compressed(path, scores=scores, times=shot['times'], valid=shot['valid'],
                                is_disrupt=shot['is_disrupt'], down=shot['down'], shot=shot['shot'])
            _durable_json(folder / f'{shot["shot"]}.json', dict(curve_sha256=_hash(path), audit=audit))
            rows.append(dict(shot=shot['shot'], is_disrupt=shot['is_disrupt'], down=shot['down'],
                times=shot['times'][shot['valid']], scores=scores[shot['valid']]))
            if (i+1) % 40 == 0: print(json.dumps(dict(stage='VAL160', arm=name, shots=i+1)), flush=True)
        results[name] = simple_train.metrics(rows, parent['threshold'], lead=10.)
        _durable_json(args.output / 'validation.json', results)
        print(json.dumps(dict(stage='VAL160_completed', arm=name,
            **{k: v for k, v in results[name].items() if k != 'predictions'})), flush=True)
    r, s = results['risk'], results['static']
    reasons = []
    if r['tp'] < s['tp']: reasons.append('timely_TP_below_static')
    if r['fp'] > s['fp']: reasons.append('FP_above_static')
    for name in ('identity', 'signal'):
        if r['J'] <= results[name]['J']: reasons.append(f'not_strictly_better_than_{name}')
    if r['J'] <= s['J']: reasons.append('no_positive_source_J_gain')
    if _hash(DEFAULT_PARENT) != PARENT_SHA256: raise AssertionError('parent file changed')
    for model in models.values():
        for k, v in initial.items(): torch.testing.assert_close(model.state_dict()[k].cpu(), v, rtol=0, atol=0)
    _durable_json(args.output / 'completed.json', dict(source_gate_passed=not reasons, reasons=reasons,
        parent_and_frozen_tensors_unchanged=True, elapsed_s=time.monotonic()-started,
        matrices={n: m.metric.detach().cpu().tolist() for n, m in models.items()},
        independent_future_validation=False))
    print(json.dumps(dict(completed=True, source_gate_passed=not reasons, reasons=reasons)), flush=True)


if __name__ == '__main__': main()
