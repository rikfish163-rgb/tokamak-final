"""Bounded source-only aligned-copy pilot; point diagnostics, not stream scores."""

import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

import cta_copy_kernel_20261107 as kernel
import moving_peak_replay_20261105 as r
from experiments.risk_aligned_online.causal_input import apply_input_contract
from motion_metric_model_20261107 import scramble


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / 'probe_outputs/motion_metric_source_20261107_v3'
EAST = ROOT / 'probe_outputs/motion_metric_east20_20261107_v2'
PLAN = ROOT / 'CTA_COPY_PILOT_PROTOCOL_20261107.md'
KINDS = ('forecast_only', 'aligned')
GROUPS = ('safe', 'earlier', 'timely')
ARMS = ('static', 'real', 'time', 'channel', 'scalar')
SEED, STEPS, BATCH, PER_GROUP = 131, 192, 12, 48


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def choose_train_rows(rows):
    rng = np.random.default_rng(SEED)
    chosen = []
    for group in GROUPS:
        pool = [row for row in rows if row['group'] == group]
        if len(pool) < PER_GROUP:
            raise ValueError('insufficient predeclared TRAIN group')
        chosen.extend(pool[i] for i in rng.choice(len(pool), PER_GROUP, replace=False))
    return chosen


def load_windows(rows, role, device):
    roles = r.simple_data.source_roles()
    mapping = read(r.COLD / 'input_map.json') if role == 'east20' else None
    allowed = set(mapping['calibration_ids'] if mapping is not None else roles[role])
    if any(row['shot'] not in allowed for row in rows):
        raise ValueError('row outside declared cohort')
    if role == 'train' and (allowed & set(roles['val']) or allowed & set(roles['test'])):
        raise ValueError('source role overlap')
    arrays = {k: [] for k in ('x0', 'a0', 'xr', 'ar', 'xq', 'aq', 'truth', 'mask')}
    cache = {}
    for row in rows:
        sid = row['shot']
        if sid not in cache:
            if role == 'east20':
                raw = r.simple_data.read_shot('EAST', sid, rate_contract='nominal')
                signal = r.prepare_actual(raw, 21, mapping)
                signal = dict(signal, times=raw['times'])
            else:
                signal = apply_input_contract(r.simple_data.read_shot('J-TEXT', sid))
                signal['available'][:, 2] = False
            flag, down = r.io23.read_outcome('EAST' if role == 'east20' else 'J-TEXT', sid)
            cache[sid] = signal, flag, down
        signal, flag, down = cache[sid]
        origin, release, query = (row[k] for k in ('origin', 'release', 'query'))
        if release != origin + 26 or query != origin + 42 or origin < 127:
            raise ValueError('predeclared maturity clock changed')
        if bool(flag) != row['disruptive'] or abs(float(signal['times'][query])-row['query_time']) > 1e-9:
            raise ValueError('source identity or time changed')
        lead = round((down-float(signal['times'][query]))*1000, 9) if flag else None
        boundary = 125. if role == 'east20' else row['source_horizon_ms']
        deadline = 30. if role == 'east20' else 10.
        group = 'safe' if not flag else 'timely' if deadline < lead <= boundary+1e-6 else 'earlier'
        if group != row['group'] or row['label'] != int(group == 'timely') or (flag and lead <= deadline):
            raise ValueError('HDF5 time-label contract changed')
        for suffix, end in (('0', origin), ('r', release), ('q', query)):
            x, a = r.simple_data.window(signal, end, 128)
            arrays['x'+suffix].append(x)
            arrays['a'+suffix].append(a)
        arrays['truth'].append(signal['values'][origin+1:origin+26])
        mask = signal['available'][origin+1:origin+26].copy()
        mask[:, [2, 10]] = False
        arrays['mask'].append(mask)
    return {k: torch.as_tensor(np.stack(v), device=device) for k, v in arrays.items()}


def part(data, indices):
    return {k: v[indices] for k, v in data.items()}


def encoder_names(model):
    return [name for name, value in model.named_parameters()
            if value.requires_grad and not name.startswith('forecast_head.')]


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def unchanged(model, before):
    return all(torch.equal(value.detach().cpu(), before[k]) for k, value in model.state_dict().items())


@torch.no_grad()
def teacher_outputs(base, data):
    outputs = {}
    for suffix in ('0', 'r', 'q'):
        outputs[suffix] = kernel.forward_features(base, data['x'+suffix], data['a'+suffix])
    return outputs


def train_copies(base, data, rows):
    teacher = teacher_outputs(base, data)
    models = {kind: kernel.copy_for_training(base) for kind in KINDS}
    optimizers = {kind: torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-4)
                  for kind, m in models.items()}
    groups = {g: np.array([i for i, row in enumerate(rows) if row['group'] == g]) for g in GROUPS}
    rng, history = np.random.default_rng(SEED), []
    for step in range(STEPS):
        indices = np.concatenate([rng.choice(groups[g], BATCH//3, replace=True) for g in GROUPS])
        d = part(data, indices)
        for kind, model in models.items():
            optimizer = optimizers[kind]
            optimizer.zero_grad(set_to_none=True)
            at_origin = kernel.forward_features(model, d['x0'], d['a0'])
            signal = model.forecast_loss(at_origin['forecast'], d['truth'], d['mask'])
            at_query = kernel.forward_features(model, d['xq'], d['aq'])
            alignment = .5 * (F.mse_loss(at_origin['risk_features'], teacher['0']['risk_features'][indices])
                              + F.mse_loss(at_query['risk_features'], teacher['q']['risk_features'][indices]))
            loss = signal + (alignment if kind == 'aligned' else 0.)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite source-copy objective')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            history.append(dict(step=step+1, kind=kind, objective=float(loss.detach()),
                                signal=float(signal.detach()), alignment=float(alignment.detach()), gradient=float(gradient)))
        if (step+1) % 64 == 0:
            print('trained', step+1, history[-2:], flush=True)
    return models, history


def calibration(model, data):
    names = encoder_names(model)
    parameters = dict(model.named_parameters())
    budget = .01 * math.sqrt(sum(float(parameters[k].detach().double().square().sum()) for k in names))
    norms = []
    for i in range(len(data['x0'])):
        d = part(data, slice(i, i+1))
        info = kernel.inner_gradient(model, d['x0'], d['a0'], d['truth'], d['mask'])
        norms.append(float(info['gradient_norm']))
    reference = float(np.quantile(norms, .9))
    if not np.isfinite(reference) or reference <= 0 or budget <= 0:
        raise ValueError('finite source-only gradient calibration required')
    return dict(reference_norm=reference, budget=budget, gradient_norms=norms,
                encoder_parameters=sum(parameters[k].numel() for k in names))


def group_mean(values, rows, group):
    by_shot = defaultdict(list)
    for value, row in zip(values, rows):
        if row['group'] == group:
            by_shot[row['shot']].append(float(value))
    return float(np.mean([np.mean(v) for v in by_shot.values()]))


def evaluate(base, models, calibrations, data, rows, tau, out, role):
    teacher = teacher_outputs(base, data)
    baseline_path = REFERENCE/'val_predictions.npz' if role == 'source_val' else EAST/'predictions.npz'
    original = teacher['q']['logit'].cpu().numpy()
    with np.load(baseline_path, allow_pickle=False) as previous:
        static_error = float(np.max(np.abs(original-previous['static'])))
    if static_error > 5e-6:
        raise ValueError('teacher changed from previously sealed point baseline')
    values, records = {}, {}
    for kind, model in models.items():
        before = snapshot(model)
        with torch.no_grad():
            native = kernel.forward_features(model, data['xq'], data['aq'])
            release = kernel.forward_features(model, data['xr'], data['ar'])['logit']
        curves = {arm: [] for arm in ARMS}
        norms, gradients, proxy_before, proxy_after, unmatched = [], [], [], [], []
        for i in range(len(rows)):
            d = part(data, slice(i, i+1))
            real = kernel.inner_step(model, d['x0'], d['a0'], d['truth'], d['mask'],
                                     **{k: calibrations[kind][k] for k in ('reference_norm', 'budget')})
            with torch.no_grad():
                rq = kernel.forward_features(model, d['xq'], d['aq'], parameters=real['parameters'])
                rr = kernel.forward_features(model, d['xr'], d['ar'], parameters=real['parameters'])
                ro = kernel.forward_features(model, d['x0'], d['a0'], parameters=real['parameters'])
                curves['static'].append(float(native['logit'][i]))
                curves['real'].append(float(rq['logit'][0]))
                curves['scalar'].append(float(native['logit'][i]+rr['logit'][0]-release[i]))
                proxy_after.append(float(model.forecast_loss(ro['forecast'], d['truth'], d['mask'])))
            norms.append(float(real['update_norm']))
            gradients.append(float(real['gradient_norm']))
            proxy_before.append(float(real['loss']))
            target = torch.asinh((torch.where(d['mask'], d['truth'], model.center)-model.center)/model.scale)
            for mode in ('time', 'channel'):
                wrong_target = scramble(target, d['mask'], axis=mode)
                wrong_truth = wrong_target.sinh()*model.scale+model.center
                pair = kernel.paired_steps(model, d['x0'], d['a0'], d['truth'], wrong_truth, d['mask'],
                    **{k: calibrations[kind][k] for k in ('reference_norm', 'budget')})
                if abs(float(pair['real']['update_norm'])-norms[-1]) > 1e-6:
                    raise ValueError('control altered primary real update')
                unmatched.append(bool(pair.get('unmatched_control', False)))
                with torch.no_grad():
                    z = kernel.forward_features(model, d['xq'], d['aq'], parameters=pair['wrong']['parameters'])
                curves[mode].append(float(z['logit'][0]))
        if not unchanged(model, before):
            raise ValueError('point adaptation mutated its parent copy')
        values.update({kind+'_'+k: np.asarray(v, np.float64) for k, v in curves.items()})
        values[kind+'_update_norm'] = np.asarray(norms)
        records[kind] = dict(update_norm_median=float(np.median(norms)), update_norm_max=max(norms),
            gradient_norm_median=float(np.median(gradients)), unmatched_controls=sum(unmatched),
            proxy_before_mean=float(np.mean(proxy_before)), proxy_after_mean=float(np.mean(proxy_after)),
            static_logit_drift_mean=float(np.mean(np.abs(np.asarray(curves['static'])-original))),
            static_logit_drift_max=float(np.max(np.abs(np.asarray(curves['static'])-original))),
            static_cls_mse=float(F.mse_loss(native['risk_features'], teacher['q']['risk_features'])),
            adaptation_logit_change_median=float(np.median(np.abs(np.asarray(curves['real'])-curves['static']))))
        print('evaluated', role, kind, 'windows', len(rows), flush=True)
    values['teacher'] = original.astype(np.float64)
    with (out/(role+'_predictions.npz')).open('xb') as handle:
        np.savez_compressed(handle, **values)
    write(out/(role+'_sealed.json'), dict(status='SEALED_LOCAL_COPY_DIAGNOSTIC', rows=rows,
                                         teacher_parity_error=static_error, cohort_role=role))
    labels = np.asarray([row['label'] for row in rows])
    losses = {name: np.logaddexp(0., (1-2*labels)*(v-tau)) for name,v in values.items() if not name.endswith('_update_norm')}
    table = {}
    for kind in KINDS:
        table[kind] = {}
        for group in GROUPS:
            teacher_bce = group_mean(losses['teacher'], rows, group)
            static_bce = group_mean(losses[kind+'_static'], rows, group)
            table[kind][group] = dict(teacher_bce=teacher_bce, static_bce=static_bce,
                static_minus_teacher=static_bce-teacher_bce,
                adapted_minus_same_parent={arm:group_mean(losses[kind+'_'+arm]-losses[kind+'_static'], rows, group)
                                          for arm in ARMS if arm != 'static'})
    write(out/(role+'_result.json'), dict(status='POINT_DIAGNOSTIC_NOT_STREAM', table=table,
                                         records=records, teacher_parity_error=static_error))
    return table


def main(out):
    if out.exists():
        raise FileExistsError(out)
    if shutil.disk_usage(ROOT).free < 1024**3:
        raise RuntimeError('less than predeclared 1GiB disk reserve')
    start = time.monotonic()
    r.configure()
    torch.cuda.reset_peak_memory_stats()
    base, checkpoint = r.load_parent(r.DEFAULT_PARENT)
    base.cuda()
    source_before = snapshot(base)
    prior = read(REFERENCE/'protocol.json')
    train_rows = choose_train_rows(prior['fit_rows'])
    out.mkdir()
    write(out/'protocol.json', dict(status='FROZEN_COPY_PILOT_PROTOCOL', seed=SEED, steps=STEPS, batch=BATCH,
        rows=train_rows, kinds=KINDS, threshold=checkpoint['threshold'], source=str(r.DEFAULT_PARENT),
        source_state=r.state_digest(base), source_training_only=True, target_training=False,
        code={p.name:p.read_text() for p in (Path(__file__), Path(kernel.__file__), PLAN)}))
    data = load_windows(train_rows, 'train', base.center.device)
    models, history = train_copies(base, data, train_rows)
    calibrations = {kind:calibration(model, data) for kind,model in models.items()}
    if not unchanged(base, source_before) or any(p.grad is not None for p in base.parameters()):
        raise ValueError('qualified source changed')
    for kind, model in models.items():
        state = snapshot(model)
        for name in ('risk_head.weight', 'risk_head.bias', 'fast_origin', 'center', 'scale'):
            if not torch.equal(state[name], source_before[name]):
                raise ValueError('copy frozen head or normalization changed')
        torch.save(state, out/(kind+'.pt'))
    write(out/'fit.json', dict(status='FIT_COMPLETE', history=history, calibrations=calibrations,
                               seconds=time.monotonic()-start))
    tau = math.log(checkpoint['threshold']/(1-checkpoint['threshold']))
    tables = {}
    for role, path, load_role in (('source_val', REFERENCE/'val_sealed.json', 'val'),
                                  ('east20', EAST/'sealed.json', 'east20')):
        rows = read(path)['rows']
        data = load_windows(rows, load_role, base.center.device)
        tables[role] = evaluate(base, models, calibrations, data, rows, tau, out, role)
    if not unchanged(base, source_before):
        raise ValueError('qualified source changed during diagnosis')
    screen = {}
    for role, table in tables.items():
        candidate = table['aligned']
        def mean_delta(arm):
            return .5*sum(candidate[g]['adapted_minus_same_parent'][arm] for g in ('safe','timely'))
        screen[role] = (all(candidate[g]['adapted_minus_same_parent']['real'] < -1e-6 for g in ('safe','timely'))
            and candidate['earlier']['adapted_minus_same_parent']['real'] <= 1e-6
            and all(mean_delta('real') < mean_delta(k)-1e-6 for k in ('time','channel','scalar')))
    size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    write(out/'completed.json', dict(status='COMPLETE_LOCAL_COPY_PILOT', seconds=time.monotonic()-start,
        window_screens=screen, candidate='aligned', source_unchanged=True,
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(), output_bytes=size,
        full_stream_executed=False, independent_future_validation=False))
    print('complete', screen, 'seconds',time.monotonic()-start,'bytes',size,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args=parser.parse_args()
    main(args.out)
