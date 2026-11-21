"""Matched source-objective preparation and native-token EAST adaptation.

Source TEST is scored, never optimized. Each parent has its own static control;
the fixed alarm threshold, cold20 map and full360 order are common. Adaptation
uses the exact one-step H25 inner update used during source preparation.
"""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import moving_peak_replay_20261105 as r
from experiments.risk_aligned_online import simple_data, simple_train
from experiments.risk_aligned_online.fast_meta import temporary_step
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import PARENT_SHA256
from experiments.risk_aligned_online.simple23 import Config, Simple23
from run_layer0_query_memory_20261105 import raw_for
from run_rolling_safety_20261106 import cohort, recorded, require_parity
from verify_confirmation_raw_replay_20261105 import counts, pair, raw_truth, score


FIT = r.ROOT / 'probe_outputs/source_meta_objective_20261107_v1'
ARMS = ('static', 'intra', 'wrong_time', 'scalar')
PARENTS = ('original', 'before', 'after')
FIRST, STRIDE, BATCH = 153, 32, 128


def schedule(n):
    origins = np.arange(127, n - 26, STRIDE, dtype=np.int64)
    return origins, origins + 26


def native_curves(base, signal, *, warm=False):
    """H25 targets end at q-1; the new temporary state first predicts q."""
    base.eval()
    x, available = r.latent.make_views(signal, base.center.device)
    with torch.no_grad():
        static = torch.cat([base(x[lo:lo+BATCH], available[lo:lo+BATCH])['logit']
                            for lo in range(0, len(x), BATCH)])
    curves = {k: static.sigmoid().cpu().numpy().copy() for k in ARMS}
    origins, queries = schedule(len(x))
    info = dict(updates=0, state_norm_max=0., wrong_norm_error=0.)
    if warm or not len(origins):
        return curves, info
    values = torch.as_tensor(signal['values'], device=x.device)
    mask = torch.as_tensor(signal['available'], device=x.device)
    states, wrongs, anchors = [], [], []
    for lo in range(0, len(origins), BATCH):
        o, q = origins[lo:lo+BATCH], queries[lo:lo+BATCH]
        target = torch.as_tensor(o[:, None] + np.arange(1, 26)[None], device=x.device)
        e = dict(x=x[o], a=available[o], truth=values[target], mask=mask[target])
        fast, _ = temporary_step(base, e, create_graph=False)
        wrong, _ = temporary_step(base, dict(e, truth=e['truth'].flip(1),
                                            mask=e['mask'].flip(1)), create_graph=False)
        initial = base.initial_fast(len(o)).detach()
        delta, other = fast.detach() - initial, wrong.detach() - initial
        norm, other_norm = delta.norm(dim=1, keepdim=True), other.norm(dim=1, keepdim=True)
        if ((other_norm == 0) & (norm > 0)).any():
            raise ValueError('a nonzero real update requires a nonzero direction control')
        # Zero feedback gradient stays zero; nonzero controls retain direction
        # and receive the same norm, not a separately tuned learning rate.
        wrong = initial + other / torch.where(other_norm > 0, other_norm, 1.) * norm
        info['wrong_norm_error'] = max(info['wrong_norm_error'], float(
            ((wrong-initial).norm(dim=1)-norm[:, 0]).abs().max()))
        info['state_norm_max'] = max(info['state_norm_max'], float(norm.max()))
        states.append(fast.detach())
        wrongs.append(wrong)
        with torch.no_grad():
            anchors.append(base(x[q], available[q], fast.detach())['logit'] - static[q])
    fast, wrong, anchor = torch.cat(states), torch.cat(wrongs), torch.cat(anchors)
    with torch.no_grad():
        for lo in range(FIRST, len(x), BATCH):
            end = torch.arange(lo, min(lo+BATCH, len(x)), device=x.device)
            active = (end-FIRST) // STRIDE
            for name, state in (('intra', fast), ('wrong_time', wrong)):
                curves[name][lo:lo+len(end)] = base(x[end], available[end], state[active])['logit'].sigmoid().cpu().numpy()
            curves['scalar'][lo:lo+len(end)] = (static[end]+anchor[active]).sigmoid().cpu().numpy()
    if any(not np.isfinite(v).all() for v in curves.values()):
        raise FloatingPointError('nonfinite native-token risk curve')
    info['updates'] = len(origins)
    return curves, info


def load_models():
    original, parent = r.load_parent(r.DEFAULT_PARENT)
    models = dict(original=original.cuda())
    protocols, calibrated = {}, dict(original=float(parent['threshold']))
    for name, weight in (('before', 0.), ('after', .5)):
        folder = FIT / name
        training = r.read(folder / 'training.json')
        p = training['protocol']
        if (p['parent_sha256'] != PARENT_SHA256 or p['mode'] != 'aligned'
                or (p['steps'], p['batch_size'], p['lr'], p['seed']) != (512, 16, 1e-5, 131)
                or p['meta_risk_weight'] != weight or len(training['history']) != 512):
            raise ValueError('the exact paired source-only preparation is required')
        for row in training['history']:
            expected = ((1-weight)*row['risk_before'] + weight*row['risk_after']
                        + .2*row['signal_loss'] + .1*row['alignment_loss'])
            if row['risk_after_weight'] != weight or abs(row['objective']-expected) > 2e-7:
                raise ValueError('actual logged backward objective differs from the declared arm')
        # These are the two explicitly produced local checkpoints from this
        # fixed run, whose saved source metadata uses NumPy pickle objects.
        saved = torch.load(folder/'source.pt', map_location='cpu', weights_only=False)
        model = Simple23(Config(**saved['model_config']), saved['source_center'], saved['source_scale'])
        model.load_state_dict(saved['model'], strict=True)
        if (model.cfg != original.cfg or not torch.equal(model.center, original.center.cpu())
                or not torch.equal(model.scale, original.scale.cpu())
                or not torch.equal(model.fast_origin, original.fast_origin.cpu())):
            raise ValueError('source input, temporary origin or model architecture changed')
        models[name] = model.cuda().eval().requires_grad_(False)
        protocols[name] = p
        calibrated[name] = float(saved['threshold'])
    if protocols['before']['episode_sha256'] != protocols['after']['episode_sha256']:
        raise ValueError('paired training did not receive the same source episodes')
    return models, parent, protocols, calibrated


def source_test(models, calibrated, threshold):
    """Static source qualification only; this is not a J-TEXT online branch."""
    roles = simple_data.source_roles()
    shots = simple_train.load_shots(roles['test'], simple_data.advanced_times())
    report = {}
    for name, model in models.items():
        scored = simple_train.score(model, shots, 'cuda')
        fixed = simple_train.metrics(scored, threshold)
        own = simple_train.metrics(scored, calibrated[name])
        report[name] = dict(fixed_threshold=fixed, source_val_threshold=own,
            fixed_gate=fixed['tpr'] >= .9 and fixed['fpr'] < .2,
            own_val_gate=own['tpr'] >= .9 and own['fpr'] < .2)
    if (report['original']['fixed_threshold']['tp'], report['original']['fixed_threshold']['fp']) != (100, 13):
        raise ValueError('qualified original source TEST220 recount changed')
    return report


def replay(out, models, parent, preparations, source_report):
    start = time.monotonic()
    reference, cfg, archive, (real, stale, warm) = cohort('fixed360')
    if parent['threshold'] != r.THRESHOLD or warm != 20:
        raise ValueError('fixed threshold and original20 preparation required')
    mapping = r.read(r.COLD/'input_map.json')
    if mapping['calibration_ids'] != cfg['order'][:20]:
        raise ValueError('the original preparation20 input map is required')
    out.mkdir(parents=True, exist_ok=False)
    (out/'shots').mkdir()
    r.write_new(out/'protocol.json', dict(
        status='FROZEN_SOURCE_OBJECTIVE_NATIVE360', source=str(r.DEFAULT_PARENT),
        parent_sha256=PARENT_SHA256, preparations=preparations, order=cfg['order'], warm=warm,
        arms=ARMS, parents=list(models), threshold=r.THRESHOLD, first=FIRST, stride=STRIDE,
        temporary_site='original token FiLM; reset to original at each H25 feedback block',
        source_TEST_selection=False, jtext_online=False, target_label_fitting=False,
        inter_updates=0, baseline=str(reference), independent_confirmation=False,
        source_files={str(p): p.read_text() for p in (Path(__file__),
            r.ROOT/'experiments/risk_aligned_online/joint_meta.py',
            r.ROOT/'experiments/risk_aligned_online/fast_meta.py',
            r.ROOT/'experiments/risk_aligned_online/simple23.py')}))
    r.write_new(out/'source_test220.json', source_report)
    frozen = {name: {k: v.cpu().clone() for k, v in model.state_dict().items()}
              for name, model in models.items()}
    records, witnesses = [], []
    with patch.object(r.io23, 'read_outcome', side_effect=RuntimeError('EAST outcomes unavailable during prediction')):
        for rank, note in enumerate(archive['sealed'], 1):
            sid = note['shot']
            times, baseline, _, _ = recorded(reference, note, rank, real, stale)
            raw = raw_for(sid, cfg['raw_sha256'])
            np.testing.assert_array_equal(times, raw['times'])
            signal = r.prepare_actual(raw, rank, mapping)
            curves, info = {}, {}
            for name, model in models.items():
                result, info[name] = native_curves(model, signal, warm=rank <= warm)
                curves.update({f'{name}_{k}': v for k, v in result.items()})
            error = require_parity(curves['original_static'], baseline)
            if rank in (21, 181, 359):
                stop = min(len(times), max(FIRST+1, len(times)//2))
                partial = {k: signal[k][:stop] for k in ('values', 'available')}
                for name, model in models.items():
                    truncated, _ = native_curves(model, partial)
                    witnesses.append(dict(shot=sid, parent=name, points=stop,
                        max_error=max(require_parity(v, curves[f'{name}_{k}'][:stop])
                                      for k, v in truncated.items())))
            r.save_arrays(out/'shots'/f'{rank:04d}_{sid}.npz', **curves)
            records.append(dict(shot=sid, rank=rank, static_error=error, updates=info))
            if rank % 30 == 0:
                print('predicted', rank, 'of 360; seconds', round(time.monotonic()-start, 1), flush=True)
    for name, model in models.items():
        if any(not torch.equal(v.cpu(), frozen[name][k]) for k, v in model.state_dict().items()):
            raise ValueError('a persistent source model changed during intra-only replay')
    r.write_new(out/'completed.json', dict(status='SEALED_SOURCE_OBJECTIVE_NATIVE360',
        records=records, witnesses=witnesses, seconds=time.monotonic()-start,
        persistent_models_unchanged=True))
    print('sealed complete native360 comparison', flush=True)


def predict(out):
    r.configure()
    models, parent, preparations, calibrated = load_models()
    replay(out, models, parent, preparations, source_test(models, calibrated, r.THRESHOLD))


def recount(out):
    cfg, done = (r.read(out/f'{k}.json') for k in ('protocol', 'completed'))
    reference, original, archive, (real, stale, warm) = cohort('fixed360')
    if (done['status'] != 'SEALED_SOURCE_OBJECTIVE_NATIVE360' or cfg['order'] != original['order']
            or [v['shot'] for v in done['records']] != cfg['order']):
        raise ValueError('all360 sealed before the HDF5-only outcome recount')
    parents = cfg['parents']
    rows = {f'{p}_{a}': [] for p in parents for a in ARMS}
    for rank, note in enumerate(archive['sealed'], 1):
        sid = note['shot']
        times, _, _, _ = recorded(reference, note, rank, real, stale)
        truth, clock = raw_truth(r.io23.path_for('EAST', sid), sid)
        np.testing.assert_array_equal(times, clock)
        with np.load(out/'shots'/f'{rank:04d}_{sid}.npz', allow_pickle=False) as data:
            for key in rows:
                rows[key].append(score(times, data[key], truth, cfg['threshold']))
    metrics = {k: counts(v) for k, v in rows.items()}
    if (metrics['original_static']['tp'], metrics['original_static']['fp']) != (116, 29):
        raise ValueError('original source/static360 reproduction changed')
    for value in metrics.values():
        value['fpr_cap_met'] = {str(cap): value['fpr'] <= cap for cap in (.10, .15, .20)}
    comparisons = {p: {a: pair(rows[f'{p}_{a}'], rows[f'{p}_static']) for a in ARMS[1:]}
                   for p in parents}
    operational = {p: metrics[f'{p}_intra']['fpr'] <= .20
                     and metrics[f'{p}_intra']['tp'] > metrics[f'{p}_static']['tp'] for p in parents}
    r.write_new(out/'result.json', dict(status='HDF5_SOURCE_OBJECTIVE_NATIVE360',
        metrics=metrics, paired_own_static=comparisons, independent_intra_gain=operational,
        prefix20={k: counts(v[:20]) for k, v in rows.items()},
        suffix340={k: counts(v[20:]) for k, v in rows.items()}, rows=rows,
        source_qualification=r.read(out/'source_test220.json'), independent_confirmation=False))
    print({k: (v['tp'], v['fp'], v['tp_30_to_3000']) for k, v in metrics.items()}, flush=True)
    print('independent intra gain', operational, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('predict', 'score'))
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    predict(args.out) if args.phase == 'predict' else recount(args.out)
