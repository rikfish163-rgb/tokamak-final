"""First32 EAST finite sensitivity probe, NOT a learned online policy.

Compare always-negative/hold/always-positive causal parameter actions. Outcome
oracle chooses among these three only after all traces are sealed. It is not
an upper bound for every possible signed policy or a deployable result.
"""
import argparse
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .. import simple_data, simple_train
from ..simple_online import _durable_json, _hash, seal_predictions
from ..redesign_20261004.intra import _batch_windows
from ..redesign_20261004.prepare_prior import (
    DEFAULT_PARENT, PARENT_SHA256, load_parent, classifier_state, assert_classifier_unchanged,
)
from ..ffn_intra_20261005.model import FfnModel


def fixed_sign_curve(available, sign):
    """q used for each emitted risk; only prior complete segments affect it."""
    available = np.asarray(available)
    if available.ndim != 2 or available.shape[1] != 23 or available.dtype != bool:
        raise ValueError('boolean [N,23] observations required')
    if type(sign) is not int or sign not in (-1, 0, 1):
        raise ValueError('fixed signed action -1/0/+1 required')
    curve = np.empty(len(available), np.float64); q = 0.
    for end in range(len(available)):
        curve[end] = q
        remainder = end % 26
        if remainder == 5 and available[end-4:end+1].any():
            q = float(np.clip(q + .02*sign, -1., 1.))
        elif remainder == 25 and available[end-19:end+1].any():
            q = float(np.clip(q + .08*sign, -1., 1.))
    return curve


def choose_historical_action(costs):
    if set(costs) != {-1, 0, 1} or any(e not in (0, 1) or not np.isfinite(r) for e, r in costs.values()):
        raise ValueError('three finite historical (shot error, whole risk) costs required')
    error = min(e for e, _ in costs.values())
    risk = min(r for e, r in costs.values() if e == error)
    tied = {a for a, (e, r) in costs.items() if e == error and r <= risk + 1e-6}
    return next(a for a in (0, -1, 1) if a in tied)


def bias_fit(shot_pairs):
    """Equal-shot weighted fit d≈beta*q on actual finite-step raw logits."""
    qd = qq = dd = 0.; used = 0; parts = []
    for q, d in shot_pairs:
        q, d = np.asarray(q, np.float64), np.asarray(d, np.float64)
        if q.shape != d.shape or not np.isfinite(q).all() or not np.isfinite(d).all():
            raise ValueError('matching finite q and logit differences required')
        keep = q != 0
        if not keep.any(): continue
        q, d = q[keep], d[keep]; used += 1
        qd += np.mean(q*d); qq += np.mean(q*q); dd += np.mean(d*d)
        parts.append((q, d))
    if qq == 0 or dd == 0:
        return dict(beta=None, energy_fraction=None, n_shots=used, reason='no_measurable_effect')
    beta = qd / qq
    residual = sum(float(np.mean((d-beta*q)**2)) for q, d in parts)
    return dict(beta=float(beta), energy_fraction=float(1-residual/dd), n_shots=used,
                reason='posthoc_raw_logit_explanation_only')


def predict_fixed(model, signals, direction, sign):
    if signals['device'] != 'EAST':
        raise ValueError('target signal-only probe')
    values, available = np.asarray(signals['values']), np.asarray(signals['available'])
    available = available & np.isfinite(values)
    times = np.asarray(signals['times'])
    if len(times) != len(values) or np.any(np.diff(times) <= 0) or not np.isfinite(times).all():
        raise ValueError('finite increasing signal times required')
    q = fixed_sign_curve(available, sign)
    logits = np.empty(len(q), np.float32); scores = np.empty(len(q), np.float32)
    clean = dict(values=values, available=available)
    start = 0
    while start < len(q):
        stop = start + 1
        while stop < min(len(q), start+256) and q[stop] == q[start]: stop += 1
        x, a = _batch_windows(simple_data, clean, range(start, stop), model, model.center.device)
        delta = (float(q[start]) * direction).detach().requires_grad_(True)
        out = model(x, a, delta)['logit']
        if not torch.isfinite(out).all(): raise FloatingPointError('nonfinite finite-action risk')
        logits[start:stop] = out.detach().cpu().numpy()
        scores[start:stop] = out.detach().sigmoid().cpu().numpy()
        start = stop
    return logits, scores, q


def whole_cost(shot, scores, threshold):
    valid = np.asarray(shot['valid'], bool).copy()
    if shot['is_disrupt']: valid &= np.asarray(shot['times']) <= shot['down']
    row = dict(shot=shot['shot'], is_disrupt=bool(shot['is_disrupt']), down=shot['down'],
               times=np.asarray(shot['times'])[valid], scores=np.asarray(scores)[valid])
    result = simple_train.metrics([row], threshold, lead=30.)['predictions'][0]
    eligible = valid.copy()
    if shot['is_disrupt']:
        eligible &= np.round((shot['down']-np.asarray(shot['times']))*1000., 9) > 30.
    error = int(not result['timely']) if shot['is_disrupt'] else int(result['alarm'])
    if not eligible.any():
        return row, (error, None), result
    peak = float(np.max(np.asarray(scores)[eligible]))
    peak = float(np.clip(peak, float(np.float32(1e-7)), float(np.float32(1-1e-7))))
    margin = math.log(peak/(1-peak)) - math.log(threshold/(1-threshold))
    risk = float(np.logaddexp(0., -margin if shot['is_disrupt'] else margin))
    return row, (error, risk), result


def reference_order(static):
    protocol = json.loads((static.parent/'protocol.json').read_text())
    # Archived runner hashes the ORIGINAL manifest, then serializes a pretty
    # copy into its output. Check original bytes and exact parsed copy content;
    # treating the formatting-only copy as the original rejects valid evidence.
    original = Path(protocol['args']['order'])
    canonical = json.loads(original.read_text())
    copied = json.loads((static.parent/'order.json').read_text())
    if (_hash(original) != protocol['order_sha256'] or canonical != copied
            or len(copied) != 2953 or protocol['source_sha256'] != PARENT_SHA256
            or protocol['rate_contract'] != 'nominal'):
        raise ValueError('same complete nominal2953 source-static reference required')
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direction', type=Path, required=True)
    parser.add_argument('--static', type=Path, required=True, help='completed all2953 static arm directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('local CUDA required')
    torch.set_num_threads(4); torch.manual_seed(131); started = time.monotonic()
    direction_summary = json.loads((args.direction.parent/'summary.json').read_text())
    if _hash(args.direction) != direction_summary['direction_sha256']:
        raise ValueError('direction artifact changed')
    asset = torch.load(args.direction, map_location='cpu', weights_only=True)
    if asset['protocol']['parent_sha256'] != PARENT_SHA256 or not asset['protocol']['source_only']:
        raise ValueError('source-only qualified parent direction required')
    order = reference_order(args.static)
    selected = order[:32]  # Fixed prefix, never search for a favorable block.
    base, state = load_parent(DEFAULT_PARENT); base = base.cuda(); model = FfnModel(base)
    before = classifier_state(base); threshold = float(state['threshold'])
    direction = asset['direction'].to(model.center)
    if (direction.shape != (model.delta_count,) or not torch.isfinite(direction).all()
            or abs(float(direction.double().norm())-1.) > 1e-6):
        raise ValueError('unit source FFN direction required')
    args.output.mkdir(parents=True, exist_ok=False)
    for sign in (-1, 0, 1): (args.output/str(sign)).mkdir()
    root = Path(__file__).parents[1]
    files = [Path(__file__), root/'signed_intra_20261005/direction.py', root/'simple23.py',
             root/'simple_data.py', root/'io23.py', root/'simple_train.py', root/'simple_online.py',
             root/'redesign_20261004/intra.py', root/'ffn_intra_20261005/model.py']
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(f, str(f.relative_to(root)))
    protocol = dict(parent_sha256=PARENT_SHA256, direction_sha256=_hash(args.direction),
        order_sha256=_hash(args.static.parent/'order.json'), shot_ids=[r['shot'] for r in selected],
        selected_count=32, selection='first32 fixed full-stream prefix, no outcome selection',
        signs=[-1, 0, 1], segment_steps=[.02, .08], radius=1., first_effect=6,
        effect_schedule='after risk at origin+5 and origin+25, valid new levels only; origins0,26,...',
        source_unchanged=True, threshold=threshold, alarm_tuning=False, rate_contract='nominal',
        trained_policy=False, interpretation='three fixed legal action schedules; posthoc oracle over these three is NOT online or a general upper bound',
        code_sha256={str(f.resolve()): _hash(f) for f in files}, hardware=torch.cuda.get_device_name(),
        evidence='DEVELOPMENT_DIAGNOSTIC_ONLY; no new physical or independent validation evidence')
    _durable_json(args.output/'protocol.json', protocol)
    records = []; metric_rows = {str(a): [] for a in (-1, 0, 1)}; oracle = []
    pairs = {str(a): [] for a in (-1, 1)}; all_max_difference = 0.; crossing_differences = 0
    for index, item in enumerate(selected, 1):
        sid = int(item['shot']); stem = f'{index:04d}_{sid}'
        receipt = json.loads((args.static/f'{stem}_seal.json').read_text())
        if (receipt['shot'] != sid or receipt['threshold'] != threshold
                or _hash(args.static/receipt['npz']) != receipt['sha256']
                or _hash(Path(item['provenance']['path'])) != item['provenance']['sha256']):
            raise ValueError('sealed static or raw input identity differs')
        signals = simple_data.read_shot('EAST', sid, rate_contract='nominal')
        traces = {}
        for sign in (0, -1, 1):
            logits, scores, q = predict_fixed(model, signals, direction, sign)
            new_receipt = seal_predictions(args.output/str(sign), index, sid, signals['times'], scores,
                                          arm=f'fixed_sign_{sign}', threshold=threshold)
            np.savez_compressed(args.output/str(sign)/f'{stem}_states.npz', logits=logits, q=q)
            traces[sign] = dict(logits=logits, scores=scores, q=q, receipt=new_receipt)
        with np.load(args.static/receipt['npz'], allow_pickle=False) as old:
            if not np.array_equal(old['times'], signals['times']): raise ValueError('static time grid mismatch')
            difference = float(np.max(np.abs(old['scores']-traces[0]['scores'])))
            disagree = int(np.sum((old['scores'] >= threshold) != (traces[0]['scores'] >= threshold)))
        if difference > 3e-6 or disagree: raise RuntimeError('static reference risk parity failed')
        all_max_difference = max(all_max_difference, difference); crossing_differences += disagree
        # All fixed policies' whole-shot traces are now durable, BEFORE outcome access.
        shot = simple_data.attach_outcome(signals, 'EAST', {})
        results, costs, rows = {}, {}, {}
        for sign, trace in traces.items():
            row, cost, prediction = whole_cost(shot, trace['scores'], threshold)
            metric_rows[str(sign)].append(row); rows[sign] = row; costs[sign] = cost
            results[str(sign)] = dict(error=cost[0], whole_risk=cost[1], prediction=prediction,
                                      final_q=float(trace['q'][-1]), seal_sha256=trace['receipt']['sha256'])
        action = choose_historical_action(costs) if all(r is not None for _, r in costs.values()) else 0
        oracle.append(rows[action])
        valid = np.asarray(shot['valid'], bool)
        for sign in (-1, 1):
            pairs[str(sign)].append((traces[sign]['q'][valid],
                                    (traces[sign]['logits']-traces[0]['logits'])[valid]))
        records.append(dict(index=index, shot=sid, is_disrupt=bool(shot['is_disrupt']),
            down=shot['down'], static_reference_max_abs=difference, traces=results,
            posthoc_oracle_sign=action, oracle_is_not_deployed=True,
            surrogate_missing_reason=None if all(r is not None for _, r in costs.values()) else 'no_timely_endpoint_keep_hold'))
        _durable_json(args.output/'records.json', records)
        if index % 4 == 0:
            print(json.dumps(dict(stage='finite_probe', shots=index, total=32, seconds=time.monotonic()-started)), flush=True)
    assert_classifier_unchanged(base, before)
    if _hash(DEFAULT_PARENT) != PARENT_SHA256: raise RuntimeError('qualified parent file changed')
    report = dict(protocol=protocol, metrics={key: simple_train.metrics(rows, threshold, lead=30.)
                    for key, rows in metric_rows.items()}, posthoc_oracle_metrics=simple_train.metrics(oracle, threshold, lead=30.),
        bias_diagnostic={key: bias_fit(value) for key, value in pairs.items()},
        classifier_unchanged=True, source_reference_max_abs=all_max_difference,
        threshold_crossing_disagreements=crossing_differences, elapsed_seconds=time.monotonic()-started,
        training_performed=False, admitted_online_model=False)
    _durable_json(args.output/'summary.json', report)
    print(json.dumps(dict(stage='finite_probe_complete', metrics={k: {f: v[f] for f in ('tp','fp','n_disrupt','n_safe')}
        for k, v in report['metrics'].items()}, bias=report['bias_diagnostic'],
        oracle={f: report['posthoc_oracle_metrics'][f] for f in ('tp','fp')},
        seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
