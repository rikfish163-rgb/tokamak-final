"""Recount sealed traces; audit timing, factorial contrasts and sampled replay."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.audit import paired_flips
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..fixed_test_protocol import load_test_contract
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..target_feedback_meta_20261023.audit import recount
from ..warmstart_alignment_20261022.alignment import transform
from .run import READ, NAMES, ARMS, load_candidate, predict


def audit(root, replay):
    p, done, full = [READ(root / f) for f in ('protocol.json', 'completed.json', 'per_shot.json')]
    assert done['status'] == 'COMPLETED' and done['all_parameters_unchanged']
    assert sha(root / 'protocol.json') == done['protocol_sha256']
    assert sha(root / 'preparation_probe.json') == done['probe_sha256']
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    assert sha(Path(p['plan_path'])) == p['plan_sha256']
    for f, h in p['code_sha256'].items():
        assert sha(io23.ROOT / f) == h, f
    reference = Path(p['reference'])
    for f, h in p['reference_hashes'].items():
        assert sha(reference / f) == h, f
    assert sha(reference / 'static_trained.pt') == p['checkpoint_sha256']
    assert p['contract'] == load_test_contract('EAST')
    order = p['contract']['shot_ids']
    assert len(p['selected_ids']) == len(set(p['selected_ids'])) == 20
    assert not set(p['selected_ids']) & set(order)
    receipts = READ(root / 'prediction_receipts.json')
    assert [r['shot'] for r in receipts] == order
    old_receipts = {r['shot']: r for r in READ(reference / 'prediction_receipts.json')}
    totals = {a: dict(tp=0, fp=0, n_disrupt=0, n_safe=0, leads=[], early=0) for a in NAMES}
    contrasts = {'ridge_vs_mean_recent': ('recent_mean', 'recent_ridge'),
                 'ridge_vs_mean_lagged': ('lagged_mean', 'lagged_ridge'),
                 'lagged_vs_recent_ridge': ('recent_ridge', 'lagged_ridge'),
                 'lagged_vs_recent_mean': ('recent_mean', 'lagged_mean'),
                 'warmup_change': ('reference_fit', 'recent_ridge')}
    contrasts.update({a + '_vs_prior': ('prior', a) for a in (*ARMS, 'recent_ridge')})
    differences = {k: dict(n=0, abs_sum=0., max_abs=0., changed_leads=0) for k in contrasts}
    statistics = {a: dict(points=0, active=0, fitted=0, error_sum=0., targets=0, correction_sum=0.) for a in ARMS}
    timing = []
    for index, seal in enumerate(receipts):
        sid = seal['shot']
        path = root / 'shots' / f'{sid}.npz'
        sp = root / 'shots' / f'{sid}_stats.json'
        old = reference / 'shots' / f'{sid}.npz'
        assert sha(path) == seal['prediction_sha256'] and sha(sp) == seal['stats_sha256']
        assert sha(old) == seal['reference_sha256'] == old_receipts[sid]['prediction_sha256']
        disrupt, down = io23.read_outcome('EAST', sid)
        with np.load(path, allow_pickle=False) as z, np.load(old, allow_pickle=False) as prior:
            assert set(z.files) == {'times', *NAMES}
            np.testing.assert_array_equal(z['times'], prior['times'])
            np.testing.assert_array_equal(z['prior'], prior['static_prior'])
            np.testing.assert_array_equal(z['reference_fit'], prior['static_fit'])
            np.testing.assert_array_equal(z['recent_ridge'][256:], z['reference_fit'][256:])
            for a in (*ARMS, 'recent_ridge'):
                np.testing.assert_array_equal(z[a][:256], z['prior'][:256])
            for a in NAMES:
                row = recount(sid, z['times'], z[a], disrupt, down, p['threshold'])
                assert row == full[a]['predictions'][index], (sid, a)
                v = totals[a]
                v['n_disrupt' if disrupt else 'n_safe'] += 1
                v['tp'] += row['timely']; v['fp'] += not disrupt and row['alarm']
                if row['timely']:
                    v['leads'].append(row['lead_ms']); v['early'] += row['lead_ms'] > 3000
            for name, (a, b) in contrasts.items():
                diff = np.abs(z[a].astype(float) - z[b]); v = differences[name]
                v['n'] += len(diff); v['abs_sum'] += float(diff.sum())
                v['max_abs'] = max(v['max_abs'], float(diff.max()))
                v['changed_leads'] += full[a]['predictions'][index]['lead_ms'] != full[b]['predictions'][index]['lead_ms']
            times = z['times']; n = len(times)
            assert np.all(np.diff(times) > 0) and n > 256
            lag = (times[256:] - times[128:-128]) * 1000
            # The last training target is patch14, which ends eight points
            # before its reference endpoint; patch15 is not used in fitting.
            recent_age = (times[256:] - times[248:-8]) * 1000
            past_age = (times[256:] - times[120:-136]) * 1000
            timing.append(dict(shot=sid, points=n,
                first_active_elapsed_ms=float((times[256]-times[0])*1000),
                reference_lag_ms_min=float(lag.min()), reference_lag_ms_max=float(lag.max()),
                last_fit_target_age_recent_ms=float(np.median(recent_age)),
                last_fit_target_age_lagged_ms=float(np.median(past_age))))
        stats = READ(sp); assert set(stats) == set(ARMS)
        for a, v in stats.items():
            assert v['points'] == n and v['active'] == n - 256
            assert 0 <= v['fitted'] <= v['active'] and 0 <= v['targets'] <= 23 * v['active']
            assert v['error_sum'] >= 0 and v['correction_sum'] >= 0
            for k in statistics[a]: statistics[a][k] += v[k]
    for a, v in totals.items():
        metric = done['metrics'][a]
        for key in ('tp', 'fp', 'n_disrupt', 'n_safe'): assert metric[key] == v[key]
        assert v['n_disrupt'] == v['n_safe'] == 180
        assert metric['median_lead_ms'] == float(np.median(v['leads']))
        assert metric['very_early_over3000'] == v['early']
        assert metric['tpr'] == v['tp']/180 and metric['fpr'] == v['fp']/180
        assert metric['J'] == metric['tpr']-metric['fpr']
    for v in differences.values(): v['mean_abs'] = v['abs_sum']/v['n']
    for v in statistics.values():
        v['mean_held_smoothl1'] = v['error_sum']/v['targets']
        v['mean_correction_norm_active'] = v['correction_sum']/v['active']
    replayed = []
    if replay:
        torch.set_num_threads(2); torch.manual_seed(131)
        model, threshold, mapping = load_candidate()
        assert threshold == p['threshold'] and not any(v.requires_grad for v in model.parameters())
        state = state_cpu(model)
        for sid in (order[0], order[len(order)//2], order[-1]):
            s = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
            scores, stats = predict(model, transform({k:s[k] for k in ('values','available')}, mapping), replay_recent=True)
            with np.load(root / 'shots' / f'{sid}.npz', allow_pickle=False) as z:
                for a in scores: np.testing.assert_array_equal(scores[a], z[a])
            assert {a:stats[a] for a in ARMS} == READ(root / 'shots' / f'{sid}_stats.json')
            replayed.append(dict(shot=sid, curves=4, max_abs_error=0., stats_exact=True))
        assert_unchanged(model, state)
    flips = {k:paired_flips(full[a]['predictions'],full[b]['predictions']) for k,(a,b) in contrasts.items()}
    changed = sorted({sid for result in flips.values() for ids in result.values() for sid in ids})
    details = {sid:{a:next(r for r in full[a]['predictions'] if r['shot']==sid) for a in NAMES} for sid in changed}
    j = {a:v['J'] for a,v in done['metrics'].items()}
    effects = dict(ridge_minus_mean_recent=j['recent_ridge']-j['recent_mean'],
        ridge_minus_mean_lagged=j['lagged_ridge']-j['lagged_mean'],
        lagged_minus_recent_ridge=j['lagged_ridge']-j['recent_ridge'],
        lagged_minus_recent_mean=j['lagged_mean']-j['recent_mean'])
    effects['interaction_lagged_minus_recent_ridge_effect'] = effects['ridge_minus_mean_lagged']-effects['ridge_minus_mean_recent']
    result = dict(status='VERIFIED', audit_sha256=sha(Path(__file__)),
        completed_sha256=sha(root/'completed.json'), protocol_sha256=sha(root/'protocol.json'),
        all_traces_independently_recounted=len(NAMES)*len(order), sampled_cuda_replays=replayed,
        full_independent_reinference=False, frozen_parameters_verified_on_replay=bool(replay),
        probability_differences=differences, statistics=statistics, paired_contrasts=flips,
        changed_shot_details=details, descriptive_factor_J=effects, actual_time_by_shot=timing,
        independent_confirmation=False, uncertainty='single seed, development-exposed TEST; no significance claim',
        label_access_claim='TEST split identities/classes were available; HDF outcome access was blocked until traces sealed')
    _durable_json(root/'verification.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('actual_time_by_shot','changed_shot_details')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    parser.add_argument('--replay', action='store_true'); args=parser.parse_args()
    audit(args.root, args.replay)
