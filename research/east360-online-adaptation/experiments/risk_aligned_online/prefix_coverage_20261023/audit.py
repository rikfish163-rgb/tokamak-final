"""Recount sealed curves, replay predictions, and refit both fold0 objectives."""
import argparse
import math
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..credit_horizon_20261023.run import READ, REFERENCE, load_model, schedule
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..east_joint_feedback_20261023.learning import enable
from ..horizon_memory_20261023.audit import independently_score, aggregate
from ..increment_forecast_20261023.learning import predict
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_source_train
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .run import ARMS, PREVIOUS, PREVIOUS_SHA, EXTERNAL, EXTERNAL_SHA, PLAN, train


def changes(before, after):
    pairs = list(zip(before, after))
    assert all(x['shot'] == y['shot'] for x, y in pairs)
    return dict(
        rescued=[x['shot'] for x, y in pairs if x['is_disrupt'] and not x['timely'] and y['timely']],
        lost=[x['shot'] for x, y in pairs if x['is_disrupt'] and x['timely'] and not y['timely']],
        removed_fp=[x['shot'] for x, y in pairs if not x['is_disrupt'] and x['alarm'] and not y['alarm']],
        added_fp=[x['shot'] for x, y in pairs if not x['is_disrupt'] and not x['alarm'] and y['alarm']])


def main(root):
    if (root/'verification.json').exists():
        raise FileExistsError(root/'verification.json')
    start = time.monotonic()
    p, done = READ(root/'protocol.json'), READ(root/'completed.json')
    assert done['status'] == 'COMPLETED'
    assert sha(PREVIOUS/'completed.json') == PREVIOUS_SHA == p['previous_sha256']
    assert sha(EXTERNAL/'completed.json') == EXTERNAL_SHA == p['external_sha256']
    for name, digest in done['artifact_sha256'].items():
        assert sha(root/name) == digest, name
    for name, digest in p['code_sha256'].items():
        assert sha(io23.ROOT/name) == digest, name
    for group in ('checkpoints', 'input_sha256'):
        for name, digest in p[group].items():
            assert sha(name) == digest, name
    assert sha(PLAN) == p['plan_sha256'] and sha(DEFAULT_PARENT) == PARENT_SHA256
    assert len(p['external_ids']) == 65
    all_ids = {s for plan in p['folds'] for s in plan['fit']+plan['held']}
    assert len(all_ids) == 20 and not all_ids.intersection(p['external_ids'])
    assert not (all_ids | set(p['external_ids'])).intersection(io23.cohort_ids('EAST'))
    torch.set_num_threads(2)
    torch.manual_seed(131)
    original_outcome = io23.read_outcome
    source_access = []

    def source_guard(device, sid):
        assert device == 'J-TEXT' and sid in p['source_train_ids']
        source_access.append(sid)
        return original_outcome(device, sid)

    with patch.object(io23, 'read_outcome', side_effect=source_guard):
        source, source_ids = load_source_train()
    assert source_ids == source_access == p['source_train_ids']
    counts = dict(offline_steps=0, intra_steps=0)
    replays, fit_access, summaries, all_changes, timing = [], [], {}, {}, {}
    held_rows = {a: [] for a in ARMS}
    training_counts = {}
    for plan in p['folds']:
        f = plan['fold']
        folder = root/f'fold{f}'
        warm = load_model(f)
        initial = state_cpu(warm)
        allowed = enable(warm)
        warm.set_stage('frozen')
        model = load_model(f)
        model.load_state_dict(torch.load(folder/'prefix.pt', map_location='cuda', weights_only=True)['model'])
        model.set_stage('frozen')
        assert_unchanged(model, initial, tuple(allowed))
        frozen = state_cpu(model)
        mapping = READ(folder/'input_map.json')
        assert mapping == READ(REFERENCE/f'fold{f}'/'input_map.json')
        history, oldhistory = READ(folder/'training.json'), READ(PREVIOUS/f'fold{f}'/'training.json')
        assert len(history) == len(oldhistory) == 64
        safe_prefix_targets = 0
        for row, old in zip(history, oldhistory):
            assert row['step'] == old['step']
            assert row['source_examples'] == old['source_examples']
            assert len(row['source_examples']) == 32
            assert all(e['shot'] in source_ids for e in row['source_examples'])
            for e, oe in zip(row['episodes'], old['meta']['episodes']):
                assert all(e[k] == oe[k] for k in ('shot', 'count', 'ends', 'disruptive', 'all_counts'))
                assert e['shot'] in plan['fit'] and min(e['ends']) >= 256
                if e['disruptive']:
                    assert e['prefix_ends'] == [] and e['prefix_risk'] is None
                    expected = .5*(e['risk_before']+e['risk_after'])+.2*e['forecast']
                else:
                    expected_ends = np.linspace(0, 255, len(e['ends']), dtype=int).tolist()
                    assert e['prefix_ends'] == expected_ends
                    safe_prefix_targets += len(expected_ends)
                    expected = .5*(e['prefix_risk']+e['risk_after'])+.2*e['forecast']
                assert math.isclose(e['objective'], expected, rel_tol=2e-6, abs_tol=2e-7)
            assert row['objective'] == row['source_bce']+sum(e['objective'] for e in row['episodes'])/2
        training_counts[str(f)] = dict(steps=64, prefix_target_exposures=safe_prefix_targets)
        counts['offline_steps'] += len(history)

        def fit_guard(device, sid):
            assert device == 'EAST' and sid in plan['fit']
            fit_access.append(dict(fold=f, shot=sid, phase='fit'))
            return original_outcome(device, sid)

        with patch.object(io23, 'read_outcome', side_effect=fit_guard):
            shots = []
            for sid in plan['fit']:
                s = simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal')), 'EAST', {})
                shots.append(dict(s, **transform({k: s[k] for k in ('values', 'available')}, mapping)))
            entries, _ = schedule(shots, plan['fit'], f)
            assert entries == READ(folder/'schedule.json')
            if f == 0:
                for arm in ('prefix', 'baseline'):
                    out = root/f'audit_refit0_{arm}'
                    out.mkdir()
                    refit, hist = train(warm, shots, entries, p['threshold'], out, f, source, arm=arm)
                    if arm == 'prefix':
                        assert hist == history
                        assert_unchanged(refit, frozen)
                    else:
                        oldstate = torch.load(PREVIOUS/'fold0'/'meta.pt', map_location='cpu', weights_only=True)['model']
                        assert_unchanged(refit, oldstate)
                        for r, old in zip(hist, oldhistory):
                            assert r['step'] == old['step'] and r['source_examples'] == old['source_examples']
                            for key in ('source_bce', 'objective', 'grad_norm'):
                                assert r[key] == old['meta'][key], key
                            for e, oe in zip(r['episodes'], old['meta']['episodes']):
                                assert {k: e[k] for k in oe} == oe
                    del refit

        def deny(*args, **kwargs):
            raise AssertionError('no outcome during prediction replay')

        with patch.object(io23, 'read_outcome', side_effect=deny):
            for sid in plan['held']+[54314, 77639, 89117]:
                s = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
                signal = transform({k: s[k] for k in ('values', 'available')}, mapping)
                pred, updates, _ = predict(model, signal)
                assert updates == READ(folder/'shots'/f'{sid}_updates.json')
                with np.load(folder/'shots'/f'{sid}.npz') as z:
                    for a in ('static', 'intra'):
                        np.testing.assert_array_equal(pred[a], z[f'prefix_{a}'])
                    np.testing.assert_array_equal(s['times'], z['times'])
                replays.append(dict(fold=f, shot=sid, exact=True, arms=2))
        assert_unchanged(model, frozen)
        assert_unchanged(warm, initial)
        summaries[str(f)], all_changes[str(f)], timing[str(f)] = {}, {}, {}
        for group, ids in (('held', plan['held']), ('external', p['external_ids'])):
            rows = {a: [] for a in ARMS}
            buckets = {a: dict(padded=[], full_before_update=[], after_update=[]) for a in ARMS}
            saved = READ(folder/f'{group}_per_shot.json')
            for sid in ids:
                d, down = original_outcome('EAST', sid)
                reference = PREVIOUS if group == 'held' else EXTERNAL
                key = 'meta' if group == 'held' else 'prepared'
                with np.load(folder/'shots'/f'{sid}.npz') as z, np.load(reference/f'fold{f}'/'shots'/f'{sid}.npz') as old:
                    valid = np.ones(len(z['times']), bool) if not d else (down-z['times'])*1000 >= -1e-6
                    for a in ('static', 'intra'):
                        np.testing.assert_array_equal(z[f'old_{a}'], old[f'{key}_{a}'])
                    np.testing.assert_array_equal(z['prefix_static'][:256], z['prefix_intra'][:256])
                    for a in ARMS:
                        rows[a].append(independently_score(sid, d, down, z['times'][valid], z[a][valid], p['threshold']))
                        if not d:
                            crossed = [i for i, (v, ok) in enumerate(zip(z[a], valid)) if ok and v >= p['threshold']]
                            if crossed:
                                i = crossed[0]
                                bucket = 'padded' if i < 127 else 'full_before_update' if i < 256 else 'after_update'
                                buckets[a][bucket].append(dict(shot=sid, index=i, time=float(z['times'][i])))
                    updates = READ(folder/'shots'/f'{sid}_updates.json')
                    assert len(updates) == len(range(256, len(z['times']), 128))
                    for i, r in enumerate(updates):
                        count = 256+128*i
                        assert r['last_observation'] == count-1 and r['first_affected_prediction'] == count
                        assert all(e+25 <= count-1 for e in r['forecast_origins'])
                        assert r['parameter_delta_norm'] <= .00101
                        counts['intra_steps'] += r['gradient_steps']
            for a in ARMS:
                assert rows[a] == saved[a]['predictions']
                if group == 'held': held_rows[a].extend(rows[a])
            summaries[str(f)][group] = {a: aggregate(rs, p['threshold']) for a, rs in rows.items()}
            all_changes[str(f)][group] = {f'{b}->{a}': changes(rows[b], rows[a]) for b, a in (
                ('old_static', 'prefix_static'), ('old_intra', 'prefix_intra'), ('prefix_static', 'prefix_intra'))}
            timing[str(f)][group] = buckets
        print(dict(stage='audit', fold=f, replays=len(replays), seconds=round(time.monotonic()-start, 2)), flush=True)
    assert summaries == done['metrics'] and all_changes == done['changes']
    assert counts == done['counts'] and fit_access == READ(root/'fit_label_access.json')
    assert timing == READ(root/'first_false_alarms.json')
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    result = dict(status='VERIFIED_DIAGNOSTIC', completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)), scalar_recounts=4*70*4, full_replays=replays,
        refitted_fold0=['prefix', 'baseline'], baseline_exactly_matches_old_meta=True,
        held_metrics={a: aggregate(rows, p['threshold']) for a, rows in held_rows.items()},
        training_counts=training_counts, counts=counts, source_unchanged=True,
        test360_accessed=False, seconds=time.monotonic()-start,
        limitation='Development cohorts; only four disruptive identities per group; neural replay uses same implementation.')
    _durable_json(root/'verification.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
