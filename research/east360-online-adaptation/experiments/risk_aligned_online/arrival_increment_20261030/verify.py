"""Recompute saved-array metrics independently of the experiment evaluator."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from ..simple_train import sha
from ..simple_online import _durable_json


def read(path):
    return json.loads(path.read_text())


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def verify(directory):
    protocol = read(directory/'protocol.json')
    cache = Path(protocol['cache'])
    outcomes = read(cache/'outcomes.json')
    reported = read(directory/'per_shot.json')
    indexed = {(r['fold'], r['shot'], r['arm'], r['group']): r for r in reported}
    if len(indexed) != len(reported):
        raise AssertionError('duplicate per-shot records')
    differences, checked, norms, episodes = [], set(), {}, 0
    counts = {a: 0 for a in protocol['arms']}
    all_held = []
    for spec in protocol['folds']:
        assert not set(spec['fit']) & set(spec['held'])
        assert len(spec['fit']) == 16 and len(spec['held']) == 4
        all_held.extend(spec['held'])
        folder = directory/f"fold{spec['fold']}"
        neural, sealed = load(folder/'neural.npz'), load(folder/'sealed.npz')
        baseline = load(cache/f"fold{spec['fold']}"/'neural.npz')
        assert sha(folder/'neural.npz') == read(folder/'neural_seal.json')['sha256']
        assert sha(folder/'sealed.npz') == read(folder/'sealed.json')['sha256']
        for name in ('ids', 'ends', 'future_times', 'baseline_logits', 'origin_logit'):
            np.testing.assert_array_equal(neural[name], baseline[name])
        lookup = {(sid, end): i for i, (sid, end) in enumerate(zip(sealed['ids'], sealed['ends'], strict=True))}
        joined = np.array([lookup[sid, end] for sid, end in zip(neural['ids'], neural['ends'], strict=True)])
        np.testing.assert_array_equal(neural['origin_time'], sealed['origin_time'][joined])
        np.testing.assert_array_equal(neural['arrival_time'], sealed['arrival_time'][joined])
        assert (neural['future_times'] > neural['arrival_time'][:, None]).all()
        episodes += len(joined)
        for arm in protocol['arms']:
            e = sealed[arm+'_potentials'][joined, 1]-sealed[arm+'_potentials'][joined, 0]
            z = neural['origin_logit'].astype(float)
            expected = expit(z)-expit(z+e)
            expected[sealed['observed_channels'][joined] == 0] = 0
            np.testing.assert_allclose(neural[arm+'_evidence_gradient'], expected, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(neural[arm+'_state_norm'], .05*np.abs(expected), rtol=3e-6, atol=1e-9)
            assert np.max(neural[arm+'_state_norm']) <= .050001
            counts[arm] += int(np.count_nonzero(neural[arm+'_state_norm']))
            norms[arm] = max(norms.get(arm, 0.), float(neural[arm+'_state_norm'].max()))
        np.testing.assert_array_equal(neural['identical_logits'], neural['baseline_logits'])
        assert not neural['identical_state_norm'].any()
        for sid in spec['held']:
            selected = neural['ids'] == sid
            times = neural['future_times'][selected]
            outcome = outcomes[str(sid)]
            if outcome['disruptive']:
                remaining = (outcome['down']-times)*1000
                valid = remaining >= -1e-6
                y = ((remaining <= 125+1e-6) & valid).astype(float)
                timely = valid & (y == 1) & (np.round(remaining, 9) > 30)
                original = (outcome['down']-neural['origin_time'][selected])*1000
                maturity = (outcome['down']-neural['arrival_time'][selected])*1000
                intersection = (np.round(original, 9) <= 125) & (original >= 0) & (np.round(maturity, 9) > 30)
                groups = dict(danger_timely=timely,
                              danger_origin_intersection=timely & intersection[:, None],
                              danger_late=valid & (y == 1) & ~timely, early_normal=valid & (y == 0))
            else:
                y = np.zeros_like(times)
                groups = {'safe': np.ones_like(times, bool)}
            z0 = neural['baseline_logits'][selected].astype(float)
            loss0 = np.logaddexp(0, z0)-y*z0
            for arm in protocol['arms']:
                z1 = neural[arm+'_logits'][selected].astype(float)
                loss1 = np.logaddexp(0, z1)-y*z1
                for group, mask in groups.items():
                    if not mask.any():
                        continue  # Absent group is not included in the aggregate.
                    row_key = (spec['fold'], sid, arm, group)
                    checked.add(row_key)
                    row = indexed[row_key]
                    assert row['timepoints'] == int(mask.sum())
                    assert row['windows'] == int(mask.any(1).sum())
                    for name, value in dict(before_bce=loss0[mask].mean(), after_bce=loss1[mask].mean(),
                                            delta_bce=(loss1-loss0)[mask].mean(),
                                            mean_logit_change=(z1-z0)[mask].mean()).items():
                        differences.append(abs(float(value)-row[name]))
    assert len(set(all_held)) == len(all_held) == 20
    assert checked == set(indexed)
    assert max(differences) < 1e-12
    complete = read(directory/'completed.json')
    assert complete['nonzero_episode_updates'] == counts
    for arm, by_group in complete['groups'].items():
        for group, record in by_group.items():
            values = np.array([r['delta_bce'] for r in reported if r['arm'] == arm and r['group'] == group])
            assert len(values) == record['shots']
            assert int((values < 0).sum()) == record['improved']
            assert abs(float(values.mean())-record['mean_delta_bce']) < 1e-12
    accesses = read(directory/'accesses.json')
    assert len(accesses) == 100
    assert all(r['role'] == 'fit' for r in accesses[:80])
    assert all(r['role'] == 'evaluate_after_all_seals' for r in accesses[80:])
    for spec in protocol['folds']:
        assert {r['shot'] for r in accesses[:80] if r['fold'] == spec['fold']} == set(spec['fit'])
    result = dict(checked_per_shot_rows=len(checked), maximum_metric_difference=max(differences),
                  episodes=episodes, source_baseline_arrays_exact=True, folds_disjoint=True,
                  feedback_and_state_norms_verified=True, future_after_arrival=True,
                  nonzero_episode_updates=counts, maximum_state_norm=norms,
                  identical_control_exact=True,
                  note='Independent saved-array recomputation, not a second neural forward replay.')
    _durable_json(directory/'independent_audit.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    verify(parser.parse_args().directory)
