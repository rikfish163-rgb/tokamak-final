"""Independent CPU retrieval and Cholesky error replay of sealed outputs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..conditional_forecast_20261023.audit import cholesky_error
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha


def reference_context(signal, ends, center, scale, active):
    x = signal['values']; observed = signal['available'] & np.isfinite(x)
    q = np.arcsinh((np.where(observed, x, center).astype(float)-center)/scale)[:, active]
    values, valid = [], []
    for end in ends:
        ok = observed[end-49:end+1, active].all()
        values.append(np.r_[q[end], q[end-24:end+1].mean(0)-q[end-49:end-24].mean(0), q[end]-q[end-8]])
        valid.append(ok)
    values = np.array(values); valid = np.array(valid)
    values[~valid] = np.nan
    return values, valid


def reference_neighbors(history, query, ids, labels):
    result = {}
    for c in (0, 1):
        shotids = np.unique(ids[labels == c]); indices, distances = [], []
        for sid in shotids:
            bank = np.flatnonzero((ids == sid) & (labels == c))
            d = cdist(query, history[bank], metric='sqeuclidean')/history.shape[1]
            near = d.argmin(1)
            indices.append(bank[near]); distances.append(d[np.arange(len(query)), near])
        result[c] = np.stack(indices, 1), np.stack(distances, 1), shotids
    return result


def reference_scores(paths, truth, valid, cov, active):
    h = truth.shape[1]
    errors = np.column_stack([cholesky_error(paths[:, j, :h], truth, valid, cov, active) for j in range(paths.shape[1])])
    mean = cholesky_error(np.mean(paths[:, :, :h], axis=1), truth, valid, cov, active)
    mixture = -logsumexp(-errors, axis=1)+np.log(paths.shape[1])
    return mean, mixture, errors


def main(root):
    start = time.monotonic(); output = root/'verification.json'
    if output.exists(): raise FileExistsError(output)
    read = lambda p: json.loads(p.read_text())
    done, protocol = read(root/'completed.json'), read(root/'protocol.json')
    for name, digest in done['artifacts'].items(): assert sha(root/name) == digest, name
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    reference = Path(protocol['reference'])
    assert sha(reference/'completed.json') == protocol['reference_sha256']
    assert read(reference/'verification.json')['status'] == 'VERIFIED'
    base, _ = load_parent(DEFAULT_PARENT)
    center, scale = [v.cpu().numpy().astype(float) for v in (base.center, base.scale)]
    summaries = read(root/'per_shot.json'); accesses = []; errors_checked = neighbors_checked = 0
    replay_medians = []
    max_distance = max_error = 0.
    for fold in protocol['folds']:
        directory = root/f'fold{fold["fold"]}'; ref = reference/f'fold{fold["fold"]}'
        with np.load(directory/'bank.npz') as b, np.load(ref/'fit.npz') as old:
            ids, y = b['ids'], b['y']; active, covariance = b['active'], b['covariance']
            targets = b['target']
            historical_context = {view: b[f'{view}_white'] for view in ('cls', 'signal')}
            assert set(ids) == set(fold['fit']) and not set(ids) & set(fold['held'])
            for key in ('ids', 'y', 'target', 'origins'):
                np.testing.assert_array_equal(b[key], old[key][b['original_usable']])
            np.testing.assert_array_equal(b['cls_context'], old['features'][b['original_usable']])
            wrong = y.copy(); weights = np.zeros(len(y))
            for sid in np.unique(ids):
                indices = np.flatnonzero(ids == sid)
                wrong[indices] = y[indices[(np.arange(len(indices))-len(indices)//2) % len(indices)]]
                raw = apply_input_contract(simple_data.read_shot('EAST', int(sid), rate_contract='nominal'))
                context, valid = reference_context(raw, b['origins'][indices], center, scale, active)
                assert valid.all()
                np.testing.assert_allclose(context, b['signal_context'][indices], atol=1e-12)
            np.testing.assert_array_equal(wrong, b['misplaced'])
            for c in (0, 1):
                shots = np.unique(ids[y == c])
                for sid in shots:
                    use = (ids == sid) & (y == c); weights[use] = .5/len(shots)/use.sum()
            for view in ('cls', 'signal'):
                x = b[f'{view}_context']; mean = np.average(x, axis=0, weights=weights)
                std = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights)); keep = b[f'{view}_keep']
                np.testing.assert_allclose(mean, b[f'{view}_center'], atol=1e-12)
                np.testing.assert_allclose(std, b[f'{view}_scale'], atol=1e-12)
                standardized = (x[:, keep]-mean[keep])/std[keep]
                cov = (standardized*weights[:, None]).T @ standardized
                cov = .9*cov + .1*np.diag(np.diag(cov))
                chol = b[f'{view}_chol']
                np.testing.assert_allclose(cov, chol @ chol.T, atol=1e-12)
                np.testing.assert_allclose(np.linalg.solve(chol, standardized.T).T, b[f'{view}_white'], atol=1e-11)
            for sid in fold['held']:
                raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
                a = raw['available'] & np.isfinite(raw['values'])
                q = np.arcsinh((np.where(a, raw['values'], center).astype(float)-center)/scale)
                disruptive, down = io23.read_outcome('EAST', sid)
                accesses.append(dict(shot=sid, phase='audit_after_completed'))
                with np.load(directory/f'{sid}.npz') as z, np.load(ref/f'{sid}.npz') as previous:
                    ends = z['origins']; context, valid = reference_context(raw, ends, center, scale, active)
                    np.testing.assert_array_equal(valid, z['context_valid'])
                    np.testing.assert_allclose(context, z['signal_context'], equal_nan=True, atol=1e-12)
                    np.testing.assert_array_equal(z['cls_context'], previous['features'])
                    np.testing.assert_array_equal(z['static'], previous['static'])
                    for view in ('cls', 'signal'):
                        keep = b[f'{view}_keep']
                        x = (z[f'{view}_context'][valid][:, keep]-b[f'{view}_center'][keep])/b[f'{view}_scale'][keep]
                        query = np.linalg.solve(b[f'{view}_chol'], x.T).T
                        for label in ('real', 'misplaced'):
                            selected = reference_neighbors(historical_context[view], query, ids, y if label == 'real' else wrong)
                            scores = {}
                            for c, (indices, distances, shotids) in selected.items():
                                key = f'{view}_{label}{c}'
                                np.testing.assert_array_equal(indices, z[f'{key}_indices'])
                                np.testing.assert_array_equal(shotids, z[f'{key}_shotids'])
                                np.testing.assert_allclose(distances, z[f'{key}_distance'], atol=1e-9, rtol=1e-8)
                                max_distance = max(max_distance, float(abs(distances-z[f'{key}_distance']).max()))
                                neighbors_checked += indices.size
                                paths = targets[indices]
                                for h in (5, 10, 25):
                                    mature = ends[valid]+h < len(q); en = ends[valid][mature]
                                    index = en[:, None]+np.arange(1, h+1)
                                    truth = q[index]-q[en, None]; mask = a[index] & a[en, None]
                                    expected = reference_scores(paths[mature], truth, mask, covariance, active)
                                    np.testing.assert_allclose(expected[2], z[f'{key}_individual{h}'], atol=1e-8, rtol=1e-7, equal_nan=True)
                                    for agg, value in zip(('mean', 'mix'), expected[:2]):
                                        all_value = np.full(len(ends), np.nan); all_value[np.flatnonzero(valid)[mature]] = value
                                        saved = z[f'{key}_{agg}_energy{h}']
                                        np.testing.assert_allclose(all_value, saved, atol=1e-8, rtol=1e-7, equal_nan=True)
                                        max_error = max(max_error, float(np.nanmax(abs(all_value-saved))))
                                        scores[c, agg, h] = all_value; errors_checked += len(value)
                            for agg in ('mean', 'mix'):
                                for h in (5, 10, 25):
                                    score = scores[0, agg, h]-scores[1, agg, h]
                                    np.testing.assert_allclose(score, z[f'{view}_{label}_{agg}_score{h}'], atol=1e-8, rtol=1e-7, equal_nan=True)
                                    select = np.isfinite(score)
                                    if disruptive:
                                        select &= (np.round((down-z['times'])*1000, 9) <= 125) & (np.round((down-z[f'times{h}'])*1000, 9) > 30)
                                    row = next(r for r in summaries if (r['shot'], r['view'], r['label'], r['aggregation'], r['horizon']) == (sid, view, label, agg, h))
                                    assert row['critical']['n'] == int(select.sum())
                                    median = float(np.median(score[select]))
                                    np.testing.assert_allclose(median, row['critical']['median'], atol=1e-8, rtol=1e-7)
                                    replay_medians.append((view, label, agg, h, sid, bool(disruptive), median))
        print(dict(fold=fold['fold'], audited=True, seconds=time.monotonic()-start), flush=True)
    saved_summary = read(root/'summary.json')
    for key, row in saved_summary['results'].items():
        view, label, agg, h = key.split('_')
        records = [r for r in replay_medians if r[:4] == (view, label, agg, int(h))]
        assert len(records) == 20
        assert row['danger_correct'] == sum(r[5] and r[6] > 0 for r in records)
        assert row['safe_correct'] == sum(not r[5] and r[6] < 0 for r in records)
    real, wrong = [saved_summary['results'][f'signal_{label}_mix_5'] for label in ('real', 'misplaced')]
    miss = next(r[6] for r in replay_medians if r[:5] == ('signal', 'real', 'mix', 5, 70868))
    expected_gate = bool(real['danger_correct'] >= 3 and real['safe_correct'] >= 12 and miss > 0
                         and real['danger_correct']+real['safe_correct'] > wrong['danger_correct']+wrong['safe_correct'])
    assert saved_summary['neural_trial_gate'] == done['neural_trial_gate'] == expected_gate
    _durable_json(output, dict(status='VERIFIED', completed_sha256=sha(root/'completed.json'), audit_sha256=sha(Path(__file__)),
        seconds=time.monotonic()-start, held_shots=len(accesses), neighbors_checked=neighbors_checked, error_windows_checked=errors_checked,
        shot_summaries_checked=len(summaries), max_distance_difference=max_distance, max_error_difference=max_error,
        source_unchanged=sha(DEFAULT_PARENT) == PARENT_SHA256, test360_accessed=False,
        method='independent sliced past-context; SciPy squared distances; Cholesky errors and scipy logsumexp'))
    print(output.read_text(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
