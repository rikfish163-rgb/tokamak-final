"""Independent weighted least-squares and Cholesky feedback audit."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from .run import encode, REFERENCE


def balanced_weights(ids, y):
    w = np.empty(len(ids), float)
    for i, (sid, c) in enumerate(zip(ids, y)):
        w[i] = .5 / len(set(ids[y == c])) / np.count_nonzero((ids == sid) & (y == c))
    return w


def cholesky_error(pred, target, valid, cov, active):
    errors = np.full(valid.shape[:2], np.nan)
    for use in np.unique(valid.reshape(-1, valid.shape[-1]) & active, axis=0):
        if not use.any():
            continue
        rows, ts = np.where(((valid & active) == use).all(-1))
        r = (pred[rows, ts]-target[rows, ts])[:, use]
        chol = np.linalg.cholesky(cov[np.ix_(use, use)])
        white = np.linalg.solve(chol, r.T)
        errors[rows, ts] = (white**2).sum(0)/use.sum()
    total = np.nansum(errors, axis=1); n = np.isfinite(errors).sum(1)
    result = np.full(len(errors), np.nan)
    np.divide(total, n, out=result, where=n > 0)
    return result


def main(root):
    output = root/'verification.json'
    if output.exists():
        raise FileExistsError(output)
    done = json.loads((root/'completed.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    for name, digest in done['artifacts'].items():
        assert sha(root/name) == digest, name
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    torch.set_num_threads(2)
    base, _ = load_parent(DEFAULT_PARENT)
    center, scale = [v.cpu().numpy().astype(float) for v in (base.center, base.scale)]
    forecasts_checked = 0; max_fit = 0.; max_error = 0.; replay_max = 0.; all_rows = []
    for fold in protocol['folds']:
        directory = root/f'fold{fold["fold"]}'
        with np.load(directory/'fit.npz') as data, np.load(directory/'heads.npz') as heads:
            ids, y = data['ids'], data['y']
            assert set(ids) == set(fold['fit']) and not set(ids) & set(fold['held'])
            w = balanced_weights(ids, y)
            np.testing.assert_allclose((w[:, None]*data['features']).sum(0), heads['center'], atol=1e-12)
            keep = heads['keep']; x = (data['features'][:, keep]-heads['center'][keep])/heads['scale'][keep]
            design = np.column_stack((x, np.ones(len(x))))
            target = np.where(heads['active'], data['target'], 0.).reshape(len(x), -1)
            wrong = y.copy()
            for sid in set(ids):
                indices = np.flatnonzero(ids == sid); offset = len(indices)//2
                wrong[indices] = y[indices[(np.arange(len(indices))-offset) % len(indices)]]
            for name in ('unconditional', 'real0', 'real1', 'misplaced0', 'misplaced1'):
                labels = wrong if name.startswith('misplaced') else y
                select = np.ones(len(y), bool) if name == 'unconditional' else labels == int(name[-1])
                weights = balanced_weights(ids, labels)[select]; weights /= weights.sum()
                a = design[select]*np.sqrt(weights[:, None]); b = target[select]*np.sqrt(weights[:, None])
                reg = np.eye(a.shape[1])*np.sqrt(.1); reg[-1, -1] = 0
                expected = np.linalg.lstsq(np.vstack((a, reg)), np.vstack((b, np.zeros((len(reg), b.shape[1])))), rcond=None)[0]
                delta = float(abs(expected-heads[name]).max()); max_fit = max(max_fit, delta)
                np.testing.assert_allclose(expected, heads[name], atol=1e-9, rtol=1e-7)
            model = PairedFeedback(base, np.ones(23)).cuda()
            checkpoint = REFERENCE/f'fold{fold["fold"]}'/'increment.pt'
            model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True)['model'])
            model.set_stage('frozen')
            for sid in fold['held']:
                raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
                signal = {k: raw[k] for k in ('values', 'available')}
                a = raw['available'] & np.isfinite(raw['values'])
                q = np.arcsinh((np.where(a, raw['values'], center).astype(float)-center)/scale)
                with np.load(directory/f'{sid}.npz') as z:
                    ends = z['origins']; h, p = encode(model, signal, ends)
                    np.testing.assert_array_equal(h, z['features']); np.testing.assert_array_equal(p, z['static'])
                    xx = (h[:, keep]-heads['center'][keep])/heads['scale'][keep]
                    for name in ('unconditional', 'real0', 'real1', 'misplaced0', 'misplaced1'):
                        pred = (np.column_stack((xx, np.ones(len(xx)))) @ heads[name]).reshape(-1, 25, 23)
                        np.testing.assert_array_equal(pred, z[name])
                    disruptive, down = io23.read_outcome('EAST', sid)
                    for horizon in (5, 10, 25):
                        eligible = ends+horizon < len(q); en = ends[eligible]
                        indices = en[:, None]+np.arange(1, horizon+1)
                        target = q[indices]-q[en, None]; valid = a[indices] & a[en, None]
                        np.testing.assert_array_equal(z[f'times{horizon}'][eligible], raw['times'][en+horizon])
                        scores = {}
                        for name in ('unconditional', 'real0', 'real1', 'misplaced0', 'misplaced1', 'persistence'):
                            pred = np.zeros((len(en), horizon, 23)) if name == 'persistence' else z[name][eligible, :horizon]
                            error = cholesky_error(pred, target, valid, heads['covariance'], heads['active'])
                            saved = z[f'{name}_error{horizon}'][eligible]
                            np.testing.assert_allclose(error, saved, atol=1e-8, rtol=1e-7, equal_nan=True)
                            max_error = max(max_error, float(np.nanmax(abs(error-saved))))
                            scores[name] = error; forecasts_checked += len(error)
                        for arm in ('real', 'misplaced'):
                            score = scores[f'{arm}0']-scores[f'{arm}1']
                            np.testing.assert_allclose(score, z[f'{arm}_score{horizon}'][eligible], atol=1e-8, rtol=1e-7)
                            if disruptive:
                                lead_origin = np.round((down-raw['times'][en])*1000, 9)
                                lead_mature = np.round((down-raw['times'][en+horizon])*1000, 9)
                                select = (lead_origin <= 125) & (lead_mature > 30)
                            else:
                                select = np.ones(len(en), bool)
                            all_rows.append((sid, horizon, arm, bool(disruptive), float(np.median(score[select]))))
    rows = json.loads((root/'per_shot.json').read_text())
    for sid, horizon, arm, disruptive, median in all_rows:
        r = next(r for r in rows if r['shot'] == sid and r['horizon'] == horizon and r['arm'] == arm)
        assert r['disruptive'] == disruptive
        np.testing.assert_allclose(median, r['critical']['median'], atol=1e-8, rtol=1e-7)
    _durable_json(output, dict(status='VERIFIED', completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)), ridge_fits=20, full_cuda_shots_replayed=20,
        error_windows_checked=forecasts_checked, shot_summaries_checked=len(all_rows),
        max_ridge_difference=max_fit, max_error_difference=max_error,
        method='CPU augmented least squares; Cholesky whitened errors; full CUDA context/prediction replay',
        source_unchanged=sha(DEFAULT_PARENT) == PARENT_SHA256, test360_accessed=False))
    print(output.read_text(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
