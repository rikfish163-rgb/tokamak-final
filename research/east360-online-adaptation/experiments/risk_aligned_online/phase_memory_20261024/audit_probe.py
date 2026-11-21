"""Check diagnostic probes against primal ridge solves and independent ROC AUC.

Reads archived historical embeddings only; never reads a stream shot or changes
the operational risk model. The probe is not an online prediction component.
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from ..simple_online import _durable_json
from ..simple_train import sha
from .probe import METHODS
from .run import DEFAULT_PARENT, MODES, PARENT_SHA256, READ


def main(screen):
    root = screen / 'linear_probe'
    done = READ(root / 'completed.json')
    protocol = READ(root / 'protocol.json')
    for name, digest in done['artifact_sha256'].items():
        assert sha(root / name) == digest, name
    assert sha(screen / 'completed.json') == protocol['source_screen_sha256']
    assert sha(screen / 'embeddings.npz') == protocol['embedding_sha256']
    assert sha(Path(__file__).with_name('probe.py')) == protocol['code_sha256']
    with np.load(screen / 'embeddings.npz') as z:
        data = {key: z[key].copy() for key in z.files}
    with np.load(root / 'coefficients.npz') as z:
        coefs = {key: z[key].copy() for key in z.files}
    fits = READ(root / 'fits.json')
    folds = READ(root / 'folds.json')
    results = READ(root / 'results.json')
    shots, labels = data['shots'], data['labels']
    metrics = 0
    max_relative_primal_error = 0.
    pairing_geometry = {}
    for mode in MODES:
        early, late = data[mode + '_early'], data[mode + '_late']
        qe, ql = data[mode + '_query_early'], data[mode + '_query_late']
        pairing_geometry[mode] = {}
        for dsid in np.unique(shots[labels == 1]):
            rows = np.flatnonzero((labels == 1) & (shots != dsid))
            expected_shots = np.unique(shots[rows]).tolist()
            assert len(rows) == 48 and len(expected_shots) == 3
            rhs = {}
            for method in METHODS:
                key = f'{mode}_{dsid}_{method}'
                fit = fits[key]
                assert fit['training_rows'] == rows.tolist()
                assert fit['training_shots'] == expected_shots and dsid not in expected_shots
                ix, target = np.array(fit['early_indexes']), np.array(fit['targets'])
                np.testing.assert_array_equal(np.sort(ix), np.arange(len(rows)))
                if method == 'unpaired':
                    assert (shots[rows[ix]] != shots[rows]).all()
                    assert (target == 1).all()
                else:
                    np.testing.assert_array_equal(ix, np.arange(len(rows)))
                    if method == 'ordered':
                        assert (target == 1).all()
                    else:
                        assert np.isin(target, [-1, 1]).all()
                        assert all(target[shots[rows] == sid].sum() == 0 for sid in expected_shots)
                delta = late[rows] - early[rows[ix]]
                ridge = float(np.mean(np.sum(delta ** 2, axis=1)))
                assert ridge == fit['ridge']
                rhs[method] = delta.T @ target / len(rows)
                matrix = delta.T @ delta / len(rows) + ridge * np.eye(delta.shape[1])
                primal = np.linalg.solve(matrix, rhs[method])
                assert np.isfinite(coefs[key]).all()
                # Primal and dual ridge solves use different matrix orders.
                # Their condition is <=2 here: lambda=trace(X'X/n)>=lambda_max.
                relative = np.linalg.norm(primal - coefs[key]) / np.linalg.norm(primal)
                assert relative < 1e-12
                max_relative_primal_error = max(max_relative_primal_error, float(relative))
            # All-one targets leave the RHS invariant under pairing permutation.
            np.testing.assert_allclose(rhs['ordered'], rhs['unpaired'], atol=1e-13, rtol=1e-12)
            a, b = [coefs[f'{mode}_{dsid}_{m}'] for m in ('ordered', 'unpaired')]
            pairing_geometry[mode][str(dsid)] = dict(
                ordered_unpaired_cosine=float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
                rhs_difference_norm=float(np.linalg.norm(rhs['ordered'] - rhs['unpaired'])))
        for method in METHODS:
            rows = folds[mode][method]
            assert len(rows) == 64
            per_shot = {}
            for row in rows:
                dsid, ssid = row['held_shots']
                w = coefs[f'{mode}_{dsid}_{method}']
                li, ei, si = [np.array(row[key]) for key in ('late_rows', 'early_unique_rows', 'safe_rows')]
                assert (shots[li] == dsid).all() and (shots[ei] == dsid).all()
                assert (shots[si] == ssid).all() and (labels[si] == 0).all()
                a, b, c = ql[li] @ w, qe[ei] @ w, ql[si] @ w
                for score, key in zip((a, b, c), ('late_scores', 'early_scores', 'safe_scores')):
                    np.testing.assert_array_equal(score, row[key])
                for negative, key in ((b, 'phase_auc'), (c, 'late_safe_auc')):
                    independent = roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(negative))], np.r_[a, negative])
                    np.testing.assert_allclose(independent, row[key], rtol=0, atol=np.finfo('float64').eps)
                    metrics += 1
                diff = a - b[np.array(row['pair_to_unique'])]
                assert row['paired_order'] == float(np.mean((diff > 0) + .5 * (diff == 0)))
                metrics += 1
                per_shot.setdefault(str(dsid), []).append(row)
            for key in ('phase_auc', 'late_safe_auc', 'paired_order'):
                per = {sid: float(np.mean([r[key] for r in rr])) for sid, rr in per_shot.items()}
                assert all(results[mode][method]['per_disruptive'][sid][key] == value for sid, value in per.items())
                assert float(np.mean(list(per.values()))) == results[mode][method][key]
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    assert len(coefs) == len(fits) == done['fits'] == 120
    verification = dict(completed_sha256=sha(root / 'completed.json'), code_sha256=sha(Path(__file__)),
        fits_checked=120, metrics_checked=metrics, exact_score_replays=1920,
        max_relative_primal_dual_error=max_relative_primal_error,
        heldout_shots_used_for_fitting=False, original_source_unchanged=True,
        stream360_accessed=False, scope='historical frozen-representation diagnostic only')
    _durable_json(root / 'verification.json', verification)
    _durable_json(root / 'pairing_geometry.json', pairing_geometry)
    print(verification)
    print(dict(risk_pairing_geometry=pairing_geometry['risk']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--screen', type=Path, required=True)
    main(parser.parse_args().screen)
