"""Recount saved local geometry with NumPy Gram eigenspaces, not Torch SVD."""
import argparse
import json
from pathlib import Path

import numpy as np

from .simple_online import _durable_json, _hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('preserve previous verification')
    saved = json.loads((args.root / 'completed.json').read_text())
    assert saved['all_curves_exact'] and saved['registered_tensors_unchanged']
    rows = []
    for case in saved['rows']:
        assert case['replay_curve_max_error'] == 0
        for e in case['events']:
            path = args.root / f'{case["shot"]}_{e["query"]}_matrices.npz'
            assert _hash(path) == e['matrices_sha256']
            with np.load(path, allow_pickle=False) as f:
                j, g, qg, c = (f[k].astype(np.float64) for k in
                    ('jacobian', 'state_gradient', 'query_gradient', 'coefficients'))
            eigen, u = np.linalg.eigh(j @ j.T)
            assert eigen[-1] > 0 and eigen.min() >= -1e-12 * eigen[-1]
            eigenspectrum = np.maximum(eigen, 0)[::-1]
            np.testing.assert_allclose(eigenspectrum**.5,
                e['rank_sensitivity'][0]['singular_values'], rtol=1e-7, atol=1e-10)
            errors = []
            for r in e['rank_sensitivity']:
                keep = eigen > eigen[-1] * r['rtol']**2
                assert int(keep.sum()) == r['rank']
                coordinates = ((u[:, keep].T @ j) @ qg) / np.sqrt(eigen[keep])
                fraction = float(np.linalg.norm(coordinates) / np.linalg.norm(qg))
                error = abs(fraction - r['projected_gradient_fraction'])
                assert error < 1e-8
                errors.append(error)
            reconstruction_error = float(np.linalg.norm(j.T @ c - g) / np.linalg.norm(g))
            cosine = -float(g @ qg) / float(np.linalg.norm(g) * np.linalg.norm(qg))
            assert abs(reconstruction_error - e['chain_rule_relative_error']) < 1e-10
            assert abs(cosine - e['state_descent_to_logit_increase_cosine']) < 1e-10
            rows.append(dict(shot=case['shot'], query=e['query'],
                max_fraction_difference=max(errors), chain_rule_relative_error=reconstruction_error,
                fraction=e['rank_sensitivity'][1]['projected_gradient_fraction'], descent_cosine=cosine))
    assert len(rows) == 7
    report = dict(verification='PASS', verified_episodes=len(rows),
        saved_matrices_hash_verified=True, method='independent NumPy Gram eigendecomposition',
        negative_descent_cosines=sum(r['descent_cosine'] < 0 for r in rows),
        fraction_range=[min(r['fraction'] for r in rows), max(r['fraction'] for r in rows)],
        median_fraction=float(np.median([r['fraction'] for r in rows])),
        max_difference=max(r['max_fraction_difference'] for r in rows), rows=rows,
        source_completion_sha256=_hash(args.root / 'completed.json'),
        verification_code_sha256=_hash(Path(__file__)),
        scope='local numerical geometry only; does not independently rerun autograd or establish rescue capacity')
    _durable_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)


if __name__ == '__main__': main()
