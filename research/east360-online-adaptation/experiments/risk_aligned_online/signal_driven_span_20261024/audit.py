"""Independent shot metrics, span arithmetic and fixed-shot CUDA replay."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import ARMS, predict
from .run import (READ, PAIR, PREVIOUS, REFERENCE, CAL, make_parent, models_for_calibration,
                 state_cpu, assert_unchanged, signal, transform, DEFAULT_PARENT, PARENT_SHA256, predict_many)


def main(root):
    done, p = READ(root/'completed.json'), READ(root/'protocol.json')
    for f, h in done['artifact_sha256'].items():
        assert sha(root/f) == h, f
    for f, h in p['input_sha256'].items():
        assert sha(Path(f)) == h, f
    for f, h in p['history_sha256'].items():
        assert sha(Path(f)) == h, f
    outcomes = READ(root/'outcomes.json'); records = READ(root/'per_shot.json')
    group = {a: {False: [], True: []} for a in ('off',)+ARMS}
    updates = {a: [] for a in ARMS}; changes = {a: [] for a in ARMS}; checks = 0
    for i, r in enumerate(outcomes):
        sid, flag, down = r['shot'], r['is_disrupt'], r['down']
        with np.load(root/'shots'/f'{sid}.npz') as z, np.load(CAL/'calibration'/f'{sid}.npz') as old:
            assert float(np.max(np.abs(z['off']-old['mapped_fixed']))) == READ(root/'backend_comparison.json')[str(sid)]
            # Reproduce the data-contract time exclusion independently.
            valid = ((down-z['times'])*1000 >= -1e-6) if flag else np.ones(len(z['times']), bool)
            t = z['times'][valid]; lead = np.round((down-t)*1000., 9) if flag else None
            for a in group:
                score = z[a][valid].astype(np.float64)
                positions = np.flatnonzero(score >= p['threshold'])
                delta = float(lead[positions[0]]) if flag and len(positions) else None
                expected = dict(shot=sid, is_disrupt=flag, alarm=bool(len(positions)), lead_ms=delta,
                                timely=bool(delta is not None and delta > 30.))
                assert expected == records[a]['predictions'][i], (a, sid)
                checks += 1
                take = (lead > 30) & (lead <= 125) if flag else None
                loss = -np.log(score[take].clip(1e-12, 1)).mean() if flag else -np.log(1-score.max().clip(0, 1-1e-12))
                group[a][flag].append(float(loss))
                if a != 'off':
                    changes[a].append(dict(shot=sid, is_disrupt=flag,
                        mean_risk_delta=float((score-z['off'][valid]).mean()),
                        timely_mean_delta=float((score[take]-z['off'][valid][take]).mean()) if flag else None,
                        peak_delta=float(score.max()-z['off'][valid].max())))
        receipt = READ(root/'receipts'/f'{sid}.json')
        for a in ARMS:
            assert all(v['last_observation']+1 == v['first_affected_prediction'] for v in receipt[a])
            updates[a].extend(receipt[a])
    for a in group:
        m = done['metrics'][a]
        d, s = np.mean(group[a][True]), np.mean(group[a][False])
        np.testing.assert_allclose([d, s, (d+s)/2], [m['danger_loss'], m['safe_loss'], m['balanced_loss']], atol=2e-15, rtol=0)
        predictions = records[a]['predictions']
        assert sum(r['timely'] for r in predictions) == m['tp']
        assert sum(not r['is_disrupt'] and r['alarm'] for r in predictions) == m['fp']
    torch.set_num_threads(2); torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True); torch.backends.cuda.matmul.allow_tf32 = False
    spaces = torch.load(root/'spaces.pt', weights_only=True, map_location='cpu')
    g = torch.load(PAIR/'pair_gradients.pt', weights_only=True, map_location='cpu')['paired'].double()
    with np.load(PREVIOUS/'bank.npz') as z:
        shots, labels = z['shots'], z['labels']
    ids = np.unique(shots); classes = np.array([labels[shots == sid][0] for sid in ids])
    means = torch.stack([g[shots == sid].mean(0) for sid in ids]); means /= means.norm(dim=1, keepdim=True)
    centered = means.clone()
    for y in (0, 1):
        centered[classes == y] -= centered[classes == y].mean(0)
    geometry = {}
    for name, matrix in (('global', means), ('contrast', centered)):
        q = spaces[name].double()
        gram_error = float((q@q.T-torch.eye(len(q))).abs().max())
        residual = float((matrix-matrix@q.T@q).norm()/matrix.norm())
        assert gram_error < 1e-7 and residual < 1e-7
        s = torch.linalg.svdvals(matrix)
        assert len(q) == int((s > s[0]*1e-7).sum())
        geometry[name] = dict(rank=len(q), gram_error=gram_error, reconstruction_relative_error=residual)
    # Subtracting class means from rows does NOT make the resulting span
    # orthogonal to these means. Measure this rather than claiming separation.
    q = spaces['contrast'].double()
    geometry['class_mean_energy_retained'] = {
        str(y): float((means[classes == y].mean(0)@q.T).square().sum()/means[classes == y].mean(0).square().sum())
        for y in (0, 1)}
    parent, _ = make_parent(); model = models_for_calibration(parent)['mapped_fixed']; original = state_cpu(model)
    mapping = READ(REFERENCE/'input_map.json'); cuda_spaces = {k: v.cuda() for k, v in spaces.items()}
    def deny(*args, **kwargs):
        raise AssertionError('no outcomes during deterministic replay')
    replayed = []
    with patch.object(io23, 'read_outcome', side_effect=deny):
        for sid in p['shot_ids'][:2]:
            raw = signal(sid); mapped = transform({k: raw[k] for k in ('values', 'available')}, mapping)
            curves, receipt = predict(model, mapped, cuda_spaces)
            legacy = predict_many(model, {'parent': mapped}, device='cuda')['parent']
            with np.load(CAL/'calibration'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(legacy, z['mapped_fixed'])
            with torch.no_grad():
                independent = predict_many(model, {'parent': mapped}, device='cuda')['parent']
            np.testing.assert_array_equal(curves['off'], independent)
            with np.load(root/'shots'/f'{sid}.npz') as z:
                for a, v in curves.items():
                    np.testing.assert_array_equal(v, z[a])
            assert receipt == READ(root/'receipts'/f'{sid}.json')
            replayed.append(sid)
    summary = {}
    for a, rs in updates.items():
        active = [r for r in rs if r['gradient_steps']]
        assert len(active) == done['temporary_updates'][a]
        summary[a] = dict(updates=len(active),
            signal_loss_decreased=sum(r['real_signal_after'] < r['real_signal_before'] for r in active),
            own_loss_decreased=sum(r['loss_after'] < r['loss_before'] for r in active),
            median_gradient_norm=float(np.median([r['raw_gradient_norm'] for r in active])),
            median_retained_gradient_energy=float(np.median([(r['projected_gradient_norm']/r['raw_gradient_norm'])**2 for r in active])),
            median_step_norm=float(np.median([r['parameter_delta_norm'] for r in active])),
            median_signal_loss_delta=float(np.median([r['real_signal_after']-r['real_signal_before'] for r in active])))
    assert_unchanged(model, original); assert sha(DEFAULT_PARENT) == PARENT_SHA256
    verification = dict(completed_sha256=sha(root/'completed.json'), audit_sha256=sha(Path(__file__)),
        independent_shot_decisions=checks, independent_group_losses=3*len(group), exact_replay_shots=replayed,
        original_source_unchanged=True, fixed_parent_unchanged=True, geometry=geometry)
    _durable_json(root/'verification.json', verification)
    _durable_json(root/'diagnostic.json', dict(scope='POSTHOC_DEVELOPMENT_NOT_NEW_METHOD_SELECTION',
        learning=summary, risk_changes=changes, per_shot_group_losses=group))
    print(verification, flush=True); print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
