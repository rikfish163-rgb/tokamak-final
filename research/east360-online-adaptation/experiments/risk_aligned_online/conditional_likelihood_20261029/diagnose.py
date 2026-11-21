"""Retrospective attribution of a failed density; no fitting or policy selection."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from ..io23 import RAW23
from ..readout_identifiability_20261029.learning import predict
from ..simple_train import sha
from .run import read, training_rows


def terms(target, mean, chol):
    target, mean, chol = map(lambda a: np.asarray(a, dtype=np.float64), (target, mean, chol))
    if (target.ndim != 2 or target.shape != mean.shape or target.shape[1] != 25
            or chol.shape != (25, 25) or not all(np.isfinite(a).all() for a in (target, mean, chol))):
        raise ValueError('finite paired N25 arrays and 25x25 Cholesky required')
    residual = np.linalg.solve(chol, (target-mean).T).T
    distance = np.square(residual).sum(1)
    normalizer = math.lgamma(14.5)-math.lgamma(2)-12.5*math.log(4*math.pi)
    normalizer -= np.log(np.diag(chol)).sum()
    tail = -14.5*np.log1p(distance/4)
    return normalizer, tail, distance


def average_by_shot(values, ids):
    return float(np.mean([np.mean(values[ids == s]) for s in np.unique(ids)]))


def main(source, output):
    if output.exists():
        raise FileExistsError(output)
    protocol = read(source/'protocol.json')
    outcomes = {int(k): (v['disruptive'], v['down']) for k, v in read(source/'outcomes.json').items()}
    with np.load(source/'observations.npz', allow_pickle=False) as archive:
        data = dict(archive)
    rows, per_shot = [], []
    for spec in protocol['folds']:
        fold = spec['fold']
        fit = torch.load(source/f'fold{fold}'/'real_density.pt', weights_only=True, map_location='cpu')
        take, labels = training_rows(data, {s: outcomes[s] for s in spec['fit']})
        held = np.isin(data['ids'], spec['held'])
        train = {k: a[take] for k, a in data.items()}
        query = {k: a[held] for k, a in data.items()}
        _, query_labels = training_rows(query, {s: outcomes[s] for s in spec['held']})
        # Keep postterminal observations for score reconstruction, never for class evaluation.
        query_valid, _ = training_rows(query, {s: outcomes[s] for s in spec['held']})
        qlabels = np.full(len(query['ids']), -1)
        qlabels[query_valid] = query_labels
        normalizers = np.zeros((len(query['ids']), 23, 2))
        tails = normalizers.copy()
        observed = query['valid'] & np.isfinite(query['target']).all(1)
        for channel, channel_fit in enumerate(fit['channels']):
            if not channel_fit['supported']:
                observed[:, channel] = False
                continue
            qm = observed[:, channel]
            for label, component in enumerate(channel_fit['classes']):
                reg = {k: v.numpy() if isinstance(v, torch.Tensor) else v
                       for k, v in component['ridge'].items()}
                tm = train['valid'][:, channel] & (labels == label)
                tx, ty = train['x'][tm], train['target'][tm, :, channel]
                qx, qy = query['x'][qm], query['target'][qm, :, channel]
                tn, tt, td = terms(ty, predict(reg, tx), component['t_cholesky'])
                qn, qt, qd = terms(qy, predict(reg, qx), component['t_cholesky'])
                normalizers[qm, channel, label], tails[qm, channel, label] = qn, qt
                correct = qlabels[qm] == label
                if not correct.any():
                    continue
                train_error = np.square(ty-predict(reg, tx)).mean(1)
                query_error = np.square(qy-predict(reg, qx)).mean(1)
                rows.append(dict(fold=fold, channel=RAW23[channel], label=label,
                    train_rows=int(tm.sum()), train_shots=component['nshots'],
                    held_rows=int(correct.sum()),
                    train_mse=average_by_shot(train_error, train['ids'][tm]),
                    held_mse=average_by_shot(query_error[correct], query['ids'][qm][correct]),
                    train_mahalanobis=average_by_shot(td, train['ids'][tm]),
                    held_mahalanobis=average_by_shot(qd[correct], query['ids'][qm][correct]),
                    scale_trace=float(torch.trace(component['t_scale'])),
                    train_x_rms=float(np.sqrt(np.mean(np.square((tx-reg['mean'])/reg['scale'])))),
                    held_x_rms=float(np.sqrt(np.mean(np.square((qx[correct]-reg['mean'])/reg['scale']))))))
        with np.load(source/f'fold{fold}'/'sealed.npz', allow_pickle=False) as archive:
            np.testing.assert_allclose((normalizers+tails).sum(1), archive['real_logs'], atol=1e-9, rtol=1e-12)
        norm_diff = normalizers[:, :, 1]-normalizers[:, :, 0]
        tail_diff = tails[:, :, 1]-tails[:, :, 0]
        for sid in spec['held']:
            flag, down = outcomes[sid]
            own = (query['ids'] == sid) & query_valid
            masks = {'safe': own} if not flag else {
                'danger_timely': own & (qlabels == 1) & (np.round((down-query['arrival_time'])*1000, 9) > 30),
                'early_normal': own & (qlabels == 0),
                'danger_late': own & (qlabels == 1) & (np.round((down-query['arrival_time'])*1000, 9) <= 30)}
            for group, mask in masks.items():
                if not mask.any():
                    continue
                per_shot.append(dict(shot=sid, group=group, windows=int(mask.sum()),
                    normalizer=float(norm_diff[mask].sum(1).mean()),
                    residual_tail=float(tail_diff[mask].sum(1).mean()),
                    channels=[dict(channel=RAW23[c],
                        normalizer=float(norm_diff[mask, c].mean()),
                        residual_tail=float(tail_diff[mask, c].mean())) for c in range(23)
                        if observed[mask, c].any()]))
    groups = {}
    for group in sorted({r['group'] for r in per_shot}):
        rr = [r for r in per_shot if r['group'] == group]
        groups[group] = dict(shots=len(rr), normalizer=np.mean([r['normalizer'] for r in rr]),
                            residual_tail=np.mean([r['residual_tail'] for r in rr]))
    result = dict(scope='RETROSPECTIVE_DENSITY_FAILURE_DIAGNOSIS', groups=groups,
        covariance_model='Student-t scale; covariance would be 2*scale for df4',
        source_completed_sha256=sha(source/'completed.json'),
        source_observations_sha256=sha(source/'observations.npz'),
        code_sha256=sha(Path(__file__)), per_channel_fit=rows, per_shot=per_shot,
        changes_to_predictions=False)
    output.mkdir(parents=False)
    (output/'diagnosis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(groups, indent=2))
    for label in (0, 1):
        rr = [r for r in rows if r['label'] == label]
        print(json.dumps(dict(label=label, rows_range=[min(r['train_rows'] for r in rr), max(r['train_rows'] for r in rr)],
            median_mse_ratio=np.median([r['held_mse']/max(r['train_mse'], 1e-30) for r in rr]),
            median_held_mahalanobis=np.median([r['held_mahalanobis'] for r in rr]),
            median_train_mahalanobis=np.median([r['train_mahalanobis'] for r in rr]))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    main(args.source, args.output)
