"""Post-result attribution of arrival feedback; no fitting or neural updates."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..io23 import RAW23
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import summarize_arrival
from .verify import load, read


def contributions(fit, target, valid):
    coefficient = np.asarray(fit['q1_coef'], float)
    scale = np.asarray(fit['arrival_scale'], float)
    if (coefficient.shape != (180,) or scale.shape != (92,) or (scale <= 0).any()
            or not np.isfinite(coefficient).all() or not np.isfinite(scale).all()):
        raise ValueError('finite H64+M23+D92 coefficients and positive scales required')
    return (summarize_arrival(target, valid)/scale*coefficient[87:-1]).reshape(-1, 23, 4)


def diagnose(directory):
    protocol = read(directory/'protocol.json')
    cache = Path(protocol['cache'])
    observations = load(cache/'observations.npz')
    outcomes = read(cache/'outcomes.json')
    index = {(sid, end): i for i, (sid, end) in enumerate(
        zip(observations['ids'], observations['ends'], strict=True))}
    rows, discrepancies, fit_counts = [], [], []
    for spec in protocol['folds']:
        folder = directory/f"fold{spec['fold']}"
        fit = torch.load(folder/'real_pair.pt', weights_only=True, map_location='cpu')
        assert set(fit['fit_ids']) == set(spec['fit'])
        fit_counts.append(dict(fold=spec['fold'], label_shots=fit['label_shot_counts'],
                               label_windows=fit['label_window_counts']))
        data = load(folder/'neural.npz')
        assert sha(folder/'neural.npz') == read(folder/'neural_seal.json')['sha256']
        sealed = load(folder/'sealed.npz')
        assert sha(folder/'sealed.npz') == read(folder/'sealed.json')['sha256']
        sealed_index = {(sid, end): i for i, (sid, end) in enumerate(zip(sealed['ids'], sealed['ends'], strict=True))}
        joined = np.array([index[sid, end] for sid, end in zip(data['ids'], data['ends'], strict=True)])
        score_join = np.array([sealed_index[sid, end] for sid, end in zip(data['ids'], data['ends'], strict=True)])
        parts = contributions(fit, observations['target'][joined], observations['valid'][joined])
        evidence = parts.sum((1, 2))
        discrepancies.append(float(np.max(np.abs(evidence-sealed['real_potentials'][score_join, 1]))))
        for sid in spec['held']:
            selected = data['ids'] == sid
            times = data['future_times'][selected]
            flag = outcomes[str(sid)]['disruptive']
            if flag:
                lead = (outcomes[str(sid)]['down']-times)*1000
                masks = dict(danger_timely=(lead <= 125+1e-6) & (np.round(lead, 9) > 30),
                             early_normal=lead > 125+1e-6)
            else:
                masks = {'safe': np.ones_like(times, bool)}
            for group, mask in masks.items():
                n = mask.sum(1)
                if not n.sum():
                    continue
                avg = np.einsum('n,ncs->cs', n/n.sum(), parts[selected])
                positive = float(np.average(evidence[selected] > 0, weights=n))
                rows.append(dict(shot=sid, group=group, contributions=avg.tolist(),
                                 mean_increment=float(avg.sum()), positive_fraction=positive))
    if max(discrepancies) > 1e-10:
        raise AssertionError('attribution does not reconstruct sealed increment')
    groups = {}
    for group in ('safe', 'danger_timely', 'early_normal'):
        selected = [r for r in rows if r['group'] == group]
        avg = np.mean([r['contributions'] for r in selected], axis=0)
        channels = avg.sum(1)
        groups[group] = dict(shots=len(selected), mean_increment=float(avg.sum()),
                            mean_positive_fraction=float(np.mean([r['positive_fraction'] for r in selected])),
                            by_statistic=dict(zip(('mean_change', 'last_minus_first', 'centered_rms', 'last_change'),
                                                  avg.sum(0).tolist(), strict=True)),
                            by_channel=dict(zip(RAW23, channels.tolist(), strict=True)))
    result = dict(groups=groups, fit_counts=fit_counts, per_shot=rows,
                  max_reconstruction_difference=max(discrepancies),
                  diagnostic_only=True, used_for_training=False,
                  code_sha256=sha(Path(__file__)),
                  warning='Descriptive contribution, not causal physical importance or an intervention experiment.')
    _durable_json(directory/'feedback_diagnosis.json', result)
    print(json.dumps({k: result[k] for k in ('groups', 'fit_counts', 'max_reconstruction_difference')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    diagnose(parser.parse_args().directory)
