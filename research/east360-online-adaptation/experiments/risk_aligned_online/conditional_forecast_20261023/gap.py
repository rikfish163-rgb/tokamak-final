"""Post-hoc fit/held gap; does not select hyperparameters or change models."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import io23, simple_data
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import maturity_error
from .run import forecast


def main(root, *, matched_time=False):
    output = root/('fit_held_gap_matched.json' if matched_time else 'fit_held_gap.json')
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads((root/'protocol.json').read_text())
    rows = []
    for fold in protocol['folds']:
        directory = root/f'fold{fold["fold"]}'
        with np.load(directory/'heads.npz') as heads, np.load(directory/'fit.npz') as data:
            # Positive origins only: labels already authorized and used in fitting.
            select = data['y'] == 1
            features, target, valid, ids = [data[k][select] for k in ('features', 'target', 'valid', 'ids')]
            origins = data['origins'][select]
            clocks = {}
            if matched_time:
                for sid in np.unique(ids):
                    raw = simple_data.read_shot('EAST', int(sid), rate_contract='nominal')
                    disruptive, down = io23.read_outcome('EAST', int(sid))
                    assert disruptive and int(sid) in fold['fit']
                    clocks[int(sid)] = (raw['times'], down)
            predictions = forecast(features, heads)
            for h in (5, 10, 25):
                error = {name: maturity_error(p, target[:, :h], valid[:, :h], heads['covariance'], heads['active'])
                         for name, p in dict(predictions, persistence=np.zeros_like(predictions['real0'])).items()}
                for sid in np.unique(ids):
                    m = ids == sid
                    if matched_time:
                        ts, down = clocks[int(sid)]
                        relevant = np.flatnonzero(m)
                        lead = np.round((down-ts[origins[relevant]+h])*1000, 9)
                        m[relevant[lead <= 30]] = False
                    score = error['real0'][m]-error['real1'][m]
                    rows.append(dict(fold=fold['fold'], shot=int(sid), role='fit_positive_in_sample', horizon=h,
                        rows=int(m.sum()), correct_fraction=float((score > 0).mean()),
                        median_direction=float(np.median(score)),
                        mean_errors={name: float(e[m].mean()) for name, e in error.items()}))
    held = [r for r in json.loads((root/'per_shot.json').read_text()) if r['disruptive'] and r['arm'] == 'real']
    report = dict(posthoc=True, used_for_model_selection=False, matched_feedback_lead_gt30=matched_time,
        evidence='fit/held gap, NOT independent validation',
        completed_sha256=sha(root/'completed.json'), code_sha256=sha(Path(__file__)), training=rows,
        held=[dict(shot=r['shot'], horizon=r['horizon'], n=r['critical']['n'], median_direction=r['critical']['median'],
                   correct_fraction=r['critical']['positive_fraction'],
                   mean_errors={k: r[k]['mean'] for k in ('safe_error', 'danger_error', 'unconditional', 'persistence')}) for r in held])
    _durable_json(output, report)
    for h in (5, 10, 25):
        train = [r for r in rows if r['horizon'] == h]; test = [r for r in held if r['horizon'] == h]
        print(dict(horizon=h, train_shot_fold_positive_direction=sum(r['median_direction'] > 0 for r in train),
                   train_shot_fold_count=len(train), held_positive_direction=sum(r['critical']['median'] > 0 for r in test)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    parser.add_argument('--matched-time', action='store_true')
    args = parser.parse_args(); main(args.root, matched_time=args.matched_time)
