"""Post-hoc valid-time curve comparison; no new fitting or threshold selection."""
import argparse
from pathlib import Path

import numpy as np

from .. import simple_data
from ..diverse_support_20261023.run import READ
from ..simple_online import _durable_json
from ..simple_train import sha
from .run import REFERENCE


def main(root):
    output = root / 'curve_diagnosis.json'
    if output.exists():
        raise FileExistsError(output)
    done = READ(root / 'completed.json')
    for name, digest in done['artifact_sha256'].items():
        assert sha(root / name) == digest, name
    protocol = READ(root / 'protocol.json')
    rows = READ(root / 'per_shot.json')
    differences, cases = [], []
    norms = {'real': [], 'sham': []}
    for sid in protocol['shot_ids']:
        shot = simple_data.attach_outcome(
            simple_data.read_shot('EAST', sid, rate_contract='nominal'), 'EAST', {})
        valid = shot['valid']
        with np.load(root / 'shots' / f'{sid}.npz') as saved:
            a, b = saved['real_intra'][valid], saved['persistence_sham'][valid]
            difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
            differences.append(dict(shot=sid, max_absolute=float(difference.max()),
                                    mean_absolute=float(difference.mean())))
            if sid in (62765, 87537):
                cases.append(dict(shot=sid, is_disrupt=shot['is_disrupt'],
                    valid_peak={arm: float(saved[arm][valid].max())
                                for arm in ('static', 'real_intra', 'persistence_sham')},
                    real_sham_max_absolute=float(difference.max())))
        real = READ(REFERENCE / 'fold0/shots' / f'{sid}_updates.json')['mapped_fixed']
        sham = READ(root / 'shots' / f'{sid}_updates.json')
        assert len(real) == len(sham)
        for name, updates in (('real', real), ('sham', sham)):
            norms[name].extend(u['gradient_norm'] for u in updates)
    paired = {}
    for level in ('calibrated', 'source_threshold'):
        a = {r['shot']: r for r in rows['real_intra/' + level]['predictions']}
        b = {r['shot']: r for r in rows['persistence_sham/' + level]['predictions']}
        paired[level] = dict(
            alarm_difference=sum(a[s]['alarm'] != b[s]['alarm'] for s in a),
            timely_difference=sum(a[s]['timely'] != b[s]['timely'] for s in a),
            lead_difference=sum(a[s]['lead_ms'] != b[s]['lead_ms'] for s in a))
    result = dict(
        status='POSTHOC_VALID_TIME_COMPARISON',
        completed_sha256=sha(root / 'completed.json'),
        code_sha256=sha(Path(__file__)),
        paired=paired, changed_cases=cases,
        valid_time_max_absolute=max(r['max_absolute'] for r in differences),
        median_shot_max_absolute=float(np.median([r['max_absolute'] for r in differences])),
        equal_shot_mean_absolute=float(np.mean([r['mean_absolute'] for r in differences])),
        gradient_norm={k: dict(count=len(v), median=float(np.median(v)),
                              maximum=float(np.max(v))) for k, v in norms.items()},
        per_shot_curve_differences=differences,
        note='Post hoc, not a new experiment. Evaluation-valid prefixes only; no post-disruption samples. '
             'Equal shot weights, not a pooled timepoint mean. No threshold/parameter changes.')
    _durable_json(output, result)
    print({k: v for k, v in result.items() if k != 'per_shot_curve_differences'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
