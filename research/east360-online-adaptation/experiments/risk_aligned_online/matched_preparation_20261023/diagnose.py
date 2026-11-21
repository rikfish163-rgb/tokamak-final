"""Posthoc operating curves and early/cohort error locations, no model changes."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..labeled_alignment_20261023.posthoc import main as operating_diagnosis
from ..shot_objective_alignment_20261023.cohorts import class_ranges, consecutive_cohorts
from ..simple_online import _durable_json
from ..simple_train import sha


def main(root):
    read = lambda p: json.loads(p.read_text())
    protocol, verified = read(root/'protocol.json'), read(root/'verification.json')
    assert verified['status'] == 'VERIFIED' and sha(root/'completed.json') == verified['completed_sha256']
    operating_diagnosis(root)
    full = read(root/'per_shot.json'); early = {a:[] for a in full}
    for sid in protocol['contract']['shot_ids']:
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
            for a,result in full.items():
                rows = {r['shot']:r for r in result['predictions']}
                if not rows[sid]['is_disrupt'] and np.any(z[a][:256] >= protocol['threshold']):
                    early[a].append(sid)
    old = read(Path(protocol['reference'])/'completed.json')['metrics']
    done = read(root/'completed.json')['metrics']
    selected = read(root/'selection.json')
    result = dict(scope='POSTHOC DESCRIPTIVE ANALYSIS; not a fit or independent confirmation',
        code_sha256=sha(Path(__file__)),verification_sha256=sha(root/'verification.json'),
        per_shot_sha256=sha(root/'per_shot.json'),
        preparation_ranges=class_ranges(selected['selected']),
        first256_false_alarm_ids=early,
        cohorts={a:consecutive_cohorts(r['predictions'],early[a]) for a,r in full.items()},
        old_to_new_metric_change={a:{k:done[a][k]-old[a][k] for k in ('tp','fp','tpr','fpr','J')} for a in full})
    _durable_json(root/'cohort_diagnostic.json',result)
    print(json.dumps(dict(early_false_alarm_counts={a:len(ids) for a,ids in early.items()},
        metric_change=result['old_to_new_metric_change'])),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root',type=Path); main(p.parse_args().root)
