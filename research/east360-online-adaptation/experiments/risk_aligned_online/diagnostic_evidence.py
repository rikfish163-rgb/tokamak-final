"""Read sealed paired forecasts and audit selective within-shot effects.

Outcome labels here are retrospective EVALUATION only. Timely evaluation
segments do not establish a physical instability onset or a causal mechanism.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import io23, simple_data
from .simple_online import build_order


def parent_scale_audit():
    groups = [('J-TEXT', simple_data.source_roles()['train'][:12]),
              ('EAST', [row['shot'] for row in build_order('EAST', io23.cohort_ids('EAST'))[:12]])]
    records = []
    for domain, ids in groups:
        for sid in ids:
            shot = simple_data.read_shot(domain, sid)
            x, mask = shot['values'], shot['available']
            ip, voltage, impedance = x[:, 9], x[:, 18], x[:, 6]
            eligible = mask[:, 9] & mask[:, 18] & mask[:, 6] & (np.abs(ip) > 10000) & (np.abs(voltage) > .1)
            ratio = impedance[eligible]*ip[eligible]/voltage[eligible]
            records.append(dict(device=domain, shot=sid, signal_sha256=hashlib.sha256(
                np.ascontiguousarray(x).tobytes()+np.ascontiguousarray(mask).tobytes()).hexdigest(),
                recorded_medians={name: float(np.median(x[mask[:, i], i])) if mask[:, i].any() else None
                                  for i, name in enumerate(io23.RAW23)},
                Z_ip_over_v_loop_percentiles=np.quantile(ratio, [.1, .5, .9]).tolist() if len(ratio) else None))
    return dict(scope='12 training-source shots and first12 EAST proxy-order shots; signal-only descriptive sample',
                limitations='Different stored magnitudes are not proof of a known unit conversion; no physical unit inferred.',
                records=records)


def paired_evidence(run):
    summary = json.loads((run/'summary.json').read_text())
    predictions = {row['shot']: row for row in summary['static']['predictions']}
    stats = {'safe': [], 'disrupt_30to125ms': [], 'disrupt_earlier_than125ms': []}
    checked = 0
    for receipt_path in sorted((run/'static').glob('*_seal.json')):
        receipt = json.loads(receipt_path.read_text())
        original = receipt_path.parent/receipt['npz']
        other_receipt_path = run/'intra'/receipt_path.name
        other_receipt = json.loads(other_receipt_path.read_text())
        other = other_receipt_path.parent/other_receipt['npz']
        for path, info in ((original, receipt), (other, other_receipt)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
                raise ValueError(f'prediction seal changed: {path}')
        with np.load(original) as before, np.load(other) as after:
            np.testing.assert_array_equal(before['times'], after['times'])
            sid = int(receipt_path.name.split('_')[1])
            disruptive = predictions[sid]['is_disrupt']
            # Read outcome AFTER all forecasts are sealed; evaluator only.
            actual_disruptive, down = io23.read_outcome('EAST', sid)
            if actual_disruptive != disruptive:
                raise ValueError('outcome mismatch in retrospective audit')
            delta = after['scores'].astype(float)-before['scores'].astype(float)
            if not disruptive:
                masks = {'safe': np.ones(len(delta), bool)}
            else:
                lead = np.round((down-before['times'])*1000, 9)
                masks = {'disrupt_30to125ms': (lead > 30) & (lead <= 125),
                         'disrupt_earlier_than125ms': lead > 125}
            for name, mask in masks.items():
                if mask.any():
                    stats[name].append(dict(shot=sid, endpoints=int(mask.sum()),
                                            mean_delta=float(delta[mask].mean()),
                                            mean_absolute_delta=float(np.abs(delta[mask]).mean())))
        checked += 2
    return dict(prediction_seals_verified=checked,
                phase_definition='Evaluation segments, NOT inferred physical onset. Means weight shots equally.',
                phases={name: dict(shots=len(rows),
                                   mean_probability_change=float(np.mean([row['mean_delta'] for row in rows])),
                                   mean_absolute_probability_change=float(np.mean([row['mean_absolute_delta'] for row in rows])),
                                   records=rows) for name, rows in stats.items()},
                full_results={arm: {k: v for k, v in summary[arm].items() if k != 'predictions'}
                              for arm in ('static', 'intra')},
                changed_shots=summary['changed_shots'], scale_audit=parent_scale_audit())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = paired_evidence(args.run)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    compact = dict(prediction_seals_verified=result['prediction_seals_verified'],
                   full_results=result['full_results'],
                   phases={name: {key: value for key, value in phase.items() if key != 'records'}
                           for name, phase in result['phases'].items()})
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == '__main__':
    main()
