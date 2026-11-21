"""Signal-only retrospective audit; no outcome reads and no model updates.

Exact constancy is a measured property, NOT a missing-code determination.
Record summaries are not online features. Prefix means first recorded samples,
not physical discharge startup. Quantiles use <=128 equally spaced observations
per shot; pooled across-shot summaries therefore give each shot one vote.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import io23, simple_data
from .simple_online import _durable_json, _hash


def describe_channel(values, available, *, center, scale):
    values, available = np.asarray(values), np.asarray(available)
    if (values.ndim != 1 or not len(values) or available.shape != values.shape
            or available.dtype != bool or not np.isfinite([center, scale]).all() or scale <= 0):
        raise ValueError('nonempty channel, boolean availability and finite positive scale required')
    mask = available & np.isfinite(values)
    observed = values[mask].astype(np.float64)
    prefix = values[:100][mask[:100]]
    result = dict(n_recorded=len(values), n_observed=len(observed),
                  available_fraction=float(mask.mean()),
                  record_constant=bool(np.all(observed == observed[0])) if len(observed) else None,
                  prefix100_constant=bool(np.all(prefix == prefix[0])) if len(prefix) else None,
                  sampled_points=0, quantiles=None, median_source_z=None,
                  iqr_source_scale=None, fraction_abs_source_z_over10=None)
    if not len(observed):
        return result
    sample = observed[np.linspace(0, len(observed)-1, min(128, len(observed)), dtype=int)]
    q = np.quantile(sample, [0., .25, .5, .75, 1.])
    result.update(sampled_points=len(sample), quantiles=q.tolist(),
                  median_source_z=float((q[2]-center)/scale),
                  iqr_source_scale=float((q[3]-q[1])/scale),
                  fraction_abs_source_z_over10=float((np.abs((sample-center)/scale) > 10).mean()))
    return result


def summarize(rows):
    result = []
    for c, name in enumerate(io23.RAW23):
        columns = [row['channels'][c] for row in rows]
        observed = [r for r in columns if r['n_observed']]
        result.append(dict(channel=name, shots=len(rows),
            completely_missing=sum(r['n_observed'] == 0 for r in columns),
            constant_records=sum(r['record_constant'] is True for r in columns),
            constant_prefixes=sum(r['prefix100_constant'] is True for r in columns),
            median_available_fraction=float(np.median([r['available_fraction'] for r in columns])),
            median_source_z=float(np.median([r['median_source_z'] for r in observed])) if observed else None,
            median_iqr_source_scale=float(np.median([r['iqr_source_scale'] for r in observed])) if observed else None,
            mean_fraction_abs_source_z_over10=float(np.mean([
                r['fraction_abs_source_z_over10'] for r in observed])) if observed else None))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--order', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    center, scale = np.asarray(ckpt['source_center']), np.asarray(ckpt['source_scale'])
    if center.shape != (23,) or scale.shape != (23,):
        raise ValueError('flat23 source statistics required')
    ids = [int(r['shot']) for r in json.loads(args.order.read_text())]
    if len(ids) != 360 or len(set(ids)) != 360 or set(ids) != set(io23.cohort_ids('EAST')):
        raise ValueError('complete locked EAST360 order required')
    cohorts = {'source_train': ('J-TEXT', io23.source_roles()['train']),
               'east_first30': ('EAST', ids[:30]), 'east_later330': ('EAST', ids[30:])}
    payload = dict(scope='DEVELOPMENT_ONLY; offline quality audit; no outcome reads; no fitted changes',
                   checkpoint_sha256=_hash(args.checkpoint), order_sha256=_hash(args.order),
                   code_sha256={name: _hash(Path(__file__).with_name(name)) for name in
                                ('feature_quality.py', 'io23.py', 'simple_data.py')}, cohorts={})
    for role, (device, shot_ids) in cohorts.items():
        rows = []
        for i, sid in enumerate(shot_ids):
            shot = simple_data.read_shot(device, sid)
            rows.append(dict(shot=sid, start_seconds=float(shot['times'][0]), channels=[
                describe_channel(shot['values'][:, c], shot['available'][:, c],
                                 center=center[c], scale=scale[c]) for c in range(23)]))
            if (i+1) % 100 == 0:
                print(f'{role} {i+1}/{len(shot_ids)}', flush=True)
        payload['cohorts'][role] = dict(summary=summarize(rows), rows=rows)
    _durable_json(args.output, payload)
    for role, values in payload['cohorts'].items():
        print(role, json.dumps(values['summary']), flush=True)


if __name__ == '__main__':
    main()
