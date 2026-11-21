"""Post-run TRAIN-only signal inventory. No repair, fitting or outcome access."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..simple_online import _durable_json
from ..simple_train import sha
from .prefix import summarize_prefix


def summarize_record(signal):
    prefix = summarize_prefix(signal)
    values, available = signal['values'], signal['available']
    channels = {}
    for column, name in enumerate(io23.RAW23):
        x = values[available[:, column], column]
        channels[name] = dict(observed_points=int(len(x)),
            quantiles=np.quantile(x, [0., .1, .5, .9, 1.]).tolist() if len(x) else None,
            negative_fraction=float(np.mean(x < 0)) if len(x) else None,
            zero_fraction=float(np.mean(x == 0)) if len(x) else None)
    return dict(start_time=float(signal['times'][0]), end_time=float(signal['times'][-1]),
        recorded_points=len(values), quantile_levels=[0., .1, .5, .9, 1.],
        prefix100ms=prefix, channels=channels)


def main(root):
    read = lambda p: json.loads(p.read_text())
    verified, protocol = read(root/'verification.json'), read(root/'protocol.json')
    if verified['status'] != 'VERIFIED' or sha(root/'completed.json') != verified['completed_sha256']:
        raise ValueError('verified immutable completed run required')
    selection = read(root/'selection.json')
    ids = sorted(set(protocol['selected_ids']) | set(selection['old_selected_ids']))
    if len(ids) != 34 or set(ids) & set(protocol['contract']['shot_ids']):
        raise ValueError('expected exactly34 old/new preparation shots disjoint from TEST')
    records = {}
    for sid in ids:
        path = io23.path_for('EAST', sid)
        signal = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        row = summarize_record(signal)
        # Inventory attribute NAMES only. Never read /meta or infer a unit from
        # the column name. File hashing is byte-level provenance, not label parsing.
        with h5py.File(path, 'r') as f:
            row['root_attribute_names'] = sorted(f.attrs.keys())
            row['data_attribute_names'] = sorted(f['data'].attrs.keys())
            row['channel_attribute_names'] = {n: sorted(f['data'][n].attrs.keys()) for n in io23.RAW23}
        row.update(file_path=str(path), file_sha256=sha(path))
        records[str(sid)] = row
    result = dict(scope='POSTHOC TRAIN preparation records only; whole-record summaries are not online inputs',
        no_model_or_data_changes=True, no_test_signals_read=True, no_new_outcome_reads=True,
        input_contract='unchanged simple_data + causal_input; no empirical ne_nG gain applied in this inventory',
        interpretation='Stored numerical values, not certified physical units or diagnosis of acquisition failure',
        code_sha256=sha(Path(__file__)), verification_sha256=sha(root/'verification.json'),
        records=records)
    _durable_json(root/'preparation_record_diagnostic.json', result)
    print(json.dumps(dict(shots=len(records), output=str(root/'preparation_record_diagnostic.json'),
        bt_medians={s: r['channels']['Bt']['quantiles'][2] for s,r in records.items()})), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
