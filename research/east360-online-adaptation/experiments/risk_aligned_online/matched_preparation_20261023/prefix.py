"""Describe actual preparation signals; no sampling, fitting, or TEST signal reads."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..simple_online import _durable_json
from ..simple_train import sha


def summarize_prefix(signal, milliseconds=100.):
    times = np.asarray(signal['times']); values = np.asarray(signal['values'])
    available = np.asarray(signal['available'])
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError('finite strictly increasing actual times required')
    if (values.shape != (len(times),23) or available.shape != values.shape
            or available.dtype != bool or not np.isfinite(values[available]).all()
            or not np.isfinite(milliseconds) or milliseconds <= 0):
        raise ValueError('valid23 observations, masks and positive duration required')
    # Only diagnostic endpoint comparison is rounded, to 1e-9 ms. Raw model
    # timestamps and signals are untouched, as in the main metric contract.
    chosen = np.round((times-times[0])*1000.,9) < milliseconds
    channels = {}
    for column, name in enumerate(io23.RAW23):
        valid = chosen & available[:,column]; x = values[valid,column]
        channels[name] = dict(available_fraction=float(valid.sum()/chosen.sum()),
            median=float(np.median(x)) if len(x) else None,
            minimum=float(x.min()) if len(x) else None,maximum=float(x.max()) if len(x) else None)
    return dict(observed_points=int(chosen.sum()),duration_ms=milliseconds,
                start_time=float(times[0]),channels=channels)


def main(root):
    read = lambda p: json.loads(p.read_text())
    assert read(root/'verification.json')['status'] == 'VERIFIED'
    p = read(root/'protocol.json'); new = read(root/'selection.json')
    old_path = Path(p['reference'])/'selection.json'; old = read(old_path)
    assert sha(old_path) == p['reference_hashes']['selection.json']
    ids = sorted(set(p['selected_ids'])|set(new['old_selected_ids']))
    assert len(ids) == 34 and not set(ids)&set(p['contract']['shot_ids'])
    per_shot = {}
    for sid in ids:
        signal = apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        per_shot[str(sid)] = summarize_prefix(signal)
    groups = {}
    for name, rows in (('old',old['selected']),('new',new['selected'])):
        for d, label in ((False,'safe'),(True,'disruptive')):
            selected_ids = [r['shot'] for r in rows if r['is_disrupt'] == d]
            summary = {}
            for feature in io23.RAW23:
                raw = [per_shot[str(s)]['channels'][feature] for s in selected_ids]
                medians = [r['median'] for r in raw if r['median'] is not None]
                summary[feature] = dict(shots_observed=len(medians),
                    median_of_shot_medians=float(np.median(medians)) if medians else None,
                    minimum_shot_median=float(min(medians)) if medians else None,
                    maximum_shot_median=float(max(medians)) if medians else None,
                    mean_available_fraction=float(np.mean([r['available_fraction'] for r in raw])))
            groups[name+'_'+label] = dict(shot_ids=selected_ids,features=summary)
    result = dict(scope='Post-run preparation-signal description, not used to select or retrain',
        time_scope='first100ms from available HDF signal start, not certified plasma startup or discharge plan',
        units='existing HDF channel values after unchanged input contract; ne_nG physical units not certified',
        no_test_signals_read=True,no_new_outcome_reads=True,code_sha256=sha(Path(__file__)),
        verification_sha256=sha(root/'verification.json'),groups=groups,per_shot=per_shot)
    _durable_json(root/'preparation_prefix_diagnostic.json',result)
    print(json.dumps({k:{f:v['features'][f] for f in ('ip','Bt','ne_nG','qa_proxy')}
                      for k,v in groups.items()}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root',type=Path); main(p.parse_args().root)
