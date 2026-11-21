"""Read-only numeric provenance audit; no model fitting or unit correction.

Inventory all live EAST and canonical JTEXT TRAIN files. Read six full signal
columns, never outcomes. Algebraic ratios are observations, not unit claims.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from experiments.risk_aligned_online import io23


FIELDS = ('Bt', 'ip', 'qa_proxy', 'ne_nG', 'Z_proxy', 'v_loop')
OUTPUT = Path(__file__).with_name(Path(__file__).stem+'_v2.json')
Q = [0., .05, .5, .95, 1.]


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    finite = a[np.isfinite(a)]
    return dict(n=int(a.size), nonfinite=int(a.size-finite.size),
                quantiles=np.quantile(finite, Q).tolist() if finite.size else None)


def ratio(a, b):
    valid = np.isfinite(a) & np.isfinite(b) & (b != 0)
    return a[valid] / b[valid]


def audit_one(device, sid):
    p = io23.path_for(device, sid)
    with h5py.File(p, 'r') as f:
        x = {n: np.asarray(f['data'][n][:], dtype=np.float64) for n in FIELDS}
        attrs = {n: {k: np.asarray(v).tolist() for k, v in f['data'][n].attrs.items()}
                 for n in FIELDS}
        group_keys = dict(root=list(f.attrs), data=list(f['data'].attrs))
    # Same single explicit current unit operation; all other raw columns unchanged.
    ip = x['ip'] * (1000. if device == 'J-TEXT' else 1.)
    # Match commissioning_stream: serialization neighbours of 1000 Hz must
    # not turn an intended 100-sample prefix into 101 samples after ceil.
    count = min(len(ip), int(np.ceil(float(attrs['ip']['SampleRate'])*.1-1e-10)))
    q_ratio = ratio(x['qa_proxy'] * ip, x['Bt'])
    z_ratio = ratio(x['Z_proxy'] * ip, x['v_loop'])
    q_median = float(np.median(q_ratio)) if len(q_ratio) else None
    q_class = ('near_0.687754' if q_median is not None and abs(q_median-.6877540674)<.001
               else 'near_1' if q_median is not None and abs(q_median-1.)<.001
               else 'near_0.373999' if q_median is not None and abs(q_median-.3739991254)<.001
               else 'other')
    digest = hashlib.sha256()
    for n in FIELDS:
        digest.update(n.encode())
        digest.update(x[n].tobytes())
    digest.update(json.dumps(attrs, sort_keys=True).encode())
    first_bt = float(np.median(np.abs(x['Bt'][:count])))
    return dict(device=device, shot=sid, path=str(p), bytes=p.stat().st_size,
                inspected_channel_content_sha256=digest.hexdigest(),
                attrs=attrs, group_attribute_keys=group_keys,
                full_record={n: summary(x[n]) for n in FIELDS},
                prefix_samples=count, first100ms_Bt_abs_median=first_bt,
                first100ms_Ip_reader_abs_median=float(np.median(np.abs(ip[:count]))),
                Bt_group='lt0.1' if first_bt<.1 else '0.1to1' if first_bt<1 else 'ge1',
                qa_times_reader_Ip_over_Bt=summary(q_ratio), qa_numeric_convention=q_class,
                Z_times_reader_Ip_over_Vloop=summary(z_ratio))


def aggregate(rows):
    return dict(n=len(rows),
                Bt_groups=dict(Counter(r['Bt_group'] for r in rows)),
                qa_numeric_conventions=dict(Counter(r['qa_numeric_convention'] for r in rows)),
                first100ms_Bt_abs_median=summary([r['first100ms_Bt_abs_median'] for r in rows]),
                whole_record_medians={n: summary([r['full_record'][n]['quantiles'][2] for r in rows
                                                  if r['full_record'][n]['quantiles']]) for n in FIELDS},
                qa_ratio_per_shot_median=summary([r['qa_times_reader_Ip_over_Bt']['quantiles'][2]
                                                  for r in rows if r['qa_times_reader_Ip_over_Bt']['quantiles']]),
                Z_ratio_per_shot_median=summary([r['Z_times_reader_Ip_over_Vloop']['quantiles'][2]
                                                 for r in rows if r['Z_times_reader_Ip_over_Vloop']['quantiles']]))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    roles = io23.source_roles()
    target = sorted(int(p.stem) for p in (io23.DATA_ROOT/'EAST/unified_hdf5').glob('*/*.hdf5'))
    assert len(target)==len(set(target))==2953
    assert len(roles['train'])==1353
    run = Path(__file__).parent/'probe_outputs/anchored_full360_diagnostic_v1'
    protocol = json.loads((run/'protocol.json').read_text())
    order = protocol['order']
    rows = []
    for device, ids in [('J-TEXT', roles['train']), ('EAST', target)]:
        for j, sid in enumerate(ids):
            rows.append(audit_one(device, sid))
            if (j+1)%500==0:
                print(dict(device=device, done=j+1, total=len(ids)), flush=True)
    source = [r for r in rows if r['device']=='J-TEXT']
    east = [r for r in rows if r['device']=='EAST']
    current = [r for r in east if r['shot'] in set(order)]
    cold = [r for r in east if r['shot'] in set(order[:20])]
    result = dict(scope='SIGNAL_ONLY_DESCRIPTIVE_FOUNDATION_AUDIT',
                  raw_outcomes_read=False, trained=False, applied_unit_changes=False,
                  quantile_probabilities=Q, per_shot_full_signal_summaries=True,
                  channel_hash_scope='six float64 columns plus their attributes, not entire file',
                  warning='Numeric convention clusters are posthoc diagnostics, not certified units or deployable corrections.',
                  protocol_sha256=hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cohorts={k:aggregate(v) for k,v in [('JT_TRAIN1353',source),('EAST_all2953',east),
                                                       ('EAST_current360',current),('EAST_cold20',cold)]},
                  east_by_Bt={k:aggregate([r for r in east if r['Bt_group']==k])
                              for k in ['lt0.1','0.1to1','ge1']}, rows=rows)
    with OUTPUT.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result['cohorts'], indent=2), flush=True)


if __name__=='__main__':
    main()
