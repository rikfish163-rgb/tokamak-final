"""Exact raw-point census for TRAIN1353 and the current complete EAST360.

No outcomes or model predictions. Quantiles are time-point weighted, not
independent samples, optimizer sampling frequencies, or physical calibration.
"""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.causal_input import apply_input_contract
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from experiments.risk_aligned_online.warmstart_alignment_20261022.alignment import transform


Q = [0, .01, .05, .5, .95, .99, 1]


def describe(a):
    a = np.asarray(a)
    good = a[np.isfinite(a)]
    return dict(n=int(a.size), nonfinite=int(a.size-good.size),
                zero=int(np.count_nonzero(good==0)), negative=int(np.count_nonzero(good<0)),
                quantiles=np.quantile(good, Q).tolist() if good.size else None)


def collect(device, ids, mapping=None):
    raw, model, masks, receipts = [], [], [], []
    for j, sid in enumerate(ids):
        with h5py.File(io23.path_for(device, sid), 'r') as h:
            assert set(h['data']) == set(io23.RAW23)
            channels = [np.asarray(h['data'][n][:], np.float64) for n in io23.RAW23]
            lengths = {len(a) for a in channels}
            assert len(lengths)==1
            x64 = np.stack(channels, 1)
            has_units = {n:[k for k in h['data'][n].attrs if k.lower() in ('unit','units')]
                         for n in io23.RAW23}
        x = x64.astype(np.float32)
        assert np.array_equal(np.isfinite(x64), np.isfinite(x)), 'float32 overflow'
        available = np.isfinite(x)
        values = np.where(available, x, 0.)
        if device == 'J-TEXT':
            values[:,9] *= 1000.
        else:
            available[:,2] = False
            values[~available] = 0.
        signal = dict(values=values, available=available)
        if device == 'EAST':
            signal = apply_input_contract(signal)
            if mapping is not None:
                signal = transform(signal, mapping)
        raw.append(x)
        model.append(signal['values'])
        masks.append(signal['available'])
        receipts.append(dict(shot=sid, points=len(x),
            raw_nonfinite=int((~np.isfinite(x)).sum()),
            raw_constant_columns=[n for c,n in enumerate(io23.RAW23) if np.all(x[:,c]==x[0,c])],
            unit_attributes=has_units))
        if (j+1)%500==0:
            print(dict(device=device, read=j+1, total=len(ids)), flush=True)
    return np.concatenate(raw), np.concatenate(model), np.concatenate(masks), receipts


def main():
    output = Path(__file__).with_suffix('.json')
    if output.exists():
        raise FileExistsError(output)
    base, _ = load_parent(DEFAULT_PARENT)
    center, scale = base.center.numpy(), base.scale.numpy()
    run = Path(__file__).parent/'probe_outputs/anchored_full360_diagnostic_v1'
    protocol = json.loads((run/'protocol.json').read_text())
    mapping = json.loads((io23.ROOT/protocol['reference']/'input_map.json').read_text())
    groups = [('JT_TRAIN1353','J-TEXT',io23.source_roles()['train'],None),
              ('EAST_cold20','EAST',protocol['order'][:20],None),
              ('EAST_suffix340','EAST',protocol['order'][20:],mapping)]
    result = dict(scope='READ_ONLY_FULL23_NUMERIC_CENSUS', source_sha256=hashlib.sha256(DEFAULT_PARENT.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  quantile_probabilities=Q, units_certified=False, outcomes_read=False,
                  trained=False, quantile_weighting='all raw time points, no outcome/time-valid crop',
                  feature_order=io23.RAW23, groups={})
    source_bounds = {}
    for name, device, ids, affine in groups:
        raw, values, mask, receipts = collect(device, ids, affine)
        channels = {}
        for c, channel in enumerate(io23.RAW23):
            observed = mask[:,c]
            v = values[observed,c]
            z = np.arcsinh((v-center[c])/scale[c])
            assert np.isfinite(z).all()
            ds = describe(v)
            if name=='JT_TRAIN1353':
                source_bounds[channel] = ds['quantiles'][1], ds['quantiles'][-2]
            low, high = source_bounds[channel]
            channels[channel] = dict(raw_float32=describe(raw[:,c]),
                actual_model_input_available=ds, checkpoint_asinh=describe(z),
                unavailable_points=int((~observed).sum()),
                constant_raw_shots=sum(channel in r['raw_constant_columns'] for r in receipts),
                model_points_outside_source_raw_q01_q99=int(((v<low)|(v>high)).sum()),
                source_bounds=[low,high], checkpoint_center=float(center[c]),
                checkpoint_scale=float(scale[c]))
        result['groups'][name] = dict(shots=len(ids), points=len(raw), channels=channels, receipts=receipts)
        print(dict(group=name, shots=len(ids), points=len(raw), all23_checked=True), flush=True)
    with output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print('Written',str(output),flush=True)


if __name__=='__main__':
    main()
