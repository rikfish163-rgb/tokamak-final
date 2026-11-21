"""Read-only signal/provenance probe; no outcome or label access."""
import argparse
from pathlib import Path

import h5py
import numpy as np

from .. import io23, simple_data
from ..fixed_test_protocol import load_test_contract
from ..simple_online import _durable_json


def describe(values):
    a = np.asarray(values, float); a = a[np.isfinite(a)]
    return dict(n=len(a), quantiles=np.quantile(a, [0, .05, .5, .95, 1]).tolist()) if len(a) else dict(n=0)


def probe(device, sid):
    raw = io23.read_signals(device, sid, rate_contract='strict' if device=='J-TEXT' else 'nominal')
    x = raw['v'].astype(float); cols = {name:x[:,i] for i,name in enumerate(io23.RAW23)}
    ip = cols['ip']*(1000 if device=='J-TEXT' else 1)
    valid = np.isfinite(x).all(1) & (cols['v_loop'] != 0) & (ip != 0)
    # A measured algebraic consistency check, NOT a fabricated extraction
    # script or unit declaration. Values near one would support V_loop/Ip.
    z_ratio = cols['Z_proxy'][valid]*ip[valid]/cols['v_loop'][valid]
    same_meaning_error = np.abs(cols['Z_proxy'][valid]-cols['v_loop'][valid]/ip[valid])
    mnm = cols['MNM']; integer_distance = np.abs(mnm-np.rint(mnm))
    masked = simple_data.read_shot(device, sid, rate_contract='strict' if device=='J-TEXT' else 'nominal')
    with h5py.File(io23.path_for(device, sid), 'r') as f:
        attr_keys = {name:list(f['data'][name].attrs) for name in io23.RAW23}
        root_attrs, data_attrs = list(f.attrs), list(f['data'].attrs)
    return dict(shot=sid, device=device, metadata_keys=dict(root=root_attrs, data=data_attrs, channels=attr_keys),
        MNM=dict(raw=describe(mnm), fractional_observations=int((integer_distance > 1e-5).sum()),
                 available_after_reader=int(masked['available'][:,io23.RAW23.index('MNM')].sum())),
        Z_proxy_times_Ip_A_over_Vloop=describe(z_ratio),
        absolute_difference_from_Vloop_over_Ip_A=describe(same_meaning_error),
        ne_nG=describe(cols['ne_nG']), qa_proxy=describe(cols['qa_proxy']))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    order = load_test_contract('EAST')['shot_ids']
    source_ids = io23.source_roles()['train'][:20]
    result = dict(scope='first20 TRAIN1353 JT and exact first20 locked EAST stream; signals/attribute names only',
                  source=[probe('J-TEXT', sid) for sid in source_ids],
                  target=[probe('EAST', sid) for sid in order[:20]],
                  inference_limit='algebraic consistency does not certify raw diagnostic extraction or units',
                  outcomes_read=False)
    _durable_json(args.output, result)
    for key in ('source', 'target'):
        medians = [r['Z_proxy_times_Ip_A_over_Vloop'].get('quantiles', [None]*5)[2] for r in result[key]]
        print(key, 'Z*Ip/V per-shot medians', medians)
        print(key, 'MNM noninteger shots', sum(r['MNM']['fractional_observations']>0 for r in result[key]))


if __name__ == '__main__': main()
