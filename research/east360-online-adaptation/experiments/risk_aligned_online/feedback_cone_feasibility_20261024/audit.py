"""Independent two-vertex oracle enumeration and fixed real-probe replay."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .. import io23
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import collect_geometry, solve_weights
from .run import (READ, make_parent, models_for_calibration, state_cpu, assert_unchanged,
                  REFERENCE, signal, transform, DEFAULT_PARENT, PARENT_SHA256, evaluate_directions, summarize)


def enumerated_optimum(a,b):
    # A simplex intersected with one halfspace has either original vertices
    # or intersections of two-vertex edges with the constraint plane.
    values = [float(a[c]) for c in range(len(a)) if b[c]>=0]
    for i in range(len(a)):
        for j in range(i):
            if b[i]*b[j]<0:
                wi = b[j]/(b[j]-b[i])
                values.append(float(wi*a[i]+(1-wi)*a[j]))
    if not values:
        raise ValueError('no feasible enumerated point')
    return max(values)


def main(root):
    done = READ(root/'completed.json')
    assert done['status']=='completed'
    for name,digest in done['artifact_sha256'].items():
        assert sha(root/name)==digest,name
    rows = [r for f in sorted((root/'probes').glob('*.json')) for r in READ(f)]
    for r in rows:
        a,b = np.array(r['risk_alignment']),np.array(r['signal_alignment'])
        np.testing.assert_allclose(enumerated_optimum(a,b),r['oracle_lp']['margin'],atol=1e-9,rtol=1e-9)
        w = np.array(r['oracle_weights'])
        assert abs(w.sum()-1)<1e-8 and w.min()>-1e-8 and b@w>=-1e-8
    shared = READ(root/'shared_weights.json'); history=[r for r in rows if r['group']=='preparation']
    w, lp = solve_weights(np.array([r['risk_alignment'] for r in history]),np.array([r['signal_alignment'] for r in history]))
    np.testing.assert_array_equal(w,shared['weights']); assert lp==shared['lp']
    fresh_summary=summarize(rows)
    # Sorting files changes summation order only at sub-ulp levels. Compare
    # all leaves numerically rather than serialised dictionary string order.
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a: compare(a[k],b[k])
        else: np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
    compare(fresh_summary,READ(root/'summary.json'))
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    parent,_=make_parent(); model=models_for_calibration(parent)['mapped_fixed']; original=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json'); replay=[]
    for sid in (63226,89272,89284,89584):
        r=max((r for r in rows if r['shot']==sid),key=lambda v:v['query'])
        raw=signal(sid); mapped=transform({k:raw[k] for k in ('values','available')},mapping)
        fresh,state=collect_geometry(model,mapped,r['query'],r['label'],device='cuda')
        for key,value in fresh.items():
            assert value==r[key],(sid,key)
        decision=evaluate_directions(model,state,fresh,np.array(shared['weights']))
        assert decision['finite']==r['finite']
        np.testing.assert_array_equal(decision['oracle_weights'],r['oracle_weights'])
        replay.append(dict(shot=sid,query=r['query'],finite_arms=4,step_norms=[.001,.01]))
    assert_unchanged(model,original); assert sha(DEFAULT_PARENT)==PARENT_SHA256
    report=dict(status='passed',completed_sha256=sha(root/'completed.json'),enumerated_lp_count=len(rows),
        summary_recomputed=True,shared_lp_recomputed=True,replayed=replay,
        original_source_unchanged=True,code_sha256=sha(Path(__file__)),
        current_tests_sha256=sha(io23.ROOT/'experiments/risk_aligned_online/tests/test_feedback_cone_feasibility.py'))
    _durable_json(root/'audit.json',report);print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
