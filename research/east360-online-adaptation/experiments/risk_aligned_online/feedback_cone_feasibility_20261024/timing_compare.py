"""Post-hoc matched-endpoint comparison; not proof that clock alone caused harm."""
import argparse
from pathlib import Path

import numpy as np

from ..simple_online import _durable_json
from ..simple_train import sha
from .run import READ


def main(policy,stream):
    rows=[]
    for r in READ(policy/'probes.json'):
        if r['group']!='calibration' or not r['is_disrupt']:continue
        sid,q=r['shot'],r['query']
        with np.load(stream/'shots'/f'{sid}.npz') as p:
            baseline=-np.log(float(p['off'][q]));after=-np.log(float(p['mean010'][q]))
        probe=next(v for v in r['finite']['mean'] if v['step_norm']==.01)
        rows.append(dict(shot=sid,query=q,probe_feedback_age=25,stream_feedback_age=q-(q//128*128-1),
                         probe_delta=probe['risk_after']-r['risk_before'],stream_delta=after-baseline))
    report=dict(scope='POSTHOC_MATCHED_POINTS; different clocks and math/fused inference; no causal-clock attribution',
                policy_completed_sha256=sha(policy/'completed.json'),stream_completed_sha256=sha(stream/'completed.json'),
                rows=rows,probe_mean=float(np.mean([r['probe_delta'] for r in rows])),
                stream_mean=float(np.mean([r['stream_delta'] for r in rows])))
    _durable_json(stream/'timing_comparison.json',report)
    print({k:v for k,v in report.items() if k!='rows'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--stream',type=Path,required=True)
    a=p.parse_args();main(a.policy,a.stream)
