"""Independent scalar recount and full replay of the small TRAIN screen."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from ..task_geometry_20261023.run import load_model
from ..warmstart_alignment_20261022.alignment import transform
from .learning import ARMS,encode,step
from .run import probe


def bce(z,y):
    return np.logaddexp(0.,np.float64(z))-y*np.float64(z)


def main(root):
    read=lambda path:json.loads(path.read_text())
    p,done=read(root/'protocol.json'),read(root/'completed.json')
    assert done['protocol_sha256']==sha(root/'protocol.json')
    assert done['probes_sha256']==sha(root/'probes.json')
    for file,digest in p['code_sha256'].items():
        # Test-only additions after execution are declared below; executable
        # model/run sources and all dependencies must still be identical.
        if file.endswith('tests/test_arrived_teacher.py'):continue
        assert sha(io23.ROOT/file)==digest,file
    assert sha(Path(p['plan_path']))==p['plan_sha256']
    ref=Path(p['reference'])
    assert sha(ref/'conditional_moments.pt')==p['parent_sha256']
    for file,digest in p['reference_hashes'].items():assert sha(ref/file)==digest
    rows=read(root/'probes.json');samples=read(root/'sampling.json')
    assert len(rows)==1920 and len(samples)==640
    counts={}
    for r in rows:
        assert r['count']+24==r['query'] and r['update']['last_observation']==r['count']-1
        assert all(e<r['count'] for e in r['update']['target_ends'])
        for suffix in ('before','after'):
            np.testing.assert_allclose(r['bce_'+suffix],bce(r['logit_'+suffix],r['label']),rtol=1e-4,atol=7e-7)
        assert r['update']['parameter_delta_norm']<=.00101
        counts.setdefault((r['shot'],r['index']),set()).add(r['arm'])
    assert len(counts)==640 and all(a==set(ARMS) for a in counts.values())
    independently={}
    for arm in ARMS:
        groups={}
        for group in ('safe','disruptive_timely','disruptive_early'):
            selected=[r for r in rows if r['arm']==arm and r['group']==group]
            shots=sorted({r['shot'] for r in selected});part=[]
            for sid in shots:
                ss=[r for r in selected if r['shot']==sid]
                part.append(dict(shot=sid,before=float(np.mean([bce(r['logit_before'],r['label']) for r in ss])),
                    after=float(np.mean([bce(r['logit_after'],r['label']) for r in ss]))))
            groups[group]=dict(shots=part,before=float(np.mean([r['before'] for r in part])),
                after=float(np.mean([r['after'] for r in part])))
            for key in ('before','after'):
                np.testing.assert_allclose(groups[group][key],done['summary'][arm]['groups'][group]['bce_'+key],rtol=1e-4,atol=7e-7)
        eligible=[groups[g] for g in ('safe','disruptive_timely')]
        improved=sum(r['after']<r['before'] for r in groups['disruptive_timely']['shots'])
        passed=bool(improved>=3 and all(r['after']<=r['before'] for r in eligible)
                    and sum(r['after'] for r in eligible)<sum(r['before'] for r in eligible))
        assert passed==done['summary'][arm]['passes_screen']
        independently[arm]=dict(groups=groups,passes=passed,improved_disruptive_shots=improved)
    torch.set_num_threads(2);model,_=load_model();frozen=state_cpu(model)
    mapping=read(ref/'input_map.json');replayed=0;signals={};durations=[]
    saved={(r['shot'],r['index'],r['arm']):r for r in rows}
    with patch.object(io23,'read_outcome',side_effect=AssertionError('replay uses sealed historical labels only')):
        for sid in p['selected_ids']:
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=transform({k:s[k] for k in ('values','available')},mapping);signals[sid]=signal
            for sample in [r for r in samples if r['shot']==sid]:
                row={k:sample[k] for k in ('shot','index','query','count','group')}
                for r in probe(model,signal,row,sample['label']):
                    assert r==saved[(sid,sample['index'],r['arm'])]
                    replayed+=1
                durations.append(float((s['times'][sample['query']]-s['times'][sample['count']-1])*1000))
            print(dict(replayed=replayed,shot=sid),flush=True)
    # Deterministic extremal diagnostic cases, not new policy selection:
    # largest harm in each of safe / positive groups for advance.
    derivative=[]
    for group in ('safe','disruptive_timely'):
        r=max((v for v in rows if v['arm']=='advance' and v['group']==group),key=lambda v:v['actual_delta'])
        signal=signals[r['shot']];_,receipt,g=step(model,signal,r['count'],'advance')
        origin=initial_encoder(model);direction={k:-.001*receipt['clip_factor']*v for k,v in g.items()}
        epsilon=.01;loss=[]
        for sign in (-1,1):
            params={k:(v+sign*epsilon*direction[k]).detach().requires_grad_(True) for k,v in origin.items()}
            h=encode(model,signal,[r['query']],params,'cuda')
            logit=float(model.base.risk_head(h)[0,0].detach());loss.append(float(bce(logit,r['label'])))
        numerical=(loss[1]-loss[0])/(2*epsilon)
        np.testing.assert_allclose(numerical,r['first_order_delta'],rtol=.03,atol=2e-5)
        derivative.append(dict(shot=r['shot'],query=r['query'],group=group,
            analytic=r['first_order_delta'],central_difference=numerical,
            actual_delta=r['actual_delta'],fraction=.01))
    assert_unchanged(model,frozen);assert sha(Path(p['source_checkpoint']))==p['source_sha256']
    report=dict(status='VERIFIED',audit_sha256=sha(Path(__file__)),completed_sha256=sha(root/'completed.json'),
        protocol_sha256=sha(root/'protocol.json'),recomputed_rows=len(rows),cuda_replayed_rows=replayed,
        all_screen_probes_replayed=True,independent_numpy_groups=independently,
        extrema_derivative_checks=derivative,parent_unchanged=True,no_test_access=True,
        extra_test_sha256=sha(io23.ROOT/'experiments/risk_aligned_online/tests/test_arrived_teacher.py'),
        source_snapshot_tests='initial5 tests archived;2 additional regressions added after run, executable sources unchanged',
        actual_query_delay_ms=dict(minimum=min(durations),median=float(np.median(durations)),maximum=max(durations)))
    _durable_json(root/'verification.json',report)
    print(dict(status='VERIFIED',replayed=replayed,derivatives=derivative),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
