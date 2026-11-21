"""Post-hoc verification only: fixed decisions, prefix cost, full-shot CUDA replay."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import simple_data
from ..causal_input import apply_input_contract
from ..fixed_test_protocol import load_test_contract
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha,metrics,validation_rank,load_shots,score,select_threshold
from ..warmstart_alignment_20261022.run import public
from ..deepcoral_prefix_20261022.audit import paired_flips
from .run import ARMS,REFERENCE,predict_target,phase_metrics
from .model import CoordinateClassifier


def independent_counts(records,thresholds,lead=30.):
    """Separate direct array arithmetic, not the runner's metrics helper."""
    tp=fp=nd=ns=0;very_early=0
    for r,th in zip(records,thresholds):
        crossed=np.where(np.asarray(r['scores'])>=th)[0]
        if r['is_disrupt']:
            nd+=1
            if len(crossed):
                delta=round(float(r['down']-r['times'][crossed[0]])*1000,9)
                tp+=int(delta>lead);very_early+=int(delta>3000)
        else:ns+=1;fp+=int(bool(len(crossed)))
    return dict(tp=tp,fp=fp,n_disrupt=nd,n_safe=ns,very_early_over3000ms=very_early)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    root=p.parse_args().directory
    done=json.loads((root/'completed.json').read_text());assert done['status']=='COMPLETED'
    protocol=json.loads((root/'protocol.json').read_text());selection=json.loads((root/'selection.json').read_text())
    fit=json.loads((root/'alignment.json').read_text());full=json.loads((root/'per_shot.json').read_text())
    order=load_test_contract('EAST')['shot_ids'];assert protocol['contract']['shot_ids']==order
    for path,h in protocol['code_sha256'].items():assert sha(Path(protocol['plan']).parents[3]/path)==h
    assert sha(protocol['plan'])==protocol['plan_sha256']
    assert sha(REFERENCE/'protocol.json')==protocol['reference_protocol_sha256']
    assert sha(root/'training_manifest.jsonl')==selection['training_manifest_sha256']
    history=json.loads((root/'history.json').read_text())
    assert len(history)==10 and fit['target_ids']==order[:20]
    ids=set(simple_data.source_roles()['train']);assert set(fit['source_ids'])==ids
    manifest=[json.loads(line) for line in (root/'training_manifest.jsonl').read_text().splitlines()]
    assert len(manifest)==1024
    for i,r in enumerate(manifest):
        assert (r['epoch'],r['step'])==(1+i//256,i%256) and len(r['examples'])==64
        assert all(e['shot'] in ids and e['label'] in (0,1) and e['end']>=0 for e in r['examples'])
    cs=np.array(fit['source_cov']);ct=np.array(fit['target_cov']);a=np.array(fit['matrix'])
    np.testing.assert_allclose(a.T@(cs+np.eye(23))@a,ct+np.eye(23),rtol=0,atol=1e-9)
    seals=json.loads((root/'prediction_receipts.json').read_text());assert [r['shot'] for r in seals]==order
    activation=json.loads((root/'activation_receipt.json').read_text());assert activation['after_seals']==seals[:20]
    assert activation['target_outcomes_accessed'] is False
    rows={a:[] for a in ARMS};raw_rows=[]
    refs={r['shot']:r for r in json.loads((REFERENCE/'prediction_receipts.json').read_text())}
    for i,(sid,seal) in enumerate(zip(order,seals)):
        path=root/'shots'/f'{sid}.npz';assert sha(path)==seal['prediction_sha256']
        ref=REFERENCE/'shots'/f'{sid}.npz';assert sha(ref)==refs[sid]['prediction_sha256']
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(path,allow_pickle=False) as trace,np.load(ref,allow_pickle=False) as old:
            np.testing.assert_array_equal(s['times'],trace['times']);v=s['valid']
            for arm in ARMS:
                assert trace[arm].shape==s['times'].shape and np.isfinite(trace[arm]).all()
                assert np.all((trace[arm]>=0)&(trace[arm]<=1))
                if i<20:np.testing.assert_array_equal(trace[arm]>=protocol['parent_threshold'],old['static']>=protocol['parent_threshold'])
                rows[arm].append(dict(shot=sid,times=s['times'][v],scores=trace[arm][v].copy(),is_disrupt=s['is_disrupt'],down=s['down']))
            raw_rows.append(dict(shot=sid,times=s['times'][v],scores=old['static'][v].copy(),is_disrupt=s['is_disrupt'],down=s['down']))
    report={};threshold=protocol['parent_threshold']
    for arm,rs in rows.items():
        th=selection['models'][arm]['threshold']
        assert phase_metrics(rs,threshold,th)==full[arm]
        counts=independent_counts(rs,[threshold]*20+[th]*340)
        for k,v in counts.items():assert full[arm][k]==v
        report[arm]=dict(full360=public(full[arm]),warmup20=public(metrics(rs[:20],threshold,lead=30.)),
                        suffix340=public(metrics(rs[20:],th,lead=30.)),
                        common_parent_threshold=public(metrics(rs,threshold,lead=30.)))
        picked=min((r for r in history if r['arm']==arm),key=lambda r:validation_rank(r['validation']))
        assert picked['epoch']==selection['models'][arm]['epoch']
    raw=metrics(raw_rows,threshold,lead=30.)
    torch.set_num_threads(2);torch.manual_seed(131);torch.backends.mha.set_fastpath_enabled(False)
    replayed=[];source_checks={}
    for arm in ARMS:
        selected=selection['models'][arm];assert sha(selected['path'])==selected['sha256']
        payload=torch.load(selected['path'],map_location='cpu',weights_only=True)
        base,_=load_parent(DEFAULT_PARENT);original={k:v.clone() for k,v in base.state_dict().items()}
        model=CoordinateClassifier(base,fit,coral=arm=='ucoral').cuda()
        model.load_state_dict(payload['state']);model.eval();model.requires_grad_(False)
        for key in original:
            if key in ('center','scale','fast_origin') or key.startswith('forecast_head.'):
                torch.testing.assert_close(model.base.state_dict()[key].cpu(),original[key],rtol=0,atol=0)
        val=load_shots(simple_data.source_roles()['val'],simple_data.advanced_times())
        measured=public(select_threshold(score(model,val,'cuda'),.1));assert measured==payload['validation']
        source_checks[arm]=measured
        for index in (20,179,359):
            sid=order[index];s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            fresh=predict_target(model,s)
            with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
                np.testing.assert_array_equal(fresh,trace[arm])
            replayed.append(dict(shot=sid,arm=arm,max_abs_error=0.))
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    receipt=dict(status='VERIFIED',rescored_traces=720,replayed=replayed,selected_source_validation_replayed=source_checks,
        source_archive_unchanged=True,code_hashes_match=True,prefix_original_threshold_preserved=True,
        metrics=report,raw_reference=public(raw),source_to_target_regularized_identity_error=fit['regularized_identity_max_error'],
        empirical_covariance_distance_before=float(np.linalg.norm(cs-ct)),
        empirical_covariance_distance_after=float(np.linalg.norm(a.T@cs@a-ct)),
        paired_comparisons=dict(zscore_to_ucoral=paired_flips(full['zscore']['predictions'],full['ucoral']['predictions'])),
        audit_code_sha256=sha(Path(__file__)))
    _durable_json(root/'verification.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
