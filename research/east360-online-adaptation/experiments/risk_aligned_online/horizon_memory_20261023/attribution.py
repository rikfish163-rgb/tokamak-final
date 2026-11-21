"""Post-hoc source/map/preparation attribution on the same20 historical shots.

No training or candidate promotion. The all20 reference has seen every scored
shot and must NEVER be treated as held-out performance. Current fold mapping
uses only its15 fitting shots. All probability traces are sealed before scoring.
"""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import predict_many
from .audit import READ,independently_score,aggregate

REFERENCE=io23.ROOT/'experiments/risk_aligned_online/runs/matched_preparation20_test360_20261023'
NAMES=('source_raw','source_fold_map','fold15_static','all20_in_sample')


def main(root):
    output=root/'preparation_attribution.json'
    if output.exists():raise FileExistsError(output)
    p=READ(root/'protocol.json');verified=READ(root/'verification.json')
    assert verified['status']=='VERIFIED' and verified['completed_sha256']==sha(root/'completed.json')
    rv=READ(REFERENCE/'verification.json')
    assert rv['status']=='VERIFIED' and rv['completed_sha256']==sha(REFERENCE/'completed.json')
    rp=READ(REFERENCE/'protocol.json');checkpoint=REFERENCE/'conditional_moments.pt'
    assert sha(checkpoint)==READ(REFERENCE/'preparation_receipts.json')['conditional_moments']['checkpoint_sha256']
    assert set(p['selected_ids'])==set(rp['selected_ids'])
    torch.set_num_threads(2);torch.manual_seed(131)
    base,info=load_parent(DEFAULT_PARENT);source=PairedFeedback(base,np.ones(23)).cuda();source.set_stage('frozen')
    all20=copy.deepcopy(source);all20.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['model'])
    all20.set_stage('frozen');before_source=state_cpu(source);before_all20=state_cpu(all20)
    maps=READ(REFERENCE/'input_map.json');threshold=float(info['threshold']);assert threshold==p['threshold']
    folder=root/'attribution_traces';folder.mkdir();seals=[]
    readsignal=simple_data.read_shot
    def guarded(device,sid,**kwargs):
        assert device=='EAST' and sid in p['selected_ids']
        return readsignal(device,sid,**kwargs)
    def no_outcome(*args,**kwargs):raise AssertionError('no outcome access during attribution prediction')
    with patch.object(simple_data,'read_shot',side_effect=guarded),patch.object(io23,'read_outcome',side_effect=no_outcome):
        for plan in p['folds']:
            mapping=READ(root/f'fold{plan["fold"]}'/'input_map.json')
            for sid in plan['held']:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal={k:s[k] for k in ('values','available')}
                pred=predict_many(source,dict(source_raw=signal,source_fold_map=transform(signal,mapping)),device='cuda')
                pred['all20_in_sample']=predict_many(all20,dict(all20=transform(signal,maps)),device='cuda')['all20']
                with np.load(root/f'fold{plan["fold"]}'/'shots'/f'{sid}.npz',allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['times'],s['times']);pred['fold15_static']=saved['static'].copy()
                path=folder/f'{sid}.npz';np.savez_compressed(path,times=s['times'],**pred)
                seals.append(dict(shot=sid,fold=plan['fold'],sha256=sha(path)))
    _durable_json(folder/'seals.json',seals)
    rows=[];summaries={a:[] for a in NAMES}
    for seal in seals:
        sid=seal['shot'];s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        path=folder/f'{sid}.npz';assert sha(path)==seal['sha256']
        with np.load(path,allow_pickle=False) as trace:
            valid=s['valid'];ends=np.flatnonzero(valid)
            for a in NAMES:
                prob=trace[a][valid];times=s['times'][valid]
                r=independently_score(sid,s['is_disrupt'],s['down'],times,prob,threshold);summaries[a].append(r)
                peak=int(np.argmax(prob));eligible=np.round((s['down']-times)*1000,9)>30 if s['is_disrupt'] else np.ones(len(times),bool)
                timely_peak=float(prob[eligible].max()) if eligible.any() else None
                rows.append(dict(arm=a,fold=seal['fold'],**r,max_probability=float(prob[peak]),
                    peak_end=int(ends[peak]),peak_time=float(times[peak]),timely_max_probability=timely_peak))
    metrics={a:aggregate(rs,threshold) for a,rs in summaries.items()}
    assert metrics['fold15_static']==verified['metrics']['static']
    assert_unchanged(source,before_source);assert_unchanged(all20,before_all20);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(output,dict(status='DIAGNOSTIC',verification_sha256=sha(root/'verification.json'),
        code_sha256=sha(Path(__file__)),all20_parent_sha256=sha(checkpoint),metrics=metrics,rows=rows,
        all20_scope='IN-SAMPLE, label-exposed reference only; different calibration map; not a held-out control',
        source_unchanged=True,test_accessed=False,labels_after_seals=True))
    print(dict(status='DIAGNOSTIC',metrics=metrics),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
