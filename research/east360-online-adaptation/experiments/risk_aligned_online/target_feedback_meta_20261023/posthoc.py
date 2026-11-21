"""Descriptive failure analysis of completed traces; never changes a learner.

Full-TEST operating curves use outcomes retrospectively, not deployable
threshold selection. Calibration peak coverage uses only the historical set.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..operating_curve_audit import operating_curve,operational_maxima
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import predict_many


def main(root):
    read=lambda path:json.loads(path.read_text())
    verified=read(root/'verification.json');assert verified['status']=='VERIFIED'
    assert sha(root/'completed.json')==verified['completed_sha256']
    p=read(root/'protocol.json');result=read(root/'completed.json')
    rows={k:[] for k in result['metrics']};new_fp={k:[] for k in rows if k!='static'}
    flips=verified['paired_vs_static']
    for sid in p['contract']['shot_ids']:
        d,down=io23.read_outcome('EAST',sid)
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
            t=trace['times'];valid=np.ones(len(t),bool) if not d else (down-t)*1000>=-1e-6
            for name in rows:
                rows[name].append(dict(shot=sid,is_disrupt=d,down=down,times=t.copy(),valid=valid.copy(),scores=trace[name].copy()))
                if name!='static' and sid in flips[name]['added_fp']:
                    index=int(np.flatnonzero(trace[name]>=p['threshold'])[0])
                    new_fp[name].append(dict(shot=sid,first_crossing_index=index,
                        time_from_first_sample_ms=float((t[index]-t[0])*1000),
                        static_score_at_crossing=float(trace['static'][index]),adapted_score=float(trace[name][index]),
                        preceding_update_count=index//128-1))
    operating={}
    for name,rs in rows.items():
        c=operating_curve(rs,p['threshold']);m=operational_maxima(rs)
        for key in ('tp','fp'):assert c['fixed_source_threshold'][key]==result['metrics'][name][key]
        d,s=m['positive'][:,None],m['negative'][None,:]
        operating[name]=dict(c,timely_peak_ranking_auc=float(((d>s)+.5*(d==s)).mean()))
    history=read(root/'training.json');summary={}
    for arm in ('forecast','point_meta','maximum_meta'):
        summary[arm]={}
        for label,part in (('first8',history[:8]),('last8',history[-8:])):
            summary[arm][label]={}
            for disrupt in (False,True):
                records=[r for h in part for r in h[arm]['records'] if r['is_disrupt']==disrupt]
                summary[arm][label]['disrupt' if disrupt else 'safe']={
                    k:float(np.mean([r[k] for r in records])) for k in ('risk_before','risk_after','forecast')}
        summary[arm]['head_gradient_clipped_steps']=sum(h[arm]['head_gradient_norm']>1 for h in history)
    # Inspect whether negative examples ever include the frozen model's own
    # high-risk regions; compare chronological block sampling to entire shots.
    safe_records=[r for h in history for r in h['maximum_meta']['records'] if not r['is_disrupt']]
    coverage=dict(sampled_safe_blocks=len(safe_records),sampled_safe_blocks_with_static_alarm=
                  sum(r['risk_before']>math.log(2) for r in safe_records),shots=[])
    torch.set_num_threads(2);torch.manual_seed(131)
    base,_=load_parent(DEFAULT_PARENT);model=PairedFeedback(base,np.ones(23)).cuda()
    model.load_state_dict(torch.load(root/'warm.pt',map_location='cpu',weights_only=True)['model']);model.set_stage('frozen')
    mapping=read(root/'input_map.json')
    for sid in p['fit_ids']:
        d,_=io23.read_outcome('EAST',sid)
        if d:continue
        s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        signal=transform({k:s[k] for k in ('values','available')},mapping)
        risk=predict_many(model,dict(static=signal),device='cuda')['static']
        peak=int(np.argmax(risk));used=[r for r in safe_records if r['shot']==sid]
        eligible=risk[256:];top_block=int(256+np.argmax(eligible))//128*128 if len(eligible) else None
        coverage['shots'].append(dict(shot=sid,static_peak=float(risk[peak]),peak_index=peak,
            static_alarm=bool((risk>=p['threshold']).any()),selected_block_count=len(used),
            eligible_peak_block=top_block,selected_eligible_peak_block=any(r['count']==top_block for r in used)))
    output=dict(scope='POSTHOC DEVELOPMENT DIAGNOSTIC, NOT VALIDATION OR DEPLOYABLE THRESHOLD SELECTION',
        source_experiment=sha(root/'completed.json'),verification_sha256=sha(root/'verification.json'),
        code_sha256=sha(Path(__file__)),operating=operating,training=summary,
        training_safe_peak_coverage=coverage,new_false_alarms=new_fp,
        learning_or_prediction_modified=False,all_test_labels_used_for_operating_curve=True)
    _durable_json(root/'posthoc_diagnostic.json',output)
    print(json.dumps(dict(operating={k:dict(auc=v['timely_peak_ranking_auc'],caps=v['posthoc_best_at_cap'])
                                     for k,v in operating.items()},safe_coverage=coverage)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
