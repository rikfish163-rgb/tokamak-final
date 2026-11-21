"""Post-hoc source peak audit; does not fit/select/promote another candidate."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .. import simple_data
from ..simple_online import _hash, _durable_json
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT
from ..ffn_intra_20261005.model import FfnModel
from ..ffn_intra_20261005.stream import predict_shot
from .whole_meta import select_peak


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matched',type=Path,required=True)
    p.add_argument('--whole',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args();output=args.whole/'peak_audit.json'
    if output.exists():raise FileExistsError(output)
    summary=json.loads((args.whole/'summary.json').read_text())
    protocol=summary['protocol'];episodes=protocol['validation_episodes']
    if protocol['parent_sha256']!=_hash(DEFAULT_PARENT):
        raise ValueError('qualified source changed')
    train=set(simple_data.source_roles()['train']);advanced=simple_data.advanced_times()
    if not {e['shot'] for e in episodes}<=train:
        raise ValueError('source TRAIN only')
    shots=[simple_data.attach_outcome(simple_data.read_shot('J-TEXT',e['shot']),'J-TEXT',advanced)
           for e in episodes]
    torch.set_num_threads(4)
    base,state=load_parent(DEFAULT_PARENT);base=base.to(args.device);model=FfnModel(base)
    original={k:v.detach().clone() for k,v in base.forecast_head.state_dict().items()}
    threshold=float(state['threshold']);boundary=math.log(threshold/(1-threshold))
    paths=dict(static=None,original=None,
        signal60=args.matched/'signal_only/candidate_060.pt',
        later60=args.matched/'risk_aligned/candidate_060.pt',
        whole60=args.whole/'candidate_060.pt')
    results={};hashes={}
    for name,path in paths.items():
        if path is None:
            base.forecast_head.load_state_dict(original)
        else:
            # These explicit local paths are our just-created source-fit artifacts,
            # not arbitrary uploads. Whole protocol contains TorchVersion metadata.
            hashes[name]=_hash(path)
            artifact=torch.load(path,map_location='cpu',weights_only=False)
            pp=artifact['protocol']
            if (pp['parent_sha256']!=protocol['parent_sha256']
                    or pp['validation_episodes']!=episodes or not pp['source_only']
                    or artifact['iteration']!=60):
                raise ValueError('source candidate identity mismatch')
            base.forecast_head.load_state_dict(artifact['forecast_head'],strict=True)
        rows=[]
        for shot in shots:
            scores,_=predict_shot(model,shot,intra=name!='static',device=args.device,
                                 step=.1,radius=1.,offline_source=True)
            q=select_peak(shot,scores);prob=float(np.clip(scores[q],1e-7,1-1e-7))
            margin=math.log(prob/(1-prob))-boundary
            hits=np.flatnonzero(shot['valid'] & (scores>=threshold))
            first=int(hits[0]) if len(hits) else None
            rows.append(dict(shot=shot['shot'],label=int(shot['is_disrupt']),
                loss=float(np.logaddexp(0.,-margin if shot['is_disrupt'] else margin)),
                peak_index=q,peak_probability=float(scores[q]),before_first_update=q<26,
                first_alarm_index=first,first_alarm_lead_ms=(float(round((shot['down']-shot['times'][first])*1000.,9))
                    if first is not None and shot['is_disrupt'] else None)))
        classes={str(y):dict(count=sum(r['label']==y for r in rows),
            mean_loss=float(np.mean([r['loss'] for r in rows if r['label']==y])),
            preupdate_peaks=sum(r['before_first_update'] for r in rows if r['label']==y)) for y in (0,1)}
        results[name]=dict(classes=classes,records=rows)
        print(json.dumps(dict(stage='source_peak_audit',head=name,classes=classes)),flush=True)
    original_rows={r['shot']:r for r in results['original']['records']}
    differences=[]
    for row in results['whole60']['records']:
        old=original_rows[row['shot']]
        differences.append(dict(shot=row['shot'],label=row['label'],loss_change=row['loss']-old['loss'],
            old_peak=old['peak_index'],new_peak=row['peak_index'],
            first_alarm_changed=old['first_alarm_index']!=row['first_alarm_index']))
    if _hash(DEFAULT_PARENT)!=protocol['parent_sha256']:
        raise RuntimeError('source checkpoint changed')
    _durable_json(output,dict(source_only=True,posthoc_no_candidate_selection=True,
        parent_sha256=protocol['parent_sha256'],head_sha256=hashes,threshold=threshold,
        results=results,whole60_vs_original=sorted(differences,key=lambda r:r['loss_change']),
        interpretation='same32 source development shots; diagnostic attribution, not independent evidence'))


if __name__=='__main__':main()
