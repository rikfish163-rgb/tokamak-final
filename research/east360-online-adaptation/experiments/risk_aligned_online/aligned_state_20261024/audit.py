"""Recompute source curves and audit fixed-readout identity (not efficacy proof)."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .. import io23
from ..event_adapt_20261022.train import load_source
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_train import score,sha
from ..simple_online import _durable_json
from .model import AlignedState


def independent_count(rows, threshold, required_lead):
    tp=fp=nd=ns=0; decisions=[]
    for r in rows:
        hit=next((i for i,p in enumerate(r['scores']) if float(p)>=threshold),None)
        alarm=hit is not None; flag=bool(r['is_disrupt'])
        lead=round((r['down']-float(r['times'][hit]))*1000,9) if flag and alarm else None
        timely=lead is not None and lead>required_lead
        nd+=flag;ns+=not flag;tp+=timely;fp+=alarm and not flag
        decisions.append(dict(shot=r['shot'],alarm=alarm,timely=timely,lead_ms=lead))
    return dict(tp=tp,fp=fp,n_disrupt=nd,n_safe=ns,tpr=tp/nd,fpr=fp/ns,J=tp/nd-fp/ns,decisions=decisions)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);start=time.monotonic()
    roles=io23.source_roles();shots=load_source(roles['test'],io23.advanced_times())
    base,bc=load_parent(DEFAULT_PARENT);base.cuda();original=score(base,shots,'cuda')
    report=dict(original=independent_count(original,float(bc['threshold']),10.),candidates=[])
    for root in args.runs:
        done=json.loads((root/'completed.json').read_text());ck=torch.load(root/'source.pt',map_location='cpu',weights_only=True)
        assert sha(root/'source.pt')==done['source_sha256']
        history=json.loads((root/'history.json').read_text())
        expected=min(history,key=lambda r:(r['validation']['fpr']>=.15,-r['validation']['tpr'],r['validation']['fpr'],r['loss']['risk']))
        assert expected['epoch']==ck['epoch']==done['epoch']
        parent,_=load_parent(DEFAULT_PARENT);model=AlignedState(parent,ck['model']['level_scale'])
        model.load_state_dict(ck['model']);model.cuda().eval();model.requires_grad_(False)
        protected=[k for k in model.state_dict() if k.startswith(('base.encoder.layers.1.','base.norm.','base.risk_head.')) or k=='base.fast_origin']
        for k in protected:torch.testing.assert_close(model.state_dict()[k],base.state_dict()[k.removeprefix('base.')],atol=0,rtol=0)
        rows=score(model,shots,'cuda');max_difference=0.
        with np.load(root/'test220_traces.npz') as z:
            for r in rows:
                np.testing.assert_array_equal(r['times'],z[f'{r["shot"]}_times'])
                np.testing.assert_array_equal(r['scores'],z[f'{r["shot"]}_scores'])
        counted=independent_count(rows,float(ck['threshold']),10.)
        for k in ('tp','fp','n_disrupt','n_safe','tpr','fpr','J'):assert counted[k]==done['metrics'][k],k
        report['candidates'].append(dict(run=str(root.resolve()),completed_sha256=sha(root/'completed.json'),
            checkpoint_sha256=sha(root/'source.pt'),exact_replay_shots=len(rows),fixed_source_tensors=len(protected),
            selected_by_val_only=True,metrics=counted))
    report.update(status='SOURCE_REPLAY_AND_READOUT_VERIFIED',scope='numerical replay and identity only, not independent validation',
                  source_original_unchanged=sha(DEFAULT_PARENT)==PARENT_SHA256,seconds=time.monotonic()-start)
    _durable_json(args.output,report)
    print(json.dumps(dict(status=report['status'],original={k:v for k,v in report['original'].items() if k!='decisions'},
        candidates=[{**r,'metrics':{k:v for k,v in r['metrics'].items() if k!='decisions'}} for r in report['candidates']],
        seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
