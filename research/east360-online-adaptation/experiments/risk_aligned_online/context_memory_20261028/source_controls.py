"""Read-only source risk attribution: true, no, reversed, and constant feedback.

These are sampled source validation endpoints, NOT full-shot EAST results.
No model or acceptance criterion is changed by this post-hoc diagnosis.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..delayed_recurrent_memory_20261025.model import endpoint_corrections
from ..direction_teaching_20261026.learning import objective
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT
from ..simple_online import _durable_json
from ..simple_train import sha
from ..source_task_memory_20261026.learning import to_device
from .checkpoint import read, load_context_memory
from .model import ContextAdapter


@torch.no_grad()
def evaluate_modes(model,e,threshold):
    main=model.memory.trace(e['means'],e['complete'],mode='memory',return_before=True)
    result={}
    h,v=e['h'],e['valid']
    for mode in ('memory','off','reverse','hold'):
        t=main if mode=='memory' else model.memory.trace(e['means'],e['complete'],mode=mode,
            reference_norm=main['norm'],return_before=True)
        logits=[]
        for key in ('delta','before_delta'):
            d=endpoint_corrections(t[key],e['ends'])
            logits.append(torch.cat([model.risk(h[i:i+32],v[i:i+32],d[i:i+32]) for i in range(0,len(h),32)]))
        _,info=objective(*logits,e,threshold,'detached')
        result[mode]={k:float(v) if torch.is_tensor(v) else v for k,v in info.items()}
        result[mode]['mean_logit']=float(logits[0].mean())
        result[mode]['max_logit']=float(logits[0].max())
        result[mode]['zero_direction_points']=int(t['zero_direction'].sum())
    return result


def run(root):
    output=root/'source_risk_controls.json'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.mha.set_fastpath_enabled(False)
    p=read(root,'protocol.json')
    base,_=load_parent(DEFAULT_PARENT)
    model=ContextAdapter(base.cuda()).cuda().eval()
    receipt=load_context_memory(model,root/'context8_epoch2.pt')
    model.requires_grad_(False)
    cache=Path(p['cache_root']); reference=Path(p['reference_root'])
    if (sha(cache/'inventory.json')!=p['cache_inventory_sha256']
            or sha(reference/'local_targets.pt')!=p['reference_hashes']['local_targets.pt']):
        raise ValueError('source targets or cache changed')
    inv={r['shot']:r for r in read(cache,'inventory.json')}
    targets=torch.load(reference/'local_targets.pt',map_location='cpu',weights_only=True)
    recorded=read(root,'val_epoch2.json')['per_shot']
    if [r['shot'] for r in recorded]!=p['val_ids']:
        raise ValueError('complete source VAL160 required')
    rows=[]
    for saved in recorded:
        sid=saved['shot']; path=cache/'cache'/f'{sid}.pt'
        if inv[sid]['role']!='val' or sha(path)!=inv[sid]['cache_sha256']:
            raise ValueError('source validation cache changed')
        e=torch.load(path,map_location='cpu',weights_only=True);e.update(targets[sid])
        result=evaluate_modes(model,to_device(e,'cuda'),p['threshold'])
        for key in ('bag','local','direction'):
            if not np.isclose(result['memory'][key],saved[key],atol=1e-7,rtol=1e-6):
                raise ValueError('recomputed source risk differs from sealed epoch2')
        rows.append(dict(shot=sid,disruptive=e['flag'],modes=result))
    summary={}
    for name,sub in (('all',rows),('D',[r for r in rows if r['disruptive']]),('S',[r for r in rows if not r['disruptive']])):
        summary[name]={mode:{k:float(np.mean([r['modes'][mode][k] for r in sub]))
            for k in ('bag','local','direction','mean_logit','max_logit','zero_direction_points')}
            for mode in ('memory','off','reverse','hold')}
    before=read(root,'val_epoch0.json')['summary']
    for key in ('bag','local'):
        if not np.isclose(summary['all']['off'][key],before[key],atol=1e-7,rtol=1e-6):
            raise ValueError('no-feedback risk is not the unchanged source reference')
    _durable_json(output,dict(scope='POSTHOC_SAMPLED_SOURCE_VAL160_RISK_CONTROLS_NOT_EAST',
        model_updates=0,east_records_read=0,test220_records_read=0,source_memory=receipt,
        controls='reverse permutes horizon targets with matched context norm; hold uses zero increments; zero directions reported',
        code_sha256=sha(Path(__file__)),summary=summary,rows=rows))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    run(parser.parse_args().root)
