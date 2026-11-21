"""Independent re-forward of every later-site diagnostic witness."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..delayed_recurrent_memory_20261025.learning import batch
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .site_probe import late_logits


def run(root):
    p=read(root,'protocol.json');d=read(root,'late_site.json');original=read(root,'records.json')['records']
    if (not d['complete'] or d['reference_complete_sha256']!=sha(root/'complete.json')
            or d['reference_audit_sha256']!=sha(root/'audit.json')
            or d['witnesses_sha256']!=sha(root/'late_witnesses.npz')
            or d['code_sha256']!=sha(Path(__file__).with_name('site_probe.py'))):
        raise ValueError('diagnostic provenance changed')
    rows=d['records']
    for old,new in zip(original,rows):
        if {k:v for k,v in old.items() if not k.startswith('radius')}!={k:v for k,v in new.items() if not k.startswith('radius')}:
            raise ValueError('points or baseline changed')
    if len(rows)!=len(original):raise ValueError('point coverage changed')
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,parent=load_parent(DEFAULT_PARENT);base.cuda();reference=Path(p['reference'])
    for path,h in p['reference_hashes'].items():
        if sha(reference/path)!=h:raise ValueError('reference changed')
    mapping=read(reference,'input_map.json');hashes=read(reference,'input_hashes.json')
    slow=torch.load(reference/'models/cold.pt',map_location='cpu',weights_only=True)['slow'].cuda()
    xs=[];masks=[]
    for sid in p['order'][20:]:
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('raw changed')
        raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        ends=[r['end'] for r in rows if r['shot']==sid]
        if ends:
            sig=transform({k:raw[k] for k in ('values','available')},mapping)
            x,m=batch(sig,ends,'cpu');xs.append(x);masks.append(m)
    x=torch.cat(xs);m=torch.cat(masks);sign=np.array([1 if r['flag'] else -1 for r in rows]);groups={}
    with np.load(root/'late_witnesses.npz') as w:
        for radius in p['radii']:
            key=f'radius{radius}';s=w[key+'_state'];z=w[key+'_logits'];n=w[key+'_norms'];steps=w[key+'_best_step']
            if (s.shape!=(len(rows),128) or z.shape!=(65,len(rows)) or not np.isfinite(s).all()
                    or not np.isfinite(z).all() or not np.isfinite(n).all()
                    or (np.linalg.norm(s,axis=1)>radius+2e-6).any() or (n>radius+2e-6).any()
                    or not np.array_equal(steps,(z*sign[None]).argmax(0))):raise ValueError('invalid witness')
            probabilities=[];error=0.
            for lo in range(0,len(rows),32):
                with torch.no_grad():
                    best=late_logits(base,x[lo:lo+32].cuda(),m[lo:lo+32].cuda(),torch.from_numpy(s[lo:lo+32]).cuda(),slow)
                b=best.cpu().numpy();expected=z[steps[lo:lo+len(b)],np.arange(lo,lo+len(b))]
                error=max(error,float(np.max(np.abs(b-expected))))
                reported=np.array([r[key]['best_logit'] for r in rows[lo:lo+len(b)]])
                if not np.allclose(b,reported,atol=3e-6,rtol=0):raise ValueError('record mismatch')
                probabilities.extend(best.sigmoid().cpu().tolist())
            reached=sorted({r['shot'] for r,pb in zip(rows,probabilities) if r['flag'] and not r['cold_timely'] and pb>=parent['threshold']})
            lowered=[r['shot'] for r,pb in zip(rows,probabilities) if not r['flag'] and r['cold_alarm'] and pb<parent['threshold']]
            if error>3e-6 or reached!=d['groups'][key]['missed_shots_with_timely_point_crossing'] or lowered!=d['groups'][key]['safe_fp_peak_below_threshold']:
                raise ValueError('late site result mismatch')
            groups[key]=dict(missed_point_crossings=len(reached),safe_fp_peak_reductions=len(lowered),max_replay_logit_error=error)
    result=dict(complete=True,late_site_sha256=sha(root/'late_site.json'),groups=groups,
                points=len(rows),audit_code_sha256=sha(Path(__file__)),online_success=False)
    _durable_json(root/'late_audit.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
