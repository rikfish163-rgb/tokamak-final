"""Re-forward every saved QK/V endpoint witness, independent of optimization."""
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
from .model import AttentionAdapter


def run(root):
    if (root/'audit.json').exists():raise FileExistsError(root/'audit.json')
    p=read(root,'protocol.json');done=read(root,'complete.json');previous=Path(p['previous'])
    if not done['complete']:raise ValueError('completed capacity required')
    for path,h in done['artifacts'].items():
        if sha(root/path)!=h:raise ValueError('artifact changed')
    for path,h in p['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('code changed')
    for name in ('complete','records','audit'):
        if sha(previous/f'{name}.json')!=p[f'previous_{name}_sha256']:raise ValueError('previous changed')
    ref=Path(p['reference'])
    for path,h in p['reference_hashes'].items():
        if sha(ref/path)!=h:raise ValueError('parent changed')
    original=read(previous,'records.json')['records'];rows_by_mode={k:read(root,f'{k}_records.json') for k in p['modes']}
    for rows in rows_by_mode.values():
        if len(rows)!=len(original):raise ValueError('point count changed')
        for r,s in zip(rows,original):
            for key,value in s.items():
                if not key.startswith('radius') and key!='initial_logit' and r[key]!=value:raise ValueError('point changed')
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,parent=load_parent(DEFAULT_PARENT);base.cuda();net=AttentionAdapter(base)
    slow=torch.load(ref/'models/cold.pt',map_location='cpu',weights_only=True)['slow'].cuda()
    mapping=read(ref,'input_map.json');hashes=read(ref,'input_hashes.json');xs=[];masks=[]
    for sid in p['order'][20:]:
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('raw changed')
        raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        signal=transform({k:raw[k] for k in ('values','available')},mapping)
        x,m=batch(signal,[r['end'] for r in original if r['shot']==sid],'cpu');xs.append(x);masks.append(m)
    x=torch.cat(xs);mask=torch.cat(masks);groups={};sign=np.array([1 if r['flag'] else -1 for r in original])
    with np.load(root/'witnesses.npz') as w:
        for mode,rows in rows_by_mode.items():
            for radius in p['radii']:
                key=f'{mode}_radius{radius}';state=w[key+'_state'];history=w[key+'_logits'];norm=w[key+'_norms'];steps=w[key+'_steps']
                if (state.shape!=(len(rows),512) or history.shape!=(65,len(rows)) or norm.shape!=history.shape
                        or not np.isfinite(state).all() or not np.isfinite(history).all() or not np.isfinite(norm).all()
                        or (norm>radius+2e-6).any() or (np.linalg.norm(state,axis=1)>radius+2e-6).any()
                        or not np.array_equal(steps,(history*sign[None]).argmax(0))):raise ValueError('invalid witness')
                replay=[];matrix=[];error=0.
                for lo in range(0,len(rows),32):
                    st=torch.from_numpy(state[lo:lo+32]).cuda()
                    with torch.no_grad():
                        z=net(x[lo:lo+32].cuda(),mask[lo:lo+32].cuda(),st,slow,mode,forecast=False)['logit']
                        ds=net.deltas(st,mode);mn=sum(v.square().sum((1,2)) for v in ds.values()).sqrt()
                    zz=z.cpu().numpy();expected=history[steps[lo:lo+len(zz)],np.arange(lo,lo+len(zz))]
                    error=max(error,float(np.abs(zz-expected).max()));replay.extend(z.sigmoid().cpu().tolist());matrix.extend(mn.cpu().tolist())
                    if not np.allclose(zz,[r[f'radius{radius}']['best_logit'] for r in rows[lo:lo+len(zz)]],atol=3e-6,rtol=0):
                        raise ValueError('reported logits differ')
                if error>3e-6 or not np.allclose(matrix,[r[f'radius{radius}']['matrix_delta_frobenius'] for r in rows],atol=3e-6,rtol=1e-6):
                    raise ValueError('replay differs')
                reached=sorted({r['shot'] for r,pb in zip(rows,replay) if r['flag'] and not r['cold_timely'] and pb>=parent['threshold']})
                positive=sorted({r['shot'] for r,pb in zip(rows,replay) if r['flag'] and not r['cold_timely'] and 'positive_window_peak' in r['reasons'] and pb>=parent['threshold']})
                lowered=[r['shot'] for r,pb in zip(rows,replay) if not r['flag'] and r['cold_alarm'] and pb<parent['threshold']]
                declared=done['groups'][key]
                if (reached!=declared['missed_shots_with_timely_point_crossing'] or positive!=declared['missed_shots_with_30_125_point_crossing']
                        or lowered!=declared['safe_fp_peak_below_threshold']):raise ValueError('counts differ')
                groups[key]=dict(timely_point_crossings=len(reached),positive_window_crossings=len(positive),
                    safe_peak_reductions=len(lowered),max_replay_logit_error=error)
    result=dict(complete=True,complete_sha256=sha(root/'complete.json'),points=len(original),groups=groups,
                audit_code_sha256=sha(Path(__file__)),online_success=False)
    _durable_json(root/'audit.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
