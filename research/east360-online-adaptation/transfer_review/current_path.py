"""One-channel follow-up to the observed EAST current-input saturation, not a replacement engine."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.special import logit

from paired_forensics import (classify, contract, cut_at_budget, data, digest, dump, engine,
    experiments, backbone, ARM, GROUPS, ORDER_HASH, SOURCE_HASH, inspect_windows, loadshot, pack, windows)


def held_level(x,a):
    ids=np.flatnonzero(a); out=np.zeros_like(x)
    if len(ids): out[a]=x[ids[0]]  # no valid output exists before the anchor
    return out


def exact_z(model, aligned, valid):
    base=[]
    def tap(_m,_a,o): base.append(o[:,0].detach().cpu().numpy())
    h=model.risk_head.register_forward_hook(tap)
    try: p=backbone.predict_shot_proba(model,aligned,valid)
    finally: h.remove()
    return p,np.concatenate(base)


def run(tmp):
    idx=json.loads((tmp/'index.json').read_text()); ids=data._east360()
    assert idx['order_hash']==ORDER_HASH and sha256(','.join(map(str,ids)).encode()).hexdigest()==ORDER_HASH
    model,ck=backbone.load_source_model(); cfg=experiments.locked_configs(float(ck['val_threshold_fpr5']))[ARM]
    assert contract(cfg)==idx['contract'] and idx['contract']['source_sha256']==SOURCE_HASH
    sc,ss=engine._source(*experiments.source_coords(ck,cfg.align_mode),cfg)
    teacher,boot=(engine.InterShotState.from_source(sc,ss,cfg) for _ in range(2)); rows=[]; snapshots={}
    for rank,sid in enumerate(ids,1):
        s=loadshot(tmp,sid); t,v,a=s['t'],s['v'],s['a']; old=s['av']; Z=s['z']; M=teacher.multiplier
        # Same float32 sigmoid implementation as original; verify its pre-outcome digest below.
        parent_p=torch.as_tensor(Z,device='cuda').sigmoid().cpu().numpy()
        pred=engine.run_shot_causal(sid,t,v,a,lambda x,m:parent_p,teacher,cfg,sc,ss)
        assert np.array_equal(pred.aligned_features,old)
        assert np.array_equal(logit(pred.threshold_curve),s['line'])
        receipt=sha256(t.tobytes()+pred.risk_curve.tobytes()+pred.threshold_curve.tobytes()+s['h'].tobytes()).hexdigest()
        aligned=engine._align(t,np.where(a,v.astype(float),np.nan),boot,cfg,sc,ss)[0]
        aligned=np.where(a,aligned,0.).astype('float32'); others=np.arange(23)!=9
        assert np.array_equal(aligned[:,others],old[:,others])
        currents={'parent':old[:,9], 'boot':aligned[:,9], 'raw':v[:,9], 'level':held_level(v[:,9],a[:,9])}
        scores={}; actual={}; margins={}; seal=sha256(receipt.encode()); effect=None
        for name,ip in currents.items():
            if np.array_equal(ip,old[:,9]): p,z=parent_p,Z
            else:
                x=old.copy(); x[:,9]=ip; p,z=exact_z(model,x,a)
            scores[name]=p; _,native,_,_,_=engine._detect(p.astype(float),t,cfg,M)
            for typ,line in (('shared',pred.threshold_curve),('reanchored',native)):
                key=name+'_'+typ; ix=np.flatnonzero(p>=line); actual[key]=None if not len(ix) else float(t[ix[0]])
                margins[key]=z.astype(float)-logit(line)
                assert np.array_equal(p>=line,margins[key]>=0)
                seal.update(p.tobytes()); seal.update(line.tobytes())
        assert actual['parent_shared']==actual['parent_reanchored']==pred.alarm_time
        # What actually drove the first crossing? No outcome in this forward sensitivity.
        if pred.alarm_idx is not None:
            ii=np.array([pred.alarm_idx]); _,orig,dz=inspect_windows(model,windows(old,ii),windows(a,ii))
            assert abs(orig[0]-Z[ii[0]])<.001
            margin=float(margins['parent_shared'][ii[0]])
            effect=dict(margin=margin,dz=dz[0],held_drops_first_point=(margin-dz[0])<0)
        r=dict(shot=sid,rank=rank,firsts=actual,seal=seal.hexdigest(),parent_attested=receipt)
        y,down=data.read_outcome('EAST',sid)
        assert receipt==idx['rows'][rank-1]['seal'] and actual['parent_shared']==idx['rows'][rank-1]['first']
        r.update(down=down,first_point=effect,kind=classify(pred.alarm_time,down))
        eligible=(np.round((down-t)*1000,6)>30 if y else np.ones(len(t),bool))
        r['peak']={k:float(np.max(vv[eligible])) for k,vv in margins.items()}
        cross=(margins['parent_shared']>=0)&eligible; ev=np.flatnonzero(np.diff(np.r_[False,cross,False])).reshape(-1,2)
        # data.read_signals guarantees a common uniform clock for this archive.
        r['longest_eligible_run_ms']=max((1000*(t[j-1]-t[i])+cfg.sample_ms for i,j in ev),default=0.)
        for name,sel in [('after200',a[:,9]&((t-t[0])*1000>200))]+([('at65',
                a[:,9]&(np.arange(len(t))<=np.searchsorted(t,down-.065,side='right')-1)&
                (np.arange(len(t))>=np.searchsorted(t,down-.065,side='right')-128))] if y else []):
            assert sel.sum()>1 and np.ptp(v[sel,9])>0
            r[name]={k:float(np.ptp(x[sel].astype(float))) for k,x in currents.items()}
        rows.append(r)
        initial=boot.target_n_eff[9]==0
        for state in (teacher,boot): engine.reflect_and_update(pred,v,a,y,down,state,cfg,sc,ss)
        if initial and boot.target_n_eff[9]>0:
            safe=((t-t[0])*1000<=cfg.ref_ms)&(t<down-.250) if y else t<t[-1]-.050
            observed=v[safe&a[:,9],9].astype(float); center=float(np.median(observed))
            assert teacher.target_center[9]==np.clip(center,-1e5,1e5)
            boot.target_center[9]=center
        assert teacher.multiplier==boot.multiplier
        for k in ('target_center','target_n_eff','target_scale'): assert np.array_equal(getattr(teacher,k)[others],getattr(boot,k)[others])
        if rank in (1,2,20,90,180,270,360):
            snapshots[rank]={n:[x.target_center[9],x.target_scale[9],x.target_n_eff[9]] for n,x in (('parent',teacher),('boot',boot))}
        if rank%90==0: print(f'{rank}/360 one-channel interventions sealed',flush=True)
    assert pack(teacher)==json.loads(Path(__file__).with_name('intershot360.json').read_text())['checkpoint_states']['360']['old_joint']
    out={}; count=lambda kinds:dict(Counter(kinds))
    for k in rows[0]['firsts']:
        kinds=[classify(r['firsts'][k],r['down']) for r in rows]
        cutoff=cut_at_budget([r['peak'][k] for r in rows if r['down'] is None]); out[k]=dict(counts=count(kinds),
            blocks=[count(kinds[i:i+90]) for i in range(0,360,90)],posthoc_35fp=[sum(r['peak'][k]>=cutoff for r in rows if r['down'] is not None),sum(r['peak'][k]>=cutoff for r in rows if r['down'] is None)],
            flips={n:[r['shot'] for r,b in zip(rows,kinds) if test(r['kind'],b)] for n,test in {
            'gain_D':lambda a,b:a!='TP_timely' and b=='TP_timely','lose_D':lambda a,b:a=='TP_timely' and b!='TP_timely',
            'add_FP':lambda a,b:a!='FP' and b=='FP','remove_FP':lambda a,b:a=='FP' and b!='FP'}.items()})
    return dict(contract=idx['contract'],script_sha256=digest(__file__),forensics_script_sha256=digest(Path(__file__).with_name('paired_forensics.py')),
                order_sha256=ORDER_HASH,groups=list(GROUPS),snapshots=snapshots,summary=out,rows=rows)


if __name__=='__main__': dump(Path(__file__).with_suffix('.json'),run(Path(sys.argv[1])))
