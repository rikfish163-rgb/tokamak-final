"""Single input counterfactual for temporal_depth. No change to source, readout protocol or donors."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.special import logit
import torch

from temporal_depth import (ARM,ORDER_HASH,SOURCE_HASH,Bag,Learner,backbone,choose,contract,data,digest,dump,
    endpoints,engine,exact_z,experiments,flips,pack,report,votes,windows)


def run():
    torch.manual_seed(133);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    ids=data._east360();assert sha256(','.join(map(str,ids)).encode()).hexdigest()==ORDER_HASH
    src,ck=backbone.load_source_model();cfg=experiments.locked_configs(float(ck['val_threshold_fpr5']))[ARM];sig=contract(cfg)
    assert sig['source_sha256']==SOURCE_HASH and cfg.hold_ms==1;sc,ss=experiments.source_coords(ck,cfg.align_mode)
    st=engine.InterShotState.from_source(sc,ss,cfg);models={s+'_'+k:Learner(src,s) for s in ('H','HE','HE2') for k in ('eq','near')}
    bank={True:[],False:[]};rows=[];train=[];seen=[]
    for turn,sid in enumerate(ids,1):
        stats={k:[0.]*6 for k in models};donors=[]
        if bank[True] and bank[False]:
            ps,ns=(choose(bank[y],turn) for y in (True,False));bags=ps+ns;donors=[[b.sid for b in c] for c in (ps,ns)]
            h=max(1,seen[-1]-seen[-21:][0]);ten=lambda xx:torch.as_tensor(np.stack(xx),device='cuda')
            xx=ten([b.x for b in bags]).flatten(0,1);aa=ten([b.a for b in bags]).flatten(0,1);line=ten([b.line for b in bags]).float();base=ten([b.z0 for b in bags]).float()
            for n,m in models.items():
                w=[torch.tensor(votes(sid,c,h,n.split('_')[1]),device='cuda',dtype=torch.float32) for c in donors];stats[n]=m.step(xx,aa,line,base,*w)
        s=data.read_signals('EAST',sid);t,v,a=(s[k] for k in ('times','values','available'));captured=[]
        def native(x,ma):
            p,z=exact_z(src,x,ma);captured.append(z);return p
        parent=engine.run_shot_causal(sid,t,v,a,native,st,cfg,sc,ss);x=parent.aligned_features;z0=captured[0];L=logit(parent.threshold_curve)
        curves={'parent':z0.astype(float)-L};ds={};cache={}
        for n,m in models.items():
            z=exact_z(m.net,x.copy(),a)[1] if donors else z0;d=z.astype(float)-z0;curves[n]=z.astype(float)-L
            curves[n+'_bias']=curves['parent']+stats[n][4];ds[n]=float(np.sqrt(np.mean(d*d)))
            if n in ('HE_near','HE2_near'):cache[n]=d
        goal=min(stats[n][5] for n in cache)
        for n,d in cache.items(): curves[n+'_dose']=curves['parent']+d*(goal/stats[n][5] if stats[n][5] else 1.)
        first={};seal=sha256(t.tobytes()+L.tobytes())
        for n,m in curves.items():
            hits=np.flatnonzero(m>=0);first[n]=float(t[hits[0]]) if len(hits) else None;seal.update(n.encode());seal.update(m.tobytes())
        assert first['parent']==parent.alarm_time
        r=dict(rank=turn,shot=sid,first=first,drift_rms=ds,pre_reveal_sha=seal.hexdigest());rows.append(r)
        y,down=data.read_outcome('EAST',sid);elig=np.round((down-t)*1000,6)>30 if y else np.ones(len(t),bool)
        r.update(down=down,peak={k:float(m[elig].max()) for k,m in curves.items()})
        ix=endpoints(t,v,a,down,logit(np.clip(parent.risk_curve,1e-15,1-1e-15))-L)
        bank[y].append(Bag(sid,windows(x,ix),windows(a,ix).astype(bool),L[ix].copy(),z0[ix].copy()));seen.append(sid)
        train.append(dict(for_shot=sid,donors=donors,after_reveal_indices=ix,stats=stats));engine.reflect_and_update(parent,v,a,y,down,st,cfg,sc,ss)
        if turn%90==0: print(f'{turn}/360 old-input releases sealed',flush=True)
    main=json.loads(Path(__file__).with_name('temporal_depth.json').read_text())
    for r,rr,u,uu in zip(rows,main['ledger'],train,main['training'],strict=True):
        assert (r['shot'],r['first']['parent'],r['down'])==(rr['shot'],rr['first']['parent'],rr['down'])
        assert u['donors']==uu['donors'] and np.array_equal(u['after_reveal_indices'],uu['after_reveal_indices'])
    assert pack(st)==json.loads(Path(__file__).with_name('intershot360.json').read_text())['checkpoint_states']['360']['old_joint']
    names=list(rows[0]['first']);pairs=[('parent',n) for n in models]+[('HE_'+k,'HE2_'+k) for k in ('eq','near')]+[('HE_near_dose','HE2_near_dose')]
    return dict(contract=sig,order=ORDER_HASH,script=digest(__file__),training_script=main['script'],ledger=rows,training=train,audit={k:m.audit(src) for k,m in models.items()},
        all360=report(rows,names),blocks=[report(rows[i:i+90],names) for i in range(0,360,90)],paired={a+'__'+b:flips(rows,a,b) for a,b in pairs})


if __name__=='__main__':
    result=run();dump(Path(__file__).with_suffix('.json'),result)
    for k,v in result['all360'].items():
        if not k.endswith('_bias'): print(k,v['counts'],flush=True)
