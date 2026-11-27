"""Train on completed bags, freeze each released model, rescore disjoint past normal discharges."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.special import logit
import torch

from temporal_depth import (ARM,ORDER_HASH,SOURCE_HASH,Bag,Learner,backbone,choose,contract,data,digest,dump,
    endpoints,engine,exact_z,experiments,flips,pack,report,votes,weighthash,windows)

NAMES=('parent_cal','raw0_cal','H_near','HE_eq','HE_near','HE2_eq','HE2_near')


def run():
    torch.set_num_threads(2);torch.manual_seed(133);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    ids=data._east360();assert sha256(','.join(map(str,ids)).encode()).hexdigest()==ORDER_HASH
    src,ck=backbone.load_source_model();cfg=experiments.locked_configs(float(ck['val_threshold_fpr5']))[ARM]
    prov=contract(cfg);assert prov['source_sha256']==SOURCE_HASH and cfg.hold_ms==1
    sc,ss=experiments.source_coords(ck,cfg.align_mode);state=engine.InterShotState.from_source(sc,ss,cfg)
    models={n:Learner(src,n.split('_')[0]) for n in NAMES[2:]};bank={False:[],True:[]};cal=[];released={};history=[]
    rows=[];fits=[];notes=[];offset={};cutrank=0
    for turn,sid in enumerate(ids,1):
        active=bool(bank[True] and bank[False]);donors=[]
        if active:
            ps,ns=(choose(bank[y],turn) for y in (True,False));bags=ps+ns;donors=[[b.sid for b in q] for q in (ps,ns)]
            h=max(1,history[-1]-history[-21:][0]);ten=lambda xx:torch.as_tensor(np.stack(xx),device='cuda')
            xx=ten([b.x for b in bags]).flatten(0,1);aa=ten([b.a for b in bags]).flatten(0,1);line=ten([b.line for b in bags]).float();base=ten([b.z0 for b in bags]).float()
            for k,model in models.items():
                w=[torch.tensor(votes(sid,c,h,k.split('_')[1]),dtype=torch.float32,device='cuda') for c in donors];model.step(xx,aa,line,base,*w)
        if (turn-1)%20==0:
            cutrank=int(np.ceil(.8*(len(cal)+1)));ready=active and cutrank<=len(cal)
            released={k:deepcopy(m.net).eval() for k,m in models.items()} if ready else {}
            peaks={k:[] for k in NAMES};offset={}
            if ready:
                for c,x,a,L,pm,rawm in cal:
                    peaks['parent_cal'].append(pm);peaks['raw0_cal'].append(rawm)
                    for k,net in released.items(): peaks[k].append(float(np.max(exact_z(net,x,a)[1].astype(float)-L)))
                for k,p in peaks.items(): offset[k]=float(np.nextafter(np.sort(p)[cutrank-1],np.inf))
            notes.append(dict(rank=turn,shot=sid,past_training_max=max((b.sid for bb in bank.values() for b in bb),default=None),
                train_classes=[len(bank[True]),len(bank[False])],cal_ids=[c[0] for c in cal],order_stat=cutrank,offset=offset.copy(),
                peaks=peaks,state_hash={k:weighthash(m) for k,m in released.items()}))
        fits.append(donors)
        sig=data.read_signals('EAST',sid);t,v,a=(sig[k] for k in ('times','values','available'))
        pred=engine.run_shot_causal(sid,t,v,a,lambda x,m:backbone.predict_shot_proba(src,x,m),state,cfg,sc,ss)
        L=logit(pred.threshold_curve);parent=logit(np.clip(pred.risk_curve,1e-15,1-1e-15))-L
        x=pred.aligned_features.copy();x[:,9]=v[:,9];rawz=exact_z(src,x,a)[1];raw=rawz.astype(float)-L
        curve={'parent':parent}
        for k in NAMES:
            if not offset: curve[k]=parent;continue
            m=parent if k=='parent_cal' else raw if k=='raw0_cal' else exact_z(released[k],x,a)[1].astype(float)-L
            curve[k]=m-offset[k]
        first={};seal=sha256(t.tobytes()+L.tobytes());
        for k,m in curve.items():
            ix=np.flatnonzero(m>=0.);first[k]=float(t[ix[0]]) if len(ix) else None;seal.update(k.encode());seal.update(m.tobytes())
        assert first['parent']==pred.alarm_time
        row=dict(rank=turn,shot=sid,first=first,pre_reveal_sha=seal.hexdigest());rows.append(row)
        y,down=data.read_outcome('EAST',sid);eligible=np.round((down-t)*1000,6)>30 if y else np.ones(len(t),bool)
        row.update(down=down,peak={k:float(np.max(m[eligible])) for k,m in curve.items()})
        if not y and turn%3==0: cal.append((sid,x.copy(),a.copy(),L.copy(),float(parent.max()),float(raw.max())))
        else:
            ix=endpoints(t,v,a,down,parent);bank[y].append(Bag(sid,windows(x,ix),windows(a,ix).astype(bool),L[ix].copy(),rawz[ix].copy()))
        engine.reflect_and_update(pred,v,a,y,down,state,cfg,sc,ss);history.append(sid)
        if turn%90==0: print(f'{turn}/360 sealed with held-out normal calibration; normal holdout={len(cal)}',flush=True)
    orig=json.loads(Path(__file__).with_name('current_path.json').read_text())
    assert all(r['first']['parent']==b['firsts']['parent_shared'] and r['down']==b['down'] for r,b in zip(rows,orig['rows'],strict=True))
    assert pack(state)==json.loads(Path(__file__).with_name('intershot360.json').read_text())['checkpoint_states']['360']['old_joint']
    assert not {c[0] for c in cal}&{b.sid for group in bank.values() for b in group}
    names=('parent',*NAMES);pairs=[('parent',k) for k in NAMES]+[('HE_near','HE2_near'),('HE_eq','HE2_eq'),('H_near','HE_near'),('HE_eq','HE_near'),('HE2_eq','HE2_near')]
    return dict(contract=prov,script=digest(__file__),training_script=digest(Path(__file__).with_name('temporal_depth.py')),order=ORDER_HASH,
        ledger=rows,training_donors=fits,releases=notes,audit={k:m.audit(src) for k,m in models.items()},all360=report(rows,names),
        blocks=[report(rows[i:i+90],names) for i in range(0,360,90)],paired={a+'__'+b:flips(rows,a,b) for a,b in pairs})


if __name__=='__main__':
    res=run();dump(Path(__file__).with_suffix('.json'),res)
    for k,v in res['all360'].items(): print(k,v['counts'],flush=True)
