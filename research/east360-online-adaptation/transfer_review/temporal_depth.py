"""Ordered EAST360 comparison. Only completed bags can tune the next discharge's source copy."""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, logit
import torch
from torch.nn import functional as F

from current_path import exact_z
from paired_forensics import (ARM,KINDS,ORDER_HASH,SOURCE_HASH,backbone,classify,contract,
    cut_at_budget,data,digest,dump,engine,experiments,pack,windows)

OUT=Path(__file__).with_suffix('.json')
SCOPES=('H','HE','HE1','HE2','HE12','all')
CELLS=[(s,k) for s in SCOPES for k in ('eq','near')]+[(s,'rev') for s in ('HE','HE2')]


@dataclass
class Bag:
    sid:int
    x:np.ndarray
    a:np.ndarray
    line:np.ndarray
    z0:np.ndarray


def choose(bank,turn):
    if len(bank)<=8: return list(bank)
    groups=np.array_split(np.arange(len(bank)-1),7)
    return [bank[g[(turn+j)%len(g)]] for j,g in enumerate(groups)]+[bank[-1]]


def votes(sid,bagids,h,kind):
    assert len(bagids)>0 and max(bagids)<sid
    w=1/(1+(sid-np.asarray(bagids))/h) if kind!='eq' else np.ones(len(bagids))
    if kind=='rev': w=w[::-1]
    return w/w.sum()


def endpoints(t,v,a,down,parent_margin):
    if down is not None:
        lead=np.round(1000*(down-t),6); weak=np.flatnonzero((lead>30)&(lead<=125))
        if len(weak)<8: raise ValueError('insufficient positive grid: do not invent windows')
        return weak[np.linspace(0,len(weak)-1,8,dtype=int)]
    x=v[:,9].astype(float); g=np.full(len(t),-np.inf); count=np.cumsum(np.r_[0,a[:,9]].astype(int))
    ok=(count[128:]-count[:-128]==128)&(x[:-127]!=0); ii=np.flatnonzero(ok)
    g[ii+127]=(x[ii]-x[ii+127])/abs(x[ii]); chosen=[]; allowed=np.ones(len(t),bool)
    for curve in (parent_margin,g):
        for _ in range(4):
            i=int(np.argmax(np.where(allowed,curve,-np.inf)))
            if not allowed[i] or not np.isfinite(curve[i]): raise ValueError('insufficient negative witnesses')
            chosen.append(i); allowed &= abs(t-t[i])*1000>=16-1e-7
    return np.array(chosen)


class Learner:
    def __init__(self,src,scope):
        self.net=deepcopy(src).eval(); parent=dict(src.named_parameters()); self.active=[]; self.scope=scope
        self.mask=(torch.arange(368,device='cuda')%46<23)[None,:]
        for name,p in self.net.named_parameters():
            on=(name=='risk_head.weight' or scope=='all' or ('E' in scope and name=='embedding.weight')
                or ('1' in scope and name.startswith('encoder.layers.0.')) or ('2' in scope and name.startswith('encoder.layers.1.')))
            p.requires_grad_(on and name!='risk_head.bias')
            if p.requires_grad: self.active.append((name,p,parent[name].detach()))
        self.opt=torch.optim.Adam([p for _,p,_ in self.active],lr=3e-4,weight_decay=0.,foreach=False)

    def step(self,x,a,line,base,wp,wn):
        self.opt.zero_grad(set_to_none=True); n=len(wp)
        z=self.net(x,a).reshape(-1,8); m=z-line
        pos=m[:n].logsumexp(1)-np.log(8.); neg=m[n:].amax(1)
        ranking=(F.softplus(neg[None,:]-pos[:,None])*wp[:,None]*wn[None,:]).sum()
        anchor=sum((p-ref).square().sum() for _,p,ref in self.active)
        obj=ranking+.005*anchor
        if not torch.isfinite(obj): raise FloatingPointError('invalid objective')
        obj.backward()
        if self.scope!='all' and 'E' in self.scope: self.net.embedding.weight.grad.mul_(self.mask)
        torch.nn.utils.clip_grad_norm_([p for _,p,_ in self.active],1.,error_if_nonfinite=True)
        with torch.no_grad():
            old=[p.clone() for _,p,_ in self.active]; before=self.net(x,a).reshape(-1,8)
            self.opt.step(); factor=1.
            for _ in range(12):
                after=self.net(x,a).reshape(-1,8); step=(after-before).square().mean().sqrt().item()
                if step<=.150001: break
                scale=min(.90,.149/step);factor*=scale
                for (_,p,_),o in zip(self.active,old): p.copy_(o+(p-o)*scale)
            else: raise AssertionError('unbounded step')
            m=after-line; newloss=(F.softplus(m[n:].amax(1)[None,:]-(m[:n].logsumexp(1)-np.log(8.))[:,None])*wp[:,None]*wn[None,:]).sum()
            delta=after-base
            if not torch.isfinite(delta).all(): raise FloatingPointError('invalid proposed scores')
            if self.scope!='all' and 'E' in self.scope:
                _,p,ref=next(z for z in self.active if z[0]=='embedding.weight'); assert torch.equal(p[:,~self.mask[0]],ref[:,~self.mask[0]])
            return [float(ranking.detach()),float(newloss),step,factor,float(delta.mean()),float(delta.square().mean().sqrt())]

    def audit(self,src):
        ref=dict(src.named_parameters()); allowed={k for k,_,_ in self.active}; changed={}; total=0
        for name,p in self.net.named_parameters():
            d=p.detach()-ref[name].detach(); n=int(torch.count_nonzero(d)); assert not(n and name not in allowed),name
            if name in allowed:
                if self.scope!='all' and name=='embedding.weight':
                    assert not torch.count_nonzero(d[:,~self.mask[0]]); total+=64*8*23
                else: total+=p.numel()
                changed[name]=[n,float(torch.linalg.vector_norm(d))]
        for (k,v),(l,u) in zip(src.named_buffers(),self.net.named_buffers(),strict=True): assert k==l and torch.equal(v,u)
        return dict(trainable=total,changed=changed,sha=weighthash(self.net))


def weighthash(net):
    h=sha256()
    for k,p in net.state_dict().items(): h.update(k.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def report(rows,names):
    out={}
    for name in names:
        sub=[r for r in rows if name in r['first']]; outcomes=[classify(r['first'][name],r['down']) for r in sub]
        s=Counter(outcomes); counts={k:s[k] for k in KINDS}; ds=s['TP_timely']+s['TARDY']+s['MISS']; ns=s['FP']+s['TN']
        leads=[round(1000*(r['down']-r['first'][name]),6) for r,k in zip(sub,outcomes) if k=='TP_timely']
        cuts={}
        for p in (10,15,20):
            c=cut_at_budget([r['peak'][name] for r in sub if r['down'] is None],p)
            cuts[p]=[sum(r['peak'][name]>=c for r in sub if r['down'] is not None),sum(r['peak'][name]>=c for r in sub if r['down'] is None)]
        out[name]=dict(counts=counts,D=ds,N=ns,pass_percent={p:100*s['FP']<p*ns for p in (10,15,20)},
            timely_lead_bins=[sum(t<=125 for t in leads),sum(125<t<=3000 for t in leads),sum(t>3000 for t in leads)],posthoc=cuts)
    return out


def flips(rows,left,right):
    result={k:[] for k in ('gain_D','lose_D','add_FP','drop_FP')}
    for r in rows:
        a,b=(classify(r['first'][n],r['down']) for n in (left,right));sid=r['shot']
        if (a=='TP_timely')!=(b=='TP_timely'): result['gain_D' if b=='TP_timely' else 'lose_D'].append(sid)
        if (a=='FP')!=(b=='FP'): result['add_FP' if b=='FP' else 'drop_FP'].append(sid)
    return result


def main(output):
    torch.set_num_threads(2);torch.manual_seed(133);torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    ids=data._east360();assert sha256(','.join(map(str,ids)).encode()).hexdigest()==ORDER_HASH
    src,ck=backbone.load_source_model();cfg=experiments.locked_configs(float(ck['val_threshold_fpr5']))[ARM]
    provenance=contract(cfg);assert provenance['source_sha256']==SOURCE_HASH and cfg.hold_ms==1
    assert src.cfg==dict(window=128,patch=8,dim=64,heads=4,layers=2,ffn=128)
    sc,ss=experiments.source_coords(ck,cfg.align_mode); state=engine.InterShotState.from_source(sc,ss,cfg); src0=weighthash(src)
    models={s+'_'+k:Learner(src,s) for s,k in CELLS}; frozen={}; snap={};bank={y:[] for y in (True,False)};rows=[];seen=[]; train=[]
    for turn,sid in enumerate(ids,1):
        stats={n:[0.]*6 for n in models}; used=[];h=max(1,seen[-1]-seen[-21:][0]) if seen else 1; weights={}
        if bank[True] and bank[False]:
            ps,ns=(choose(bank[y],turn) for y in (True,False));bags=ps+ns;used=[[b.sid for b in part] for part in (ps,ns)]
            ten=lambda xx:torch.as_tensor(np.stack(xx),device='cuda')
            x=ten([b.x for b in bags]).flatten(0,1); a=ten([b.a for b in bags]).flatten(0,1)
            line=ten([b.line for b in bags]).float();base=ten([b.z0 for b in bags]).float()
            for k in ('eq','near','rev'): weights[k]=[votes(sid,g,h,k) for g in used]
            for name,model in models.items():
                wp,wn=(torch.tensor(w,dtype=torch.float32,device='cuda') for w in weights[name.split('_')[1]])
                stats[name]=model.step(x,a,line,base,wp,wn)
        if turn in (91,271):
            key='freeze_'+str(turn-1); frozen[key]=deepcopy(models['HE2_near'].net).eval();snap[key]=weighthash(frozen[key])
        sig=data.read_signals('EAST',sid); t,v,mask=(sig[k] for k in ('times','values','available'))
        pred=engine.run_shot_causal(sid,t,v,mask,lambda vv,aa:backbone.predict_shot_proba(src,vv,aa),state,cfg,sc,ss)
        fresh=pred.aligned_features.copy();fresh[:,9]=v[:,9]; rawp,rawz=exact_z(src,fresh,mask)
        L=logit(pred.threshold_curve); margins={};first={};drifts={}; seal=sha256(t.tobytes()+L.tobytes())
        def issue(n,z,line):
            m=z.astype(float)-line;hits=np.flatnonzero(m>=0);first[n]=float(t[hits[0]]) if len(hits) else None
            margins[n]=m;seal.update(n.encode());seal.update(m.tobytes())
        # Use unclipped probability logits when representable; overflow endpoints have no role near the line.
        parent_z=logit(np.clip(pred.risk_curve,1e-15,1-1e-15));issue('parent',parent_z,L);assert first['parent']==pred.alarm_time
        def ownline(p): return logit(engine._detect(p.astype(float),t,cfg,state.multiplier)[1])
        issue('raw0',rawz,L);issue('raw0_own',rawz,ownline(rawp));cache={}
        for n,model in models.items():
            p,z=exact_z(model.net,fresh,mask) if used else (rawp,rawz);issue(n,z,L);issue(n+'_own',z,ownline(p))
            delta=z.astype(float)-rawz;drifts[n]=[float(delta.mean()),float(np.sqrt(np.mean(delta**2))),float(abs(delta).max())]
            issue(n+'_bias',rawz.astype(float)+stats[n][4],L)
            if n in ('HE_near','HE2_near'): cache[n]=delta
        lower=min(stats[n][5] for n in cache)
        for n,delta in cache.items():
            factor=min(1.,lower/stats[n][5]) if stats[n][5]>0 else 1.;issue(n+'_dose',rawz.astype(float)+factor*delta,L)
        if turn>270:
            for n,model in frozen.items(): issue(n,exact_z(model,fresh,mask)[1],L)
        row=dict(rank=turn,shot=sid,first=first,pre_reveal_sha=seal.hexdigest(),drift=drifts);rows.append(row)
        y,down=data.read_outcome('EAST',sid)
        eligible=np.round(1000*(down-t),6)>30 if y else np.ones(len(t),bool);assert eligible.any()
        row.update(down=down,peak={n:float(m[eligible].max()) for n,m in margins.items()})
        picks=endpoints(t,v,mask,down,margins['parent']); xp=windows(fresh,picks);ap=windows(mask,picks).astype(bool)
        bank[y].append(Bag(sid,xp,ap,L[picks].copy(),rawz[picks].copy()))
        train.append(dict(for_shot=sid,halfspan=h,donors=used,near_weights=weights.get('near'),after_reveal_indices=picks,stats=stats));seen.append(sid)
        engine.reflect_and_update(pred,v,mask,y,down,state,cfg,sc,ss)
        if turn%90==0: print(f'{turn}/360 all releases sealed; seen D/N {len(bank[True])}/{len(bank[False])}',flush=True)
    reference=json.loads(Path(__file__).with_name('current_path.json').read_text())
    for r,q in zip(rows,reference['rows'],strict=True):
        assert (r['shot'],r['down'],*[r['first'][k] for k in ('parent','raw0','raw0_own')])==(q['shot'],q['down'],*[q['firsts'][k] for k in ('parent_shared','raw_shared','raw_reanchored')])
    assert pack(state)==json.loads(Path(__file__).with_name('intershot360.json').read_text())['checkpoint_states']['360']['old_joint']
    assert src0==weighthash(src) and all(weighthash(frozen[n])==h for n,h in snap.items())
    names=list(rows[0]['first']);pairs=[('parent',n) for n in models]+[('raw0',n) for n in models]+[(s+'_eq',s+'_near') for s in SCOPES]
    pairs += [('HE_'+w,'HE2_'+w) for w in ('eq','near','rev')]+[('HE_near_dose','HE2_near_dose'),('HE1_near','HE2_near'),('HE2_rev','HE2_near')]
    result=dict(contract=provenance,order=ORDER_HASH,script=digest(__file__),source_parameters=src0,training_steps=sum(bool(u['donors']) for u in train),
        training_stats=['ranking_before','ranking_after','step_rms','step_fraction','past_mean_change','past_rms_change'],
        audit={n:m.audit(src) for n,m in models.items()},snapshots=snap,all360=report(rows,names),
        blocks=[report(rows[i:i+90],names) for i in range(0,360,90)],suffix=report(rows[270:],['parent','raw0','HE_near','HE2_near',*snap]),
        paired={a+'__'+b:flips(rows,a,b) for a,b in pairs},suffix_flips={n:flips(rows[270:],n,'HE2_near') for n in snap},
        training=train,ledger=rows)
    dump(output,result)
    for n in ('parent','raw0',*models): print(n,result['all360'][n]['counts'],flush=True)


if __name__=='__main__': main(Path(sys.argv[1]) if len(sys.argv)>1 else OUT)
