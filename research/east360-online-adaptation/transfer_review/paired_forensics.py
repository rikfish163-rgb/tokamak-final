"""EAST360 only: repeat the sealed parent and diagnose exactly the same prefixes. No engine edits."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.special import expit, logit

from intershot360 import ORDER_HASH, SOURCE_HASH
from shot_memory import ARM, contract, digest, pack
from zero_cold_start import backbone, data, engine, experiments

OUT = Path(__file__).with_suffix('.json')
GROUPS = {'Ip_Vloop': (9,18), 'magnetic': (3,4,5,10,11), 'position': (7,8),
          'density': (12,), 'radiation': (1,14,15,16,17,19,20,21,22)}
KINDS = ('TP_timely','TARDY','MISS','FP','TN')


def dump(path, obj):
    def number(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x))
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(',',':'), default=number, allow_nan=False)+'\n')
    tmp.replace(path)


def classify(first, down):
    if down is None: return 'TN' if first is None else 'FP'
    if first is None or first>down+1e-9: return 'MISS'
    return 'TP_timely' if round(1000*(down-first),6)>30 else 'TARDY'


def collect(tmp):
    assert tmp.is_dir() and not any(tmp.iterdir()), 'use an empty private scratch directory'
    ids=data._east360(); assert sha256(','.join(map(str,ids)).encode()).hexdigest()==ORDER_HASH
    model,ck=backbone.load_source_model(); cfg=experiments.locked_configs(float(ck['val_threshold_fpr5']))[ARM]
    sig=contract(cfg); assert sig['source_sha256']==SOURCE_HASH and cfg.hold_ms==1
    sc,ss=experiments.source_coords(ck,cfg.align_mode); st=engine.InterShotState.from_source(sc,ss,cfg)
    h,z=[],[]
    def head(_mod, args, result):
        h.append(args[0].detach().cpu().numpy()); z.append(result[:,0].detach().cpu().numpy())
    handle=model.risk_head.register_forward_hook(head); rows=[]
    for rank,sid in enumerate(ids,1):
        s=data.read_signals('EAST',sid); t,v,a=(s[j] for j in ('times','values','available'))
        h.clear(); z.clear(); pred=engine.run_shot_causal(sid,t,v,a,lambda x,m:backbone.predict_shot_proba(model,x,m),st,cfg,sc,ss)
        H=np.concatenate(h); Z=np.concatenate(z); p=pred.risk_curve; tau=pred.threshold_curve
        assert H.shape==(len(t),64) and np.max(abs(expit(Z.astype(float))-p))<2e-7
        line=logit(tau); margin=Z.astype(float)-line
        assert np.array_equal(margin>=0,p>=tau)
        seal=sha256(t.tobytes()+p.tobytes()+tau.tobytes()+H.tobytes()).hexdigest()
        np.savez(tmp/f'{sid}.npz',t=t,v=v,a=a,av=pred.aligned_features,h=H,z=Z,line=line)
        first=pred.alarm_time; support=st.target_n_eff[list(cfg.align_channels)].copy()
        y,down=data.read_outcome('EAST',sid); label=classify(first,down)
        lead=np.round((down-t)*1000,6) if y else None
        elig=lead>30 if y else np.ones(len(t),bool); weak=elig&(lead<=125) if y else np.zeros(len(t),bool)
        def peak(x,mask): return float(x[mask].max()) if mask.any() else None
        rows.append(dict(rank=rank,shot=sid,block=(rank-1)//90,kind=label,first=first,down=down,
            lead=None if down is None or first is None else round((down-first)*1000,6),start=t[0],end=t[-1],rate=s['rate'],n=len(t),
            ip=abs(float(v[0,9])) if a[0,9] else None,mask0=a[0],support=support,eligible_n=elig.sum(),weak_n=weak.sum(),
            timely_peak_margin=peak(margin,elig),weak_peak_margin=peak(margin,weak),eligible_peak_z=peak(Z,elig),
            first_age=None if first is None else 1000*(first-t[0]),ref_risk=pred.ref_risk,
            post_line=peak(line,(t-t[0])*1000>=cfg.ref_ms),seal=seal))
        assert label==engine.reflect_and_update(pred,v,a,y,down,st,cfg,sc,ss).outcome
        if rank%90==0: print(f'{rank}/360 original alarms sealed',flush=True)
    handle.remove()
    old=json.loads(Path(__file__).with_name('outcome_credit.json').read_text())
    assert all((r['shot'],r['first'],r['down'])==(b[0],b[2]['parent'],b[4]) for r,b in zip(rows,old['ledger'],strict=True))
    assert pack(st)==json.loads(Path(__file__).with_name('intershot360.json').read_text())['checkpoint_states']['360']['old_joint']
    assert dict(Counter(r['kind'] for r in rows))==dict(zip(KINDS,(135,23,22,31,149)))
    dump(tmp/'index.json',dict(contract=sig,rows=rows,order_hash=ORDER_HASH))


LEADS=(125,65,31)
AGES=(250,500,1000)
REPS=('context','raw','aligned','cls','perp','source','margin')
TOK=('embedding_halves','layer1_halves','layer2_halves','layer1_cls')


def elapsed(r): return (r['down']-r['start'])*1000


def endpoint(r,age):
    end=(r['end']-r['start'])*1000
    if age<0 or age>end+1e-7: return None
    return min(r['n']-1,int(np.floor(age*r['rate']/1000+1e-8)))


def match(rows,left,right,mode):
    """Maximum cardinality, then minimum cost, per block. Eligibility never ranks by risk."""
    pairs=[]; missing=[]
    for b in range(4):
        L=[r for r in rows if r['block']==b and left(r)]; R=[r for r in rows if r['block']==b and right(r)]
        cost=np.full((len(L),len(R)),1e7)
        for i,p in enumerate(L):
            for j,q in enumerate(R):
                if not p['ip'] or not q['ip']: continue
                di=abs(np.log(p['ip']/q['ip']))
                if di>np.log(1.25): continue
                dt=0.
                if mode in ('DD','DN'):
                    # All three fixed landmarks must support a real 128-point trailing window.
                    if any((endpoint(p,elapsed(p)-h) or 0)<127 for h in LEADS): continue
                    endq=elapsed(q) if mode=='DD' else elapsed(p)
                    if any((endpoint(q,endq-h) or 0)<127 for h in LEADS): continue
                    if mode=='DD':
                        dt=abs(np.log(elapsed(p)/elapsed(q)))/np.log(2)
                        if dt>1: continue
                if mode=='ND' and (p['first_age']>elapsed(q)-31 or endpoint(q,p['first_age']) is None): continue
                dm=np.count_nonzero(np.array(p['mask0'])!=q['mask0'])
                ds=np.mean(abs(np.log1p(p['support'])-np.log1p(q['support'])))
                cost[i,j]=di/np.log(1.25)+dt+dm+ds+abs(p['rank']-q['rank'])/90
        chosen={}
        for i,j in zip(*linear_sum_assignment(cost)):
            if cost[i,j]<1e7: chosen[int(i)]=int(j)
        pairs.extend((p['shot'],R[chosen[i]]['shot']) for i,p in enumerate(L) if i in chosen)
        missing.extend(p['shot'] for i,p in enumerate(L) if i not in chosen)
    assert len({p for p,q in pairs})==len({q for p,q in pairs})==len(pairs)
    return pairs,missing


def landmarks(p,q,mode):
    if mode in ('DD','DN'):
        for h in LEADS:
            i=endpoint(p,elapsed(p)-h)
            j=endpoint(q,elapsed(q)-h if mode=='DD' else 1000*i/p['rate'])
            assert min(i,j)>=127
            yield str(h),i,j
    else:
        for name,age in [(str(v),v) for v in AGES]+[('first_FP',p['first_age'])]:
            i=endpoint(p,age); j=endpoint(q,age)
            if i is not None and j is not None and (mode!='ND' or elapsed(q)-1000*j/q['rate']>30): yield name,i,j


def halves(x):
    c=np.pad(np.cumsum(x,axis=0,dtype=float),((1,0),(0,0)))
    ends=np.arange(1,len(x)+1); recent=(c[ends]-c[np.maximum(0,ends-64)])/64
    old=np.zeros_like(recent); old[64:]=recent[:-64]
    return np.concatenate((old,recent),axis=1).astype('float32')


def features(d,ctx,params):
    cen,scl,u,_,_=params
    a=d['a']; mask=halves(a); n=len(a); out={}
    for name,key in (('raw','v'),('aligned','av')):
        warm=np.where(a,np.arcsinh((d[key].astype(float)-cen)/scl),0.)
        out[name]=np.column_stack((halves(warm),mask))
    H=d['h']; z=d['z']; line=d['line']; t=d['t']
    out.update(cls=H,perp=H-np.outer(H@u,u),source=z[:,None],margin=(z-line)[:,None],
               context=np.column_stack((np.log1p(1000*(t-t[0])),np.full(n,np.log1p(ctx['ip'] or 0.)),
                                        np.full(n,np.mean(ctx['support'])),mask)))
    return out


def windows(x,ii):
    jj=ii[:,None]-np.arange(127,-1,-1)
    return np.where((jj>=0)[:,:,None],x[np.maximum(0,jj)],0)


@torch.inference_mode()
def inspect_windows(model,windows_x,windows_a):
    """Layers and group holds at identical causal windows. No attention weights as explanations."""
    dev=next(model.parameters()).device
    x=torch.as_tensor(windows_x,device=dev); a=torch.as_tensor(windows_a,device=dev,dtype=torch.bool)
    b=len(x); obs=torch.where(a,x,model.center); xn=torch.asinh((obs-model.center)/model.scale)
    v=a.reshape(b,16,-1).any(-1); pad=~torch.cat((v,torch.ones_like(v[:,:1])),dim=1)
    emb=model.embedding(torch.cat((xn,a.to(x.dtype)),dim=-1).reshape(b,16,-1))
    tok=torch.cat((emb,model.cls.expand(b,-1,-1)),dim=1)+model.position
    def pool(tok):
        h=tok[:,:16].reshape(b,2,8,64); valid=v.reshape(b,2,8,1)
        return ((h*valid).sum(2)/valid.sum(2).clamp(min=1)).reshape(b,-1).cpu().numpy()
    layers={TOK[0]:pool(tok)}
    for i,layer in enumerate(model.encoder.layers):
        tok=layer(tok,src_key_padding_mask=pad); layers[TOK[i+1]]=pool(tok)
        if i==0: layers[TOK[3]]=tok[:,-1].cpu().numpy()
    h=model.norm(tok[:,-1]); original=model.risk_head(h)[:,0].cpu().numpy()
    pert=[]
    for cols in GROUPS.values():
        cols=list(cols); q=windows_x.copy(); ok=windows_a[:,95:96,cols]&windows_a[:,96:,cols]
        q[:,96:,cols]=np.where(ok,windows_x[:,95:96,cols],windows_x[:,96:,cols]); pert.append(q)
    qq=torch.as_tensor(np.concatenate(pert),device=dev)
    held=model(qq,a.repeat(len(pert),1,1)).reshape(len(pert),b).T.cpu().numpy()
    return layers,original,original[:,None]-held


def loadshot(tmp,sid):
    with np.load(tmp/f'{sid}.npz') as d: return dict(d)


def viewpoints(tmp,rows,plans,model,params):
    by={r['shot']:r for r in rows}; wants=defaultdict(set); grid={}
    for name,(mode,pairs,_) in plans.items():
        for p,q in pairs:
            picks=list(landmarks(by[p],by[q],mode)); grid[name,p,q]=picks
            for _,i,j in picks: wants[p].add(i); wants[q].add(j)
    views={}; feats={}; err=0.; proj=0.
    for sid,ind in wants.items():
        ii=np.array(sorted(ind)); d=loadshot(tmp,sid); fs=features(d,by[sid],params)
        xx,aa=windows(d['av'],ii),windows(d['a'],ii).astype(bool)
        layers,orig,occl=inspect_windows(model,xx,aa)
        err=max(err,float(np.max(abs(orig-d['z'][ii])))); cols=list(engine.COHERENT6)
        proj=max(proj,float(np.max(abs(fs['perp'][ii]@params[2]))))
        raw=windows(d['v'],ii); sc,ss=params[-2:]; clip=np.isclose(abs((xx[:,:,cols]-sc[cols])/ss[cols]),10,rtol=0,atol=2e-4)&aa[:,:,cols]
        def mean32(bl):
            am=aa[:,bl]; n=am.sum(1)
            return np.divide(np.where(am,raw[:,bl],0.).sum(1,dtype=float),n,out=np.full(n.shape,np.nan),where=n>0)
        changes=mean32(slice(96,128))-mean32(slice(64,96)); select=[9,18,3,7,8,12,15]
        for k,i in enumerate(ii):
            feats[sid,i]={**{name:f[i] for name,f in fs.items()},**{name:f[k] for name,f in layers.items()}}
            views[sid,i]=dict(t=d['t'][i],z=float(d['z'][i]),line=d['line'][i],hold_delta=occl[k],
                hold_valid=[int((aa[k,95,list(cc)][None,:]&aa[k,96:][:,list(cc)]).sum()) for cc in GROUPS.values()],
                clipped_samples=clip[k].sum(0),raw_change=[None if not np.isfinite(r) else float(r) for r in changes[k,select]])
    assert err<.001 and proj<1e-4,(err,proj)
    return grid,views,feats,dict(max_abs_native_vs_extracted_logit=err,max_abs_removed_head_projection=proj)


def fit_pair(A,B,signs=None):
    """Same ridge rule for every interface, no tuning. Swap labels without recomputing the factorization."""
    A,B=np.asarray(A,dtype=float),np.asarray(B,dtype=float)
    scale=np.maximum(np.concatenate((A,B)).std(axis=0),1e-6)
    D=(A-B)/scale; n=len(D)
    mat=np.linalg.solve(np.eye(D.shape[1])+D.T@D/n,D.T/n)/scale[:,None]
    return mat.sum(1),None if signs is None else mat@signs


def cut_at_budget(peaks,pct=20):
    budget=(len(peaks)*pct-1)//100
    return float(np.nextafter(sorted(peaks,reverse=True)[budget],np.inf))


def score_counts(rows): return {k:sum(r==k for r in rows) for k in KINDS}


def analyze(tmp):
    idx=json.loads((tmp/'index.json').read_text()); rows=idx['rows']; by={r['shot']:r for r in rows}
    model,ck=backbone.load_source_model(); cfg=experiments.locked_configs(float(ck['val_threshold_fpr5']))[ARM]
    assert idx['order_hash']==ORDER_HASH and idx['contract']==contract(cfg)
    weight=model.risk_head.weight.detach().cpu().numpy()[0].astype(float)
    params=(ck['norm_center'],ck['norm_scale'],weight/np.linalg.norm(weight),*experiments.source_coords(ck,cfg.align_mode))
    D=lambda r:r['down'] is not None; N=lambda r:not D(r); tp=lambda r:r['kind']=='TP_timely'
    fail=lambda r:r['kind'] in ('MISS','TARDY'); fp=lambda r:r['kind']=='FP'; tn=lambda r:r['kind']=='TN'
    specs={'Derror_TP':(fail,tp,'DD'),'FP_TN':(fp,tn,'NN'),'Derror_N':(fail,N,'DN'),
           'FP_TP':(fp,tp,'ND'),'allD_N':(D,N,'DN')}
    plans={n:(m,*match(rows,l,r,m)) for n,(l,r,m) in specs.items()}
    grid,views,fs,checks=viewpoints(tmp,rows,plans,model,params)
    paired={}; matched=[]
    for name,(mode,pairs,unmatched) in plans.items():
        details=[]
        for p,q in pairs:
            points={}
            for h,i,j in grid[name,p,q]:
                va=views[p,i]; vb=views[q,j]; points[h]=[va,vb]
                if name=='allD_N' and h=='65': matched.append((p,q,fs[p,i],fs[q,j]))
            item=dict(pair=[p,q],kinds=[by[p]['kind'],by[q]['kind']],points=points)
            if name in ('Derror_TP','FP_TN'):
                endline=by[q]['post_line']; d=loadshot(tmp,p); age=1000*(d['t']-d['t'][0]); late=age>=cfg.ref_ms
                if endline is not None:
                    hits=np.flatnonzero(d['z']>=np.where(late,endline,d['line'])); f=None if not len(hits) else float(d['t'][hits[0]])
                    item.update(case_first_on_control_postref_line=f,case_kind_on_control_postref_line=classify(f,by[p]['down']))
            details.append(item)
        paired[name]=dict(details=details,unmatched=unmatched)

    # The input for new block b is only matched pairs from blocks <b. Labels of block b
    # select evaluation pairs after the fact but never select training pairs, scales or lines.
    reps=(*REPS,*TOK); trainable=lambda q:by[q]['rank']%3!=0
    flip=np.random.default_rng(20261126).choice([-1.,1.],size=(len(matched),999)); W={}; folds={}
    tests={k:[] for k in reps}; null={k:np.zeros(999) for k in reps}
    for b in (1,2,3):
        fit=[j for j,(p,q,_,_) in enumerate(matched) if by[p]['block']<b and trainable(q)]
        test=[j for j,(p,*_) in enumerate(matched) if by[p]['block']==b]
        assert fit and test; W[b]={}
        folds[b]=dict(train_pairs=[[matched[j][0],matched[j][1]] for j in fit],test_pairs=len(test))
        assert max(max(by[matched[j][i]]['rank'] for i in (0,1)) for j in fit)<=90*b
        for k in reps:
            V=np.asarray([a[k] for _,_,a,_ in matched],float); C=np.asarray([c[k] for _,_,_,c in matched],float)
            w,wn=fit_pair(V[fit],C[fit],flip[fit]); W[b][k]=w
            score=(V[test]-C[test])@w
            random=((V[test]-C[test])@wn)*flip[test]
            null[k]+=(random>0).sum(0)+.5*(random==0).sum(0)
            tests[k].extend([[matched[j][0],matched[j][1],float(v)] for j,v in zip(test,score)])
    probe_pair={k:dict(rows=v,wins=sum(s>0 for _,_,s in v),ties=sum(s==0 for _,_,s in v),
                       null95_wins=np.percentile(null[k],[2.5,97.5])) for k,v in tests.items()}

    shadow={k:{r['shot']:r['first'] for r in rows[:90]} for k in REPS}
    for b in (1,2,3):
        cal=[r for r in rows[:90*b] if N(r) and not trainable(r['shot'])]; peaks=defaultdict(list)
        for r in cal:
            F=features(loadshot(tmp,r['shot']),r,params)
            for k in REPS: peaks[k].append(float(np.max(F[k]@W[b][k])))
        cuts={k:cut_at_budget(peaks[k]) for k in REPS}
        assert all(sum(x>=cuts[k] for x in peaks[k])*5<len(cal) for k in REPS)
        folds[b].update(cal_ids=[r['shot'] for r in cal],cuts=cuts)
        for r in rows[90*b:90*(b+1)]:
            d=loadshot(tmp,r['shot']); F=features(d,r,params)
            for k in REPS:
                idxs=np.flatnonzero(F[k]@W[b][k]>=cuts[k]); shadow[k][r['shot']]=None if not len(idxs) else float(d['t'][idxs[0]])
        print(f'Forward diagnostics: block {b+1} scored',flush=True)
    shadow_report={}
    for k,firsts in shadow.items():
        decisions=[classify(firsts[r['shot']],r['down']) for r in rows]
        shadow_report[k]=dict(firsts_in_order=list(firsts.values()),counts=score_counts(decisions),blocks=[score_counts(decisions[i:i+90]) for i in range(0,360,90)],
            flips={name:[r['shot'] for r,new in zip(rows,decisions) if test(r['kind'],new)] for name,test in {
            'gain_D':lambda a,b:a!='TP_timely' and b=='TP_timely','lose_D':lambda a,b:a=='TP_timely' and b!='TP_timely',
            'add_FP':lambda a,b:a!='FP' and b=='FP','remove_FP':lambda a,b:a=='FP' and b!='FP'}.items()})
    shifted={}
    for budget in (20,15,10):
        c=cut_at_budget([r['timely_peak_margin'] for r in rows if N(r)],budget)
        cross=lambda r:r['timely_peak_margin'] is not None and r['timely_peak_margin']>=c
        shifted[budget]=dict(cut_margin=c,fp=sum(cross(r) for r in rows if N(r)),timely=sum(cross(r) for r in rows if D(r)),
            gained=[r['shot'] for r in rows if fail(r) and cross(r)],lost=[r['shot'] for r in rows if tp(r) and not cross(r)])
    return dict(contract=idx['contract'],order_sha256=ORDER_HASH,script_sha256=digest(__file__),checks=checks,rows=rows,paired=paired,
        group_order=list(GROUPS),clip_channels=[data.RAW23[c] for c in engine.COHERENT6],
        raw_change_channels=[data.RAW23[j] for j in (9,18,3,7,8,12,15)],global_offset_posthoc=shifted,
        folds=folds,fixed_age_pairwise=probe_pair,whole_shot_shadow=shadow_report)


if __name__=='__main__':
    action,path=sys.argv[1:]; path=Path(path)
    if action=='collect': collect(path)
    elif action=='analyze': dump(OUT,analyze(path))
    else: raise ValueError(action)
