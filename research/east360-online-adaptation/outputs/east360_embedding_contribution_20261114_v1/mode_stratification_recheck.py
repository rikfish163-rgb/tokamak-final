"""Read-only CPU audit of the archived EAST360 embedding ablation."""
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np
import torch

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online import commissioning_target_separability_20261114 as target
from experiments.risk_aligned_online.hdf5_score_recheck_20261103 import score_curve
from experiments.risk_aligned_online.operating_point_audit import best_at_count, peak_for_shot, timely_auc

R = Path('/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028')
A = R/'outputs/east360_embedding_contribution_20261114_v1'
L = R/'outputs/east360_target_separability_20261114_v3_location'
C = R/'probe_outputs/east_mask_repair_static360_20261104_v1'
P = R.parent/'Fuison/experiments/risk_aligned_online/runs/commissioning_current_all_20261009/full/order.json'
REF = R/'outputs/east360_opposed_feedback_20261114_v2/per_shot.json'
ARMS = ('head_only', 'measured', 'mask_and_bias', 'full')

def read(p):
    with open(p) as f:
        return json.load(f)

def out(label, value):
    print(label, json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)

def norm(v):
    return float(np.linalg.norm(v))

def cos(a, b):
    den = norm(a)*norm(b)
    return float(a@b/den) if den else None

def describe(x):
    return [float(v) for v in np.quantile(np.asarray(x), [.25, .5, .75])]

def main():
    torch.set_num_threads(1)
    seal, protocol = read(A/'seal.json'), read(A/'protocol.json')
    order, files = seal['order'], seal['files']
    assert len(order)==len(set(order))==len(files)==360
    assert order==protocol['order'] and tuple(protocol['arms'])==ARMS
    archived = read(A/'per_shot.json')['0.2']
    old = read(REF)['0.2']['current_5']
    coverage = {r['shot']:r for r in read(C/'coverage.json')}
    prefix = {r['shot']:r for r in read(P)}
    assert len(coverage)==360 and set(order)==set(coverage) and set(order)<=set(prefix)
    split = read(R/'probe_outputs/commissioning_parent_readout_east2953_20261113_v1/postseal_bt_scale_balance.json')['raw_split']
    threshold = protocol['thresholds']['0.2']['threshold']
    data, rankings = [], {a:[] for a in ARMS}
    mismatch=[]
    for i,(sid,fn) in enumerate(zip(order,files,strict=True)):
        cov,pref=coverage[sid],prefix[sid]
        cv=cov['CIII']['available_points']; n=cov['total_points']
        assert cv in (0,n) and cov['MNM']['available_points']==0
        labels=[archived[a][i] for a in ARMS]
        assert all((q['shot'],q['disruptive'],q['down_s']) == (sid,labels[0]['disruptive'],labels[0]['down_s']) for q in labels)
        assert (old[i]['shot'],old[i]['disruptive'],old[i]['down_s']) == (sid,labels[0]['disruptive'],labels[0]['down_s'])
        y,down=labels[0]['disruptive'],labels[0]['down_s']
        x=np.load(A/'shots'/fn,allow_pickle=False)
        with np.load(C/'shots'/fn,allow_pickle=False) as f:
            times=f['times'][::8]
        with np.load(L/'shots'/fn,allow_pickle=False) as original_curves:
            n_original=len(original_curves['embedding_and_head'])
        assert x.shape==(4,len(times)) and n==n_original
        row=dict(shot=sid, rank=i+1, disrupt=y, down=down, ciii='all' if cv else 'none',
                 bt='low' if pref['initial_abs_Bt_raw']<split else 'high',
                 btval=pref['initial_abs_Bt_raw'], old=old[i], peaks={}, fixed={})
        for a,z,truth in zip(ARMS,x,labels,strict=True):
            scored=score_curve(times,z,y,down,threshold)
            if any(scored[k]!=truth[k] for k in scored): mismatch.append((sid,a,scored,truth))
            peak=peak_for_shot(times,z,y,down)
            row['peaks'][a]=peak
            row['fixed'][a]=scored
            rankings[a].append(dict(shot=sid,disruptive=y,peak=peak))
        data.append(row)
    if mismatch: raise AssertionError(('saved fixed scores disagree',mismatch[:5]))
    best={a:best_at_count(rankings[a],36) for a in ARMS}
    expected=read(A/'postseal_ranking.json')
    assert all((best[a]['tp'],best[a]['fp'])==(expected['caps']['36'][a]['tp'],expected['caps']['36'][a]['fp']) for a in ARMS)
    assert all(abs(timely_auc(rankings[a])-expected['auc'][a])<1e-12 for a in ARMS)
    out('RECHECK',dict(shots=len(data),disrupt=sum(r['disrupt'] for r in data),safe=sum(not r['disrupt'] for r in data),
                       fixed_decisions_matched=4*len(data),seal_max_full_probability_error=seal['max_full_probability_error'],bt_split=split,
                       ciii={k:sum(r['ciii']==k for r in data) for k in ('all','none')}))
    out('GLOBAL_36', {a:{**best[a], 'auc':timely_auc(rankings[a])} for a in ARMS})
    out('GLOBAL_FIXED', {a:dict(tp=sum(r['fixed'][a]['timely'] for r in data),fp=sum((not r['disrupt']) and r['fixed'][a]['alarm'] for r in data)) for a in ARMS})

    def selected(r,a):
        p=r['peaks'][a]
        return p is not None and p>=best[a]['threshold']

    def group(label, rr):
        ns=sum(not r['disrupt'] for r in rr); nd=sum(r['disrupt'] for r in rr)
        g=dict(nD=nd,nS=ns, global_cap36={})
        for a in ARMS:
            sub=[dict(disruptive=r['disrupt'],peak=r['peaks'][a]) for r in rr]
            g['global_cap36'][a] = dict(tp=sum(r['disrupt'] and selected(r,a) for r in rr),
                                       fp=sum((not r['disrupt']) and selected(r,a) for r in rr),
                                       auc=timely_auc(sub) if nd and ns else None)
        g['paired_full_vs_measured']={
          'gainedD':[r['shot'] for r in rr if r['disrupt'] and selected(r,'full') and not selected(r,'measured')],
          'lostD':[r['shot'] for r in rr if r['disrupt'] and selected(r,'measured') and not selected(r,'full')],
          'removedFP':[r['shot'] for r in rr if not r['disrupt'] and selected(r,'measured') and not selected(r,'full')],
          'addedFP':[r['shot'] for r in rr if not r['disrupt'] and selected(r,'full') and not selected(r,'measured')]}
        if nd and ns:
            cap=int(.2*ns)
            g['own_posthoc_20pct_cap']=cap
            g['own_posthoc_20pct']={a:best_at_count([dict(disruptive=r['disrupt'],peak=r['peaks'][a]) for r in rr],cap) for a in ('measured','full')}
        out('GROUP '+label,g)

    group('all',data)
    for c in ('all','none'):
        group('ciii_'+c,[r for r in data if r['ciii']==c])
    for b in ('low','high'):
        group('bt_'+b,[r for r in data if r['bt']==b])
        for c in ('all','none'):
            group('bt_'+b+'_ciii_'+c,[r for r in data if r['bt']==b and r['ciii']==c])
    for cat, pred in [('never',lambda t:not t['alarm']),('late',lambda t:t['alarm'] and not t['timely'])]:
        rr=[r for r in data if r['rank']>20 and r['disrupt'] and pred(r['old'])]
        out('HARD_'+cat,dict(n=len(rr), ids=[r['shot'] for r in rr],
            counts={a:dict(fixed=sum(r['fixed'][a]['timely'] for r in rr), cap36=sum(selected(r,a) for r in rr),
                           selected=[r['shot'] for r in rr if selected(r,a)]) for a in ARMS}))

    # The existing parent loader enforces the checkpoint integrity contract.
    parent,_=target.load_parent(target.DEFAULT_PARENT)
    new=torch.load(L/'embedding_and_head.pt',map_location='cpu',weights_only=True)['state']
    old_state=parent.state_dict()
    changed=[k for k in old_state if not torch.equal(old_state[k],new[k])]
    assert set(changed)=={'embedding.weight','embedding.bias','risk_head.weight','risk_head.bias'}
    w0=old_state['embedding.weight'].detach().numpy().astype(np.float64).reshape(64,8,46)
    w1=new['embedding.weight'].detach().numpy().astype(np.float64).reshape(64,8,46)
    b0=old_state['embedding.bias'].detach().numpy().astype(np.float64)
    b1=new['embedding.bias'].detach().numpy().astype(np.float64)
    v0,v1=w0[:,:,:23],w1[:,:,:23]
    m0,m1=w0[:,:,23:],w1[:,:,23:]
    db=b1-b0; dm=m1-m0
    modes={}
    for c in ('all','none'):
        mask=np.ones((8,23),dtype=np.float64);mask[:,2]=0;mask[:,1]=(c=='all')
        old_mask=np.einsum('ktc,tc->k',m0,mask);new_mask=np.einsum('ktc,tc->k',m1,mask)
        B0=b0+old_mask;B1=b1+new_mask;D=B1-B0;dmask=new_mask-old_mask
        modes[c]=(mask,B0,B1,D)
        out('VECTOR_'+c,dict(b0=norm(b0),b1=norm(b1),delta_b=norm(db),
            old_masks=norm(old_mask),new_masks=norm(new_mask),delta_masks=norm(dmask),
            cos_delta_bias_mask=cos(db,dmask),B0=norm(B0),B1=norm(B1),delta_B=norm(D),
            cos_B0_B1=cos(B0,B1),cos_delta_B0=cos(D,B0),cos_delta_B1=cos(D,B1)))
    out('VECTOR_COMPARE',dict(cos_delta_present_absent=cos(modes['all'][3],modes['none'][3]),
         trained_CIII_toggle=norm(modes['all'][2]-modes['none'][2]),
         source_CIII_toggle=norm(modes['all'][1]-modes['none'][1]),
         delta_CIII_toggle=norm(modes['all'][3]-modes['none'][3]),
         numeric_weight_old_frob=norm(v0),numeric_weight_new_frob=norm(v1),numeric_weight_delta_frob=norm(v1-v0),
         mask_weight_old_frob=norm(m0),mask_weight_new_frob=norm(m1),mask_weight_delta_frob=norm(dm)))
    out('EFFECTIVE_64D',dict(delta_bias=db.tolist(),modes={c:dict(source=B0.tolist(),trained=B1.tolist(),
        delta_mask=(D-db).tolist(),delta_total=D.tolist()) for c,(_,B0,B1,D) in modes.items()}))
    contributions=[]
    for j,name in enumerate(io23.RAW23):
        d=dm[:,:,j].sum(axis=1)
        contributions.append(dict(channel=name,delta=norm(d),cos_with_delta_present=cos(d,modes['all'][3])))
    out('CHANNEL_DELTAS', sorted(contributions,key=lambda x:-x['delta']))
    xx=io23.RAW23.index('xuv_ratio')
    out('XUV_CHANGE',dict(one_offset_norms=[norm(dm[:,k,xx]) for k in range(8)],
        all_eight_delta_norm=norm(dm[:,:,xx].sum(axis=1)),
        explanation='for each unavailable offset subtract that delta-mask column from delta_B; in addition the missing numeric input is zeroed'))

    # Evaluate the unmodified archived network, and equivalent input arithmetic.
    net=parent.__class__(parent.cfg,parent.center,parent.scale)
    net.load_state_dict(new,strict=True)
    net.eval()

    @torch.inference_mode()
    def folded_forward(values,mask,template):
        values=torch.from_numpy(values)
        mask=torch.from_numpy(mask)
        clean=torch.where(mask,values,net.center)
        level=torch.asinh((clean-net.center)/net.scale)
        n=len(values)
        W=net.embedding.weight.reshape(64,8,46)
        WV=W[:,:,:23].reshape(64,184)
        WM=W[:,:,23:].reshape(64,184)
        temp=torch.as_tensor(template,dtype=values.dtype).reshape(184)
        actual=mask.reshape(n,16,184).to(values.dtype)
        B=net.embedding.bias+torch.nn.functional.linear(temp,WM)
        x=(torch.nn.functional.linear(level.reshape(n,16,184),WV)+B+
           torch.nn.functional.linear(actual-temp,WM))
        valid=mask.reshape(n,16,-1).any(-1)
        x=torch.cat((x,net.cls.expand(n,-1,-1)),1)+net.position
        valid=torch.cat((valid,torch.ones(n,1,dtype=torch.bool)),1)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            x=net.encoder(x,src_key_padding_mask=~valid)
        gain,bias=net.initial_fast(n).chunk(2,-1)
        x=net.norm(x)*(1+.1*gain[:,None].tanh())+.1*bias[:,None].tanh()
        x=x.masked_fill(~valid[:,:,None],0.)
        return net.risk_head(x[:,-1]).squeeze(-1)

    # One first-in-sequence example in each CIII x Bt x outcome cell; append
    # three previously documented diagnostic shots, without fitting anything.
    chosen=[]
    for b in ('low','high'):
        for c in ('all','none'):
            for y in (True,False):
                matches=[r for r in data if (r['bt'],r['ciii'],r['disrupt'])==(b,c,y)]
                if matches: chosen.append(matches[0]['shot'])
    chosen=list(dict.fromkeys(chosen+[90874,74970,85422]))
    lookup={r['shot']:r for r in data}
    stats=[]; errors=[]; pad_residual=[]; other_mask=[]; logit_error=[]; archive_error=[]; largest=[]
    center=parent.center.detach().numpy().astype(np.float64)
    scale=parent.scale.detach().numpy().astype(np.float64)
    for sid in chosen:
        r=lookup[sid]; s=target.load_signal(sid)
        truth=io23.read_outcome('EAST',sid)
        assert truth==(r['disrupt'],r['down'])
        assert len(s['times'])==coverage[sid]['total_points']
        fn=files[r['rank']-1]
        with np.load(C/'shots'/fn,allow_pickle=False) as old_curve:
            assert np.array_equal(s['times'],old_curve['times'])
        assert int(s['available'][:,1].sum())==coverage[sid]['CIII']['available_points']
        n_prefix=prefix[sid]['prefix_samples']
        assert np.isclose(np.median(np.abs(s['values'][:n_prefix,0])),r['btval'],rtol=1e-6,atol=1e-8)
        ends=np.unique(np.r_[[0,8,64,120],(np.linspace(128,len(s['times'])-1,5).astype(int)//8)*8])
        values,mask=target.windows(s['values'],s['available'],ends)
        with torch.inference_mode():
            native=net(torch.from_numpy(values),torch.from_numpy(mask))['logit']
            folded=folded_forward(values,mask,modes[r['ciii']][0])
            logit_error.extend(torch.abs(native-folded).tolist())
            archive=np.load(A/'shots'/fn,allow_pickle=False)[3,ends//8]
            native_p=native.sigmoid().numpy()
            archive_error.extend(np.abs(native_p-archive).tolist())
            largest.extend(dict(shot=sid,end=int(e),native=float(p),archived=float(q),difference=abs(float(p-q)),
                                crosses_fixed=bool((p>=threshold)!=(q>=threshold)),
                                crosses_retrospective=bool((p>=best['full']['threshold'])!=(q>=best['full']['threshold'])))
                           for e,p,q in zip(ends,native_p,archive,strict=True))
        if sid==70404:
            indices,cpu_curves=target.curves([net],s,stride=8)
            cpu=cpu_curves[0]
            sealed=np.load(A/'shots'/fn,allow_pickle=False)[3]
            assert len(cpu)==len(sealed)
            out('CPU_WORST_SHOT',dict(shot=sid,n=len(cpu),
                max_abs_difference=float(np.max(np.abs(cpu-sealed))),
                at_128_cpu_batch512=float(cpu[16]),at_128_sealed=float(sealed[16]),
                at_128_cpu_sample_batch=float(native_p[list(ends).index(128)]),
                fixed_cpu=score_curve(s['times'][indices],cpu,r['disrupt'],r['down'],threshold),
                fixed_sealed=score_curve(s['times'][indices],sealed,r['disrupt'],r['down'],threshold),
                retrospective_cpu=score_curve(s['times'][indices],cpu,r['disrupt'],r['down'],best['full']['threshold']),
                retrospective_sealed=score_curve(s['times'][indices],sealed,r['disrupt'],r['down'],best['full']['threshold'])))
        z=np.arcsinh((np.where(mask,values,center)-center)/scale).reshape(-1,8,23)
        m=mask.reshape(-1,8,23).astype(np.float64)
        valid=m.any(axis=(1,2)); complete=(m.sum(axis=(1,2))==modes[r['ciii']][0].sum()) & np.all(m==modes[r['ciii']][0],axis=(1,2))
        val=np.einsum('ktc,ntc->nk',v1,z);dval=np.einsum('ktc,ntc->nk',v1-v0,z)
        full=np.einsum('ktc,ntc->nk',v1,z)+np.einsum('ktc,ntc->nk',m1,m)+b1
        joined=np.concatenate((z,m),axis=2).reshape(-1,368)
        direct=joined@w1.reshape(64,368).T+b1
        errors.extend(np.max(np.abs(direct-full),axis=1))
        Dactual=np.einsum('ktc,ntc->nk',dm,m)+db
        residual=Dactual-modes[r['ciii']][3]
        pad_residual.extend(np.linalg.norm(residual[valid & ~complete],axis=1))
        missed={io23.RAW23[j]:int((~s['available'][:,j]).sum()) for j in range(23) if j not in (1,2) and not s['available'][:,j].all()}
        if missed: other_mask.append(dict(shot=sid,missing=missed))
        vnorm=np.linalg.norm(val[complete],axis=1); dnorm=np.linalg.norm(dval[complete],axis=1)
        assert len(vnorm)>0
        cosv=[cos(v,modes[r['ciii']][3]) for v in val[complete]]
        cosdv=[cos(v,modes[r['ciii']][3]) for v in dval[complete]]
        combined=np.linalg.norm(dval[complete]+modes[r['ciii']][3],axis=1)
        numeric_mean=dval[complete].mean(axis=0)
        numeric_rms_variation=float(np.sqrt(np.mean(np.sum((dval[complete]-numeric_mean)**2,axis=1))))
        row=dict(shot=sid,bt=r['bt'],ciii=r['ciii'],outcome='D' if r['disrupt'] else 'S',
                 complete_patches=int(complete.sum()),new_numeric_median=float(np.median(vnorm)),
                 delta_numeric_median=float(np.median(dnorm)),delta_B=norm(modes[r['ciii']][3]),
                 delta_numeric_shot_mean_norm=norm(numeric_mean),delta_numeric_shot_rms_variation=numeric_rms_variation,
                 delta_numeric_deltaB_cos_median=float(np.median(cosdv)),combined_delta_norm_median=float(np.median(combined)),
                 numeric_deltaB_cos_median=float(np.median(cosv)))
        stats.append(row)
        out('RAW_EXAMPLE',row)
    out('RAW_CHECK',dict(shots=chosen, windows_per_shot='four prefix endpoints 0,8,64,120 plus five evenly spaced >=128; 16 patches per endpoint',
         hdf5_outcome_clock_coverage_prefix_checked=len(chosen),linear_split_max_abs_error=max(errors),
         folded_vs_native_logit_max_abs_error=max(logit_error),
         native_cpu_vs_archived_full_probability_max_abs_error=max(archive_error),
         native_cpu_vs_archive_worst=sorted(largest,key=lambda r:-r['difference'])[:5],
         native_cpu_vs_archive_pointwise_threshold_flips=sum(r['crosses_fixed'] or r['crosses_retrospective'] for r in largest),
         selected_complete_patch_medians_numeric=describe([r['new_numeric_median'] for r in stats]),
         selected_complete_patch_medians_delta_numeric=describe([r['delta_numeric_median'] for r in stats]),
         selected_shot_delta_numeric_mean_norm=describe([r['delta_numeric_shot_mean_norm'] for r in stats]),
         selected_shot_delta_numeric_rms_variation=describe([r['delta_numeric_shot_rms_variation'] for r in stats]),
         selected_shot_delta_numeric_deltaB_cos=describe([r['delta_numeric_deltaB_cos_median'] for r in stats]),
         selected_shot_combined_delta_norm=describe([r['combined_delta_norm_median'] for r in stats]),
         selected_complete_patch_medians_cos=describe([r['numeric_deltaB_cos_median'] for r in stats]),
         valid_nonfull_pattern_patches=len(pad_residual),valid_nonfull_pattern_residual_norm_range=[min(pad_residual),max(pad_residual)] if pad_residual else None,
         other_missing_in_sample=other_mask))

if __name__=='__main__':
    main()
