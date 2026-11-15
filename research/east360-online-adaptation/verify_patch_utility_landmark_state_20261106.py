"""Independently reconstruct landmark supervision, folds, fits and metrics."""

import argparse
from pathlib import Path
import numpy as np
from verify_confirmation_raw_replay_20261105 import Hashes
from verify_patch_utility_selection_20261106 import menu_analysis, deterministic_folds
from run_patch_utility_shadow_response_20261106 import m,b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-receipt", type=Path, help="optional new output; never overwrites")
    args = parser.parse_args()
    folder=Path("probe_outputs")
    root=folder/"patch_utility_landmark_state_20261106_v1"
    r=b.r.read(root/"result.json")
    h=Hashes()
    for p,sha in r["input_sha256"].items():
        assert h(p)==sha,p
    assert h(root/"held_states.npz")==r["held_sha256"] and h(root/"fits.json")==r["fit_sha256"]
    models=b.r.read(root/"fits.json")["fits"]
    tapes,ids,flags=m.load(folder/"patch_utility_bank_20261106_v1",folder/"patch_utility_target_views_20261106_v1",
                          folder/"patch_utility_selection_20261106_v1/result.json",h)
    owner=tapes[m.VIEWS[0]].owner
    fold=deterministic_folds(ids)
    n=len(owner)
    observed=np.zeros(n,bool)
    y={k:np.zeros(n,bool) for k in ("original","125ms")}
    write,good,bad=np.zeros((3,3,n),bool)
    exact_actions=0
    for v,(name,tape) in enumerate(tapes.items()):
        d=tape.data
        for q in tape.shots:
            sl=q.blocks
            k=d["clock"][sl]
            shadow=d["curves"][sl]
            ok=np.ones(shadow.shape[:2],bool)
            for j,release in enumerate(k):
                t=release+8
                available=(t<len(q.times)-1)
                observed[sl.start+j]=available
                if q.disruptive and available:
                    leads=[round(float(1000*(q.down-a)),9) for a in q.times[t+1:release+33]]
                    for key,limit in (("original",q.horizon),("125ms",125.)):
                        y[key][sl.start+j]=any(10<x<=limit+1e-6 for x in leads)
                if not available or q.native[:t+1].max()>=b.TAU:
                    ok[j,1:]=False
                else:
                    last=shadow[j,:,7].astype(float)
                    bound=np.abs(last[1:]-last[0]).reshape(-1,2).max(1)
                    ok[j,1:]=np.repeat(bound<=b.pair.LIMIT-b.pair.GUARD_HEADROOM,2)
            ok &= tape.allowed[sl]
            write[v,sl]=ok[:,1:].any(1)
            exact_actions+=int(ok[:,1:].sum())
            if len(k):
                delayed=shadow.copy()
                delayed[:,:,:8]=shadow[:,[0],:8]
                delta=menu_analysis(q,k,delayed,ok,b.TAU)["hard"]
                if q.disruptive:
                    good[v,sl]=(delta>0).any(1)
                else:
                    bad[v,sl]=(delta<0).any(1)
        print(name,int(good[v].sum()),int(bad[v].sum()),flush=True)

    def class_weight(mask,labels,index):
        w=np.zeros(mask.shape)
        for cls in (False,True):
            out={}
            for v,j in zip(*np.where(mask & (labels==cls)),strict=True):
                out.setdefault(index[j],{}).setdefault(v,[]).append(j)
            for views in out.values():
                for v,rows in views.items():
                    w[v,rows]=.5/len(out)/len(views)/len(rows)
        return w

    def manual_metrics(score,mask,labels,index):
        w=class_weight(mask,labels,index)[mask]
        yy=np.broadcast_to(labels,mask.shape)[mask]
        f=score[mask]
        ii=np.argsort(f,kind="stable")[::-1]
        w,yy,f=w[ii],yy[ii],f[ii]
        end=np.r_[np.flatnonzero(f[1:]!=f[:-1]),len(f)-1]
        tp,fp=(np.cumsum(w*(yy==q))[end] for q in (True,False))
        dp=np.diff(tp,prepend=0)
        df=np.diff(fp,prepend=0)
        auc=np.sum(dp*(fp[-1]-fp+.5*df))/(tp[-1]*fp[-1])
        ap=np.sum(dp*tp/(tp+fp))/tp[-1]
        return auc,ap

    base = {}
    for v,tape in tapes.items():
        d=tape.data
        risks=np.column_stack([d["risk"][:,0],d["risk"][:,1:9].astype(float)-b.TAU,d["risk"][:,9],
                               d["curves"][:,0,:8]-b.TAU])
        base[v]=dict(risk=risks,real=d["ordered"][:,0],reverse=d["ordered"][:,1],
                     sorted=np.concatenate([np.sort(d["ordered"][:,0,i:i+3],axis=1) for i in range(0,27,3)],axis=1))
    worst_pred,worst_stats,worst_grad=0.,0.,0.
    fold_sens={}
    with np.load(root/"held_states.npz",allow_pickle=False) as q:
        for key,a in [("ids",ids),("owner",owner),("folds",fold),("flags",flags),
                      ("writable",write),("rescue",good),("safe_harm",bad),("known",np.broadcast_to(observed,write.shape)),
                      ("y_original",y["original"]),("y_125ms",y["125ms"])]:
            assert np.array_equal(q[key],a),key
        case=good|bad
        for key,heads in models.items():
            regime,target,arm=key.split("__")
            mat=np.stack([np.column_stack((d["risk"],d[arm] if arm!="risk" else np.zeros((n,27)))) for d in base.values()])
            pred=np.zeros((3,n))
            for k,fit in enumerate(heads):
                held=fold[owner]==k
                tr=~held&(fold[owner]!=(k+1)%5)
                coef=np.array(fit["coef"])
                scale=np.array(fit["scale"])
                center=np.array(fit["center"])
                pred[:,held]=np.einsum("vbf,f->vb",(mat[:,held]-center)/scale,coef)+fit["intercept"]
                support=(write if regime=="writable" else q["known"])[:,tr]
                label=y[target][tr]
                weight=class_weight(support,label,owner[tr])[support]
                used=mat[:,tr][support]
                z=(used-center)/scale
                prob=1/(1+np.exp(-(z@coef+fit["intercept"])))
                err=weight*(prob-np.broadcast_to(label,support.shape)[support])
                grad=max(abs(err.sum()),np.max(np.abs(z.T@err+.03*coef)))
                worst_grad=max(worst_grad,float(grad))
                assert np.max(np.abs(np.average(z,axis=0,weights=weight)))<1e-9
            delta=float(np.max(np.abs(q[key]-pred)))
            worst_pred=max(worst_pred,delta)
            report=r["report"][key]
            subsets=[(write,y[label],report["writable"][label]) for label in y]
            subsets.append((case,flags[owner],report["case"]))
            subsets += [(case & (ids[owner]!=int(i))[None],flags[owner],dict(auc=v))
                        for i,v in report["case_loo_auc"].items()]
            for mask,labels,ref in subsets:
                stat=manual_metrics(pred,mask,labels,owner)
                for j,k in enumerate(("auc","ap")):
                    if k in ref:
                        worst_stats=max(worst_stats,abs(stat[j]-ref[k]))
            if key.startswith("all__125ms"):
                fold_sens[key]={}
                for k in range(5):
                    mask=case & (fold[owner]==k)[None]
                    if (mask & good).any() and (mask & bad).any():
                        fold_sens[key][str(k)]=manual_metrics(pred,mask,flags[owner],owner)[0]
    assert worst_pred<1e-10 and worst_stats<1e-10 and worst_grad<1e-6,(worst_pred,worst_stats,worst_grad)
    h.check()
    receipt = dict(status="INDEPENDENT_SOURCE_LANDMARK_CASE_AND_FIT_CHECK",
        source_result_sha256=h(root/"result.json"),checked_inputs=len(r["input_sha256"]),exact_actions=exact_actions,
        checked_held_predictions=len(models)*3*n,models_checked=sum(map(len,models.values())),
        max_prediction_difference=worst_pred,max_metric_difference=worst_stats,max_ridge_gradient=worst_grad,
        within_fold_case_auc_exploratory=fold_sens,new_val_test_east_reads=0)
    if args.write_receipt is not None:
        b.r.write_new(args.write_receipt, receipt)
    print("verified",exact_actions,len(models)*3*n,worst_pred,worst_stats,worst_grad,flush=True)


if __name__ == "__main__":
    main()
