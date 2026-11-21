# Arrival-reference cold20 audit — 2026-10-30

**Conclusion:** the saved-array arithmetic and fitted references check out, but this is a negative-stage diagnostic, not an independently validated alarm improvement. The context reference is slightly worse than the mean reference on the reported average MSE; the saved direction screen fails. No full-stream experiment, threshold change, GPU job, or physics-discovery claim was made in this audit.

## Evidence and limitations

- CPU `rl_env` generic verifier: **568/568** per-shot rows exact (maximum absolute metric difference **0**); 2,707 complete episodes, five disjoint held folds, identical control exact, future timestamps after arrival. It checks saved arrays, not a new neural forward replay.
- Independent NumPy reconstruction: all **200/200 supported channel fits** (5 folds × 2 reference arms × 20 channels) agree with saved ridge coefficients to **1.243e-14**, weighted means **0**, recomputed pair scalers **2.221e-16**, held predictions **0**, eight feedback potentials **7.106e-15**, and 20 per-shot quality records (40 values) **0** maximum absolute discrepancy. The three cached artifact hashes, prior completion hash, ten fixed-pair hashes, and four relevant live/snapshot code hashes match; all 20 cached outcome flags/times match the current EAST HDF5 metadata.
- The **real reference** uses only the 16 fold-fit shots' preterminal-valid, origin-label-0 windows (counts by fold: 2,288, 2,118, 2,226, 2,002, 2,002), with the **same corresponding frozen real pair's** prefix center/scale and arrival scale. The classifier pair/scalers themselves were fit on both classes from that fold, not on label 0 alone. Ridge weighting is equal per contributing shot, then equal per eligible window within shot, with L2=0.1 and unpenalized intercept.
- Label 0 is the operational 125-ms **origin** label, not proof of healthy plasma: 68 fit-row instances across overlapping folds become positive by arrival while remaining preterminal-valid. The **misplaced control** rolls labels within each shot: its nominal-zero fit includes 272 origin-positive row instances across folds by construction; it is not a true-negative-only reference.
- Common support is 20/23 channels in every fold, and the saved per-window mask equals observed-valid ∩ real-supported ∩ misplaced-supported. It removes **308** otherwise observed cells, all channel 1 in fold 3; that fold's original real-pair arrival coefficient for that channel is exactly zero. Thus new `zero` versus old `real` has maximum potential difference **3.553e-15**, gradient difference **4.441e-16**, state-norm difference **0**, and future-logit difference **0**; all 71 matched per-shot metric rows are exact. This equality is empirical here, not a general identity when a removed coefficient is nonzero.
- Before RMS projection, the real context predictor is negative in **3,126/54,540 = 5.73%** of common-observed held window/channel cells, affecting **1,695/2,727 = 62.16%** of held windows and all 20 shots. For origin-label-0, valid windows it is **3,006/53,180 = 5.65%**. These denominators include the 20 incomplete-future windows excluded from neural metrics; this is a per-cell count, not the fraction of all four summary features.
- For each arm, independently reconstructed `e = Σ_mask q1_arrival_coef × (arrival_summary/arrival_scale − reference)` agrees with the saved potentials; the paired arm uses its separate q1−q0 formula. The verifier confirms `g = sigmoid(origin_logit) − sigmoid(origin_logit + e)` and state norm `0.05|g|` (zero when no channels). The temporary Q/K direction is the normalized origin-logit gradient in the code; its neural replay was not independently rerun.
- Reference quality, equal weight per shot over **20 shots / 2,659** eligible label-0 windows: context **1.819001** versus mean **1.815532** MSE, context wins **12/20**; window-weighted values are **2.469875** versus **2.468305**. Shot 70394 contributes **29.443842** versus **29.399814**; a sensitivity calculation excluding it still favors mean (**0.365062** versus **0.363728**), but the reported primary calculation retains it. MSE is in pair-normalized arrival-summary units, not physical units.

## All eight arms

Cells show equal-shot mean ΔBCE = after − frozen before, followed by improved shots/eligible shots. Negative is better; group window counts can overlap. Safe: 3 shots/527 windows; early-normal: 17/2,112; timely danger: 17/80; origin-intersection danger: 17/46; late danger: 17/22.

| arm | safe | early-normal | timely danger | origin-intersection | late danger |
|---|---:|---:|---:|---:|---:|
| real | -0.000863 (1/3) | +0.001595 (7/17) | -0.035719 (12/17) | -0.040816 (12/17) | -0.059756 (14/17) |
| mean | -0.003050 (2/3) | +0.001410 (6/17) | -0.034252 (14/17) | -0.040886 (13/17) | -0.064071 (16/17) |
| zero | +0.011153 (0/3) | +0.006273 (0/17) | -0.053997 (15/17) | -0.059433 (15/17) | -0.089812 (16/17) |
| reverse | -0.001095 (1/3) | +0.001529 (6/17) | -0.004248 (12/17) | -0.008004 (12/17) | -0.006072 (13/17) |
| misplaced | +0.002586 (2/3) | +0.000149 (5/17) | +0.014473 (2/17) | +0.014611 (2/17) | +0.013434 (4/17) |
| reference_only | -0.010153 (3/3) | -0.004232 (17/17) | +0.011545 (0/17) | +0.010590 (0/17) | +0.011273 (0/17) |
| paired | -0.004670 (2/3) | -0.002673 (13/17) | -0.018418 (10/17) | -0.024867 (9/17) | -0.039187 (13/17) |
| identical | 0 (0/3) | 0 (0/17) | 0 (0/17) | 0 (0/17) | 0 (0/17) |

The pre-existing screen fails because real improves only 1/3 safe shots, is slightly worse than mean on origin-intersection (difference **+0.00007073**), and worsens early-normal mean. Zero improves danger more but worsens all safe and early-normal shots. These are reset, overlapping episodes on the same exploratory cold20, with only three safe shots; they do not establish sustained prequential TPR/FPR, calibrated likelihood ratios, independent generalization, or learned physical laws.

## Reproduction

Run from this worktree; the verifier writes only `probe_outputs/arrival_reference_cold20_v1/independent_audit.json`. The second command extracts the read-only NumPy audit below from this report; both suppress GPU visibility and bytecode writes.

```sh
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/hetaisheng/miniconda3/envs/rl_env/bin/python -B -m experiments.risk_aligned_online.arrival_increment_20261030.verify probe_outputs/arrival_reference_cold20_v1
awk '/^```python/{f=1;next} /^```/{if(f)exit} f' ARRIVAL_REFERENCE_AUDIT_20261030.md | env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/hetaisheng/miniconda3/envs/rl_env/bin/python -B -
```

```python
import hashlib, json, pathlib, zipfile
import numpy as np, torch
from experiments.risk_aligned_online import io23
p=pathlib.Path('probe_outputs/arrival_reference_cold20_v1'); read=lambda x:json.loads(x.read_text()); sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
proto=read(p/'protocol.json'); cache=pathlib.Path(proto['cache']); old=pathlib.Path(proto['prior']); outcomes=read(cache/'outcomes.json')
assert all(sha(cache/k)==v for k,v in proto['cache_hashes'].items()) and sha(old/'completed.json')==proto['prior_completed_sha256']
assert all(sha(pathlib.Path(k))==v for k,v in proto['fixed_pairs'].items())
with zipfile.ZipFile(p/'code_snapshot.zip') as z:
 for folder in ('arrival_reference_20261030','arrival_increment_20261030'):
  for name in ('learning.py','run.py'):
   k=f'experiments/risk_aligned_online/{folder}/{name}'; assert sha(pathlib.Path(k))==proto['code_sha256'][k]==hashlib.sha256(z.read(k)).hexdigest()
a=dict(np.load(cache/'observations.npz',allow_pickle=False)); n=len(a['ids']); y=np.zeros(n,int); vo=np.ones(n,bool); va=vo.copy(); arrived_pos=np.zeros(n,bool)
for sid,o in outcomes.items():
 assert io23.read_outcome('EAST',int(sid))==(o['disruptive'],o['down']); m=a['ids']==int(sid)
 if o['disruptive']:
  t=(o['down']-a['origin_time'][m])*1000; t2=(o['down']-a['arrival_time'][m])*1000; vo[m]=t>=-1e-6; va[m]=t2>=-1e-6; y[m]=(t<=125+1e-6)&vo[m]; arrived_pos[m]=(t2<=125+1e-6)&va[m]
def summary(target,valid):
 t=np.where(valid[:,None,:],target,0.); m=t.mean(1); return np.stack((m,t[:,-1]-t[:,0],np.sqrt(((t-m[:,None])**2).mean(1)),t[:,-1]),axis=2)
actual=summary(a['target'],a['valid']); reverse=summary(a['target'][:,::-1].copy(),a['valid']); quality={r['shot']:r for r in read(p/'reference_quality.json')}
err={k:0. for k in ('scaler','coefficient','mean','prediction','feedback','quality','old_potential','old_gradient','old_norm','old_logit','old_row')}; counts={'fits':0,'negative_cells':0,'cells':0,'negative_windows':0,'windows':0,'normal_negative':0,'normal_cells':0,'removed':0,'crosses':0,'shifted_true_positive':0}; scores=[]
def diff(k,u,v): err[k]=max(err[k],float(np.max(np.abs(np.asarray(u)-np.asarray(v)))))
for spec in proto['folds']:
 f=spec['fold']; folder=p/f'fold{f}'; idx=np.flatnonzero(np.isin(a['ids'],spec['fit'])&vo&va); held=np.flatnonzero(np.isin(a['ids'],spec['held'])); ids=a['ids'][idx]; lab=y[idx]; shifted=lab.copy()
 for sid in np.unique(ids):
  m=ids==sid; shifted[m]=np.roll(shifted[m],int(m.sum())//2)
 counts['crosses']+=int(((lab==0)&arrived_pos[idx]).sum()); counts['shifted_true_positive']+=int(((shifted==0)&(lab==1)).sum()); pairs={}; refs={}; predictions={}
 for name,labels in (('real',lab),('misplaced',shifted)):
  pair=torch.load(old/f'fold{f}'/f'{name}_pair.pt',map_location='cpu',weights_only=True); ref=torch.load(folder/f'{name}_reference.pt',map_location='cpu',weights_only=True); pairs[name]=pair; refs[name]=ref
  assert set(ref['fit_ids'])==set(spec['fit'])==set(pair['fit_ids']); assert np.array_equal(ref['fit_row_indices'],np.flatnonzero(labels==0)); assert set(ref['reference_fit_ids'])==set(ids[labels==0])
  w=np.zeros(len(idx))
  for c in (0,1):
   shots=np.unique(ids[labels==c])
   for sid in shots:
    m=(ids==sid)&(labels==c); w[m]=.5/len(shots)/m.sum()
  x=a['x'][idx].astype(float); center=(w[:,None]*x).sum(0); scale=np.maximum(np.sqrt((w[:,None]*(x-center)**2).sum(0)),1e-3); denom=(w[:,None,None]*a['valid'][idx,:,None]).sum(0); num=(w[:,None,None]*actual[idx]**2).sum(0); asc=np.maximum(np.sqrt(np.divide(num,denom,out=np.zeros_like(num),where=denom>0)),1e-3)
  diff('scaler',center,pair['prefix_center']); diff('scaler',scale,pair['prefix_scale']); diff('scaler',asc,np.asarray(pair['arrival_scale']).reshape(23,4)); center=np.asarray(pair['prefix_center']); scale=np.asarray(pair['prefix_scale']); asc=np.asarray(pair['arrival_scale']).reshape(23,4)
  design=np.column_stack(((x-center)/scale,np.ones(len(idx)))); target=actual[idx]/asc; grams={}; raw=np.zeros((len(held),23,4)); mean=raw.copy(); zh=np.column_stack(((a['x'][held].astype(float)-center)/scale,np.ones(len(held))))
  for c in range(23):
   ch=ref['channels'][c]; sel=np.flatnonzero((labels==0)&a['valid'][idx,c]); shots,inv,ct=np.unique(ids[sel],return_inverse=True,return_counts=True); assert np.array_equal(ch['fit_row_indices'],sel) and set(ch['fit_ids'])==set(shots); assert ch['nrows']==len(sel) and ch['nshots']==len(shots) and bool(ref['supported'][c])==(len(shots)>=2)
   if len(shots)<2: continue
   weights=1/(len(shots)*ct[inv]); z=design[sel]; key=sel.tobytes()
   if key not in grams:
    reg=np.eye(65)*.1; reg[-1,-1]=0.; grams[key]=z.T@(weights[:,None]*z)+reg
   b=np.linalg.solve(grams[key],z.T@(weights[:,None]*target[sel,c])); diff('coefficient',b,ch['coefficients']); diff('mean',(weights[:,None]*target[sel,c]).sum(0),ch['mean']); raw[:,c]=zh@np.asarray(ch['coefficients']); mean[:,c]=ch['mean']; counts['fits']+=1
  predictions[name]=(raw,mean)
 sealed=dict(np.load(folder/'sealed.npz',allow_pickle=False)); neural=dict(np.load(folder/'neural.npz',allow_pickle=False)); prior_s=dict(np.load(old/f'fold{f}'/'sealed.npz',allow_pickle=False)); prior_n=dict(np.load(old/f'fold{f}'/'neural.npz',allow_pickle=False)); common=np.asarray(refs['real']['supported'])&np.asarray(refs['misplaced']['supported']); mask=a['valid'][held]&common
 assert common.sum()==20 and np.array_equal(mask,sealed['common_mask']) and np.array_equal(common,sealed['common_supported']) and np.array_equal(mask.sum(1),sealed['observed_channels']); assert np.array_equal(sealed['ids'],a['ids'][held]) and np.array_equal(sealed['ends'],a['ends'][held]) and np.array_equal(sealed['ids'],prior_s['ids']) and np.array_equal(sealed['ends'],prior_s['ends']); assert np.array_equal(neural['ids'],prior_n['ids']) and np.array_equal(neural['ends'],prior_n['ends']) and np.array_equal(neural['baseline_logits'],prior_n['baseline_logits']) and np.array_equal(neural['origin_logit'],prior_n['origin_logit'])
 counts['removed']+=int((a['valid'][held]&~mask).sum()); neg=(predictions['real'][0][:,:,2]<0)&mask; normal=(y[held]==0)&vo[held]&va[held]; counts['negative_cells']+=int(neg.sum()); counts['cells']+=int(mask.sum()); counts['negative_windows']+=int(neg.any(1).sum()); counts['windows']+=len(held); counts['normal_negative']+=int(neg[normal].sum()); counts['normal_cells']+=int(mask[normal].sum())
 for key,name,j in (('context_reference','real',0),('mean_reference','real',1),('wrong_reference','misplaced',0)):
  pred=np.where(mask[:,:,None],predictions[name][j],0.); pred[:,:,2]=np.maximum(pred[:,:,2],0.); diff('prediction',pred,sealed[key])
 def evidence(pair,obs,ref):
  return (((obs/np.asarray(pair['arrival_scale']).reshape(23,4)-ref)*mask[:,:,None])*np.asarray(pair['q1_coef'])[87:-1].reshape(23,4)).sum((1,2))
 pair=pairs['real']; obs=actual[held]; ctx=sealed['context_reference']; mean=sealed['mean_reference']; zero=np.zeros_like(ctx); q0=np.column_stack(((a['x'][held].astype(float)-np.asarray(pair['prefix_center']))/np.asarray(pair['prefix_scale']),mask)); q1=np.column_stack((q0,(obs*mask[:,:,None]/np.asarray(pair['arrival_scale']).reshape(23,4)).reshape(len(held),-1))); b0=np.asarray(pair['q0_coef']); b1=np.asarray(pair['q1_coef']); paired=np.where(mask.any(1),q1@b1[:-1]+b1[-1]-q0@b0[:-1]-b0[-1],0.)
 values={'real':evidence(pair,obs,ctx),'mean':evidence(pair,obs,mean),'zero':evidence(pair,obs,zero),'reverse':evidence(pair,reverse[held],ctx),'misplaced':evidence(pairs['misplaced'],obs,sealed['wrong_reference']),'reference_only':evidence(pair,np.zeros_like(obs),ctx),'paired':paired,'identical':np.zeros(len(held))}
 for arm,value in values.items():
  assert not sealed[arm+'_potentials'][:,0].any(); diff('feedback',value,sealed[arm+'_potentials'][:,1])
 for key,new,previous in (('old_potential','zero_potentials','real_potentials'),('old_gradient','zero_evidence_gradient','real_evidence_gradient'),('old_norm','zero_state_norm','real_state_norm'),('old_logit','zero_logits','real_logits')): diff(key,(sealed if key=='old_potential' else neural)[new],(prior_s if key=='old_potential' else prior_n)[previous])
 for sid in spec['held']:
  mm=mask&((a['ids'][held]==sid)&normal)[:,None]; ct=mm.sum(1); eligible=ct>0; assert int(eligible.sum())==quality[sid]['windows']; row=[]
  for name in ('context','mean'):
   error=np.where(mm[:,:,None],(obs/np.asarray(pair['arrival_scale']).reshape(23,4)-sealed[name+'_reference'])**2,0).sum((1,2)); mse=float(np.mean(error[eligible]/(4*ct[eligible]))); diff('quality',[mse],[quality[sid][name+'_mse']]); row.append(mse)
  scores.append(row)
prior_rows={(r['fold'],r['shot'],r['group']):r for r in read(old/'per_shot.json') if r['arm']=='real'}
assert len(prior_rows)==sum(r['arm']=='zero' for r in read(p/'per_shot.json'))==71
for row in read(p/'per_shot.json'):
 if row['arm']=='zero':
  other=prior_rows[(row['fold'],row['shot'],row['group'])]
  for name in ('windows','timepoints','before_bce','after_bce','delta_bce','mean_logit_change'): diff('old_row',[row[name]],[other[name]])
assert counts=={'fits':200,'negative_cells':3126,'cells':54540,'negative_windows':1695,'windows':2727,'normal_negative':3006,'normal_cells':53180,'removed':308,'crosses':68,'shifted_true_positive':272} and max(err.values())<1e-12
assert np.allclose(np.mean(scores,axis=0),[1.8190011268650808,1.8155324184010282],rtol=0,atol=1e-12) and sum(a<b for a,b in scores)==12
print('counts',json.dumps(counts)); print('max_abs_discrepancies',json.dumps(err)); print('quality_mean_and_context_wins',np.mean(scores,axis=0),sum(a<b for a,b in scores))
```
