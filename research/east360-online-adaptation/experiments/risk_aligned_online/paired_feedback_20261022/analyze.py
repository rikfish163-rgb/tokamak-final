"""Reproduce source-only probes and inspect actual inner-gradient directions."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23
from ..event_adapt_20261022.data import attach_training_events
from ..event_adapt_20261022.train import load_source, event_pool, sample_batch
from ..event_prefix_probe import DEFAULT_LABELS, read_events
from ..fixed_test_protocol import load_test_contract, score_fixed_test
from ..masked_state_20261022.model import masked_views, reconstruction_loss
from ..masked_state_20261022.probe import distort
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_train import pools, score, sha
from ..simple_online import _durable_json
from .model import PairedFeedback, initial_encoder, risk_state_loss
from .prepare import diagnose, validation_examples


def load_model(directory):
    done=json.loads((directory/'completed.json').read_text())
    path=directory/('warm.pt' if done['mode']=='warm' else 'selected.pt')
    if done['status']!='COMPLETED' or sha(path)!=done['checkpoint_sha256']:
        raise AssertionError('incomplete or changed checkpoint')
    ck=torch.load(path,weights_only=True,map_location='cpu')
    base,source=load_parent(DEFAULT_PARENT)
    for k,v in base.state_dict().items():
        if not torch.equal(v,ck['model']['base.'+k]): raise AssertionError(f'parent changed: {k}')
    if ck['config']['threshold']!=source['threshold']: raise AssertionError('threshold changed')
    with zipfile.ZipFile(directory/'code_snapshot.zip') as archive:
        for k,digest in ck['config']['code_sha256'].items():
            if hashlib.sha256(archive.read(k)).hexdigest()!=digest: raise AssertionError('archive code changed')
    model=PairedFeedback(base,ck['model']['level_scale']);model.load_state_dict(ck['model'])
    model.cuda().prepare_decoder()
    return model,done,ck


def cluster_interval(records,condition):
    rows=[r for r in records if r['condition']==condition]
    ids=sorted({r['shot'] for r in rows}); rng=np.random.default_rng(1041)
    sums=np.array([sum(r['after_bce']-r['before_bce'] for r in rows if r['shot']==sid) for sid in ids])
    counts=np.array([sum(r['shot']==sid for r in rows) for sid in ids])
    boot=[]
    for _ in range(2000):
        pick=rng.integers(len(ids),size=len(ids));boot.append(sums[pick].sum()/counts[pick].sum())
    return dict(distinct_shots=len(ids),windows=len(rows),mean_BCE_change=float(sums.sum()/counts.sum()),
        cluster_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
        scope='descriptive resampling of developed VAL shots; not independent or multi-seed uncertainty')


def gradient_direction(model,examples):
    result=[]
    def vector(loss,fast):
        gg=torch.autograd.grad(loss,tuple(fast.values()),retain_graph=False,allow_unused=False)
        return torch.cat([g.detach().flatten() for g in gg])
    for e in examples:
        x=torch.as_tensor(e['x'],device='cuda')[None]; m=torch.as_tensor(e['mask'],device='cuda')[None]
        origin=initial_encoder(model)
        reference=model.predict_fast(x,m,origin,mode='risk_state')['risk_state'].detach()
        moved=distort(model,x,m,torch.as_tensor(e['gain'],device='cuda'),torch.as_tensor(e['bias'],device='cuda'))
        for c,observed in [('clean',x),('shift',moved)]:
            fast=initial_encoder(model);xx,mm,held=masked_views(observed,m)
            pred=model.predict_fast(xx,mm,fast,mode='reconstruct')['reconstruction']
            aux=reconstruction_loss(model,pred,xx,held);g=vector(aux,fast)
            logit=model.predict_fast(observed,m,fast)['logit']
            risk=F.binary_cross_entropy_with_logits(logit,torch.tensor([e['y']],device='cuda').float())
            r=vector(risk,fast)
            h=model.predict_fast(observed,m,fast,mode='risk_state')['risk_state']
            recovery=risk_state_loss(model,h,reference); q=vector(recovery,fast)
            def cosine(a,b):
                norm=a.norm()*b.norm()
                return float((a@b)/norm) if float(norm)>1e-12 else None
            result.append(dict(shot=e['shot'],end=e['end'],y=e['y'],condition=c,
                auxiliary_loss=float(aux.detach()),risk_loss=float(risk.detach()),
                auxiliary_gradient_norm=float(g.norm()),risk_gradient_norm=float(r.norm()),
                cosine_aux_risk=cosine(g,r),cosine_aux_known_recovery=cosine(g,q)))
    return result


def main():
    torch.set_num_threads(2)
    root=Path(__file__).parents[1]/'runs';roles=io23.source_roles();adv=io23.advanced_times()
    val=load_source(roles['val'],adv)
    model,done,ck=load_model(root/'paired_feedback_warm_20261022')
    # Recompute all 220 source curves, not only checkpoint metadata.
    test=load_source(roles['test'],adv);rows=score(model,test,'cuda')
    z=np.load(root/'paired_feedback_warm_20261022/test220_traces.npz')
    maximum=0.
    for row in rows:
        np.testing.assert_array_equal(row['times'],z[f'{row["shot"]}_times'])
        maximum=max(maximum,float(np.max(np.abs(row['scores']-z[f'{row["shot"]}_scores']))))
    if maximum>1e-6: raise AssertionError('source curve replay differs')
    metrics=score_fixed_test('J-TEXT',rows,ck['config']['threshold'])
    if (metrics['tp'],metrics['fp'])!=(100,13): raise AssertionError('source qualification changed')
    result={}; signatures=[]; source_inputs=[]
    for mode in ('F','R','P','C'):
        directory=root/f'paired_feedback_{mode}_20261022'
        model,done,ck=load_model(directory)
        saved=json.loads((directory/'reporting_probe.json').read_text())
        examples=validation_examples(model,val,128,922)
        reproduced=diagnose(model,examples)
        if len(saved['records'])!=256: raise AssertionError('incomplete source probe')
        maxdiff=0.
        for e,a,b in zip([e for e in examples for _ in range(2)],saved['records'],reproduced['records']):
            if (a['shot'],a['end'],a['y'],a['condition'])!=(b['shot'],b['end'],b['y'],b['condition']):
                raise AssertionError('source probe identity changed')
            for key in ('before_logit','after_logit','before_completion','after_completion','before_representation','after_representation'):
                maxdiff=max(maxdiff,abs(a[key]-b[key]))
            for prefix in ('before','after'):
                z0=a[prefix+'_logit'];expected=float(np.logaddexp(0.,z0)-a['y']*z0)
                if abs(expected-a[prefix+'_bce'])>1e-12: raise AssertionError('BCE record wrong')
        if maxdiff>2e-6: raise AssertionError('probe replay mismatch')
        if saved['probe_gate']!=reproduced['probe_gate'] or done['allow_east_candidate']!=saved['probe_gate']:
            raise AssertionError('incorrect promotion gate')
        signature=[(r['shot'],r['end'],r['y'],r['condition'],r['before_logit']) for r in saved['records']]
        signatures.append(signature)
        inputs=json.loads((directory/'training_inputs.json').read_text());source_inputs.append(inputs)
        history=json.loads((directory/'history.json').read_text())
        best=min(history,key=lambda h:h['criterion'])
        if best['step']!=done['selected_step']: raise AssertionError('non-VAL selection')
        directions=gradient_direction(model,examples[:32])
        _durable_json(directory/'gradient_directions.json',directions)
        direction_summary={}
        for condition in ('clean','shift'):
            rr=[r for r in directions if r['condition']==condition]
            direction_summary[condition]=dict(n=len(rr),
                risk_aligned=sum(r['cosine_aux_risk']>0 for r in rr),
                median_aux_risk_cosine=float(np.median([r['cosine_aux_risk'] for r in rr])),
                recovery_aligned=sum(r['cosine_aux_known_recovery'] is not None and r['cosine_aux_known_recovery']>0 for r in rr))
        result[mode]=dict(selected_step=done['selected_step'],checkpoint_sha256=done['checkpoint_sha256'],
            summary=saved['summary'],gate=saved['probe_gate'],replay_max_abs_diff=maxdiff,
            uncertainty={c:cluster_interval(saved['records'],c) for c in ('clean','shift')},
            gradient_direction=direction_summary,seconds=done['seconds'])
        print(json.dumps(dict(stage='reproduced',mode=mode,gate=saved['probe_gate'],maxdiff=maxdiff)),flush=True)
    if any(s!=signatures[0] for s in signatures[1:]): raise AssertionError('unmatched probe examples or base')
    if any(s!=source_inputs[0] for s in source_inputs[1:]): raise AssertionError('unmatched source training input order')
    # Actual label composition, not a supposition based on the sampler name.
    train=load_source(roles['train'],adv);grouped,_=read_events(DEFAULT_LABELS)
    train=attach_training_events(train,grouped,allowed_train_ids=roles['train'])
    groups,events=pools(train),event_pool(train);rng=np.random.default_rng(811)
    digest=hashlib.sha256();labels=[];event_count=0
    for _ in range(512):
        x,m,y,e,_,_=sample_batch(train,groups,events,rng,model.cfg,2,'cpu',channel_dropout=0.)
        labels.extend(y.tolist());event_count+=int((e>=0).sum())
        for i in range(2):
            gain=np.exp(rng.uniform(-.35,.35,23)).astype('float32');bias=rng.uniform(-1.,1.,23).astype('float32')
            for a in (x[i:i+1],m[i:i+1],y[i:i+1]): digest.update(a.numpy().tobytes())
            digest.update(gain.tobytes());digest.update(bias.tobytes())
    if digest.hexdigest()!=source_inputs[0]['sequence_sha256']: raise AssertionError('TRAIN sampling replay changed')
    if sha(DEFAULT_PARENT)!=PARENT_SHA256: raise AssertionError('parent file changed')
    output=io23.ROOT/'outputs/source_event_rebuild_audit_20261022/paired_feedback_summary.json'
    summary=dict(status='VERIFIED_REPLAY_AND_RECOUNT',parent_sha256=PARENT_SHA256,
        source_test={k:v for k,v in metrics.items() if k!='predictions'},source_test_contract=load_test_contract('J-TEXT'),
        source_replay_max_abs_diff=maximum,arms=result,
        matched_training=dict(**source_inputs[0],positive_windows=sum(v>.5 for v in labels),negative_windows=sum(v<=.5 for v in labels),
            windows_in_known_event=int(event_count),labels_used_only_in_offline_outer=True),
        probe_records=1024,east_accessed=False,independent_confirmation=False,
        caution='C sequentially added after F/R/P failed; single seed; synthetic observation shifts are not established physical invariances')
    _durable_json(output,summary);print(json.dumps(dict(output=str(output),source=summary['source_test'],training=summary['matched_training'])),flush=True)


if __name__=='__main__':main()
