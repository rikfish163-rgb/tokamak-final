"""Complete360, all startup cost included; no external EAST preparation data."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..delayed_recurrent_memory_20261025.learning import batch,original_predictions
from ..fixed_test_protocol import load_test_contract,score_fixed_test
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics,sha
from ..warmstart_alignment_20261022.alignment import fit_prefix_map,transform
from .model import LifecycleAdapter
from .protocol import order_stream,AccessLedger
from .learning import target_example,source_replay,update

ARMS=('source','aligned','cold','intra','inter','both','both_shadow','intra_reverse')
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/完整投运流_冷启动炮内炮间_事前约定_20261025.md'
LABEL_PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/破裂早期负样本_完整360单因素约定_20261026.md'
READOUT_PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/炮内共同变化分量_完整360单因素约定_20261026.md'
FEEDBACK_PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/分段到达反馈_完整360单因素约定_20261026.md'
STATISTICS=io23.ROOT/'experiments/risk_aligned_online/runs/warmstart_alignment_test360_v2_20261022/source_statistics.json'


@torch.no_grad()
def predict(models,signal):
    shared=models['cold'];delta,info=shared.memory_delta(signal)
    reversed_delta,_=shared.memory_delta(signal,'reverse')
    specs={'aligned':(shared,False,True),'cold':(shared,False,False),
           'intra':(shared,True,False),'inter':(models['inter'],False,False),
           'both':(models['both'],True,False),'both_shadow':(models['both'],False,False),
           'intra_reverse':(shared,'reverse',False)}
    curves={k:[] for k in specs}
    for lo in range(0,len(signal['values']),32):
        ends=np.arange(lo,min(lo+32,len(signal['values'])))
        x,m=batch(signal,ends,shared.base.center.device);h,v=shared.tokens(x,m)
        for name,(model,fast,bypass) in specs.items():
            d=delta[ends] if fast is True else reversed_delta[ends] if fast=='reverse' else torch.zeros_like(delta[ends])
            if bypass:
                # Independent actual source call, not temporarily swapping weights.
                z=shared.base(x,m)['logit']
            else:z=model.risk(h,v,d)
            if not torch.isfinite(z).all():raise FloatingPointError('nonfinite risk')
            curves[name].append(z.sigmoid().cpu().numpy())
    return {k:np.concatenate(v) for k,v in curves.items()},delta,info


def compact(rows,threshold):
    m=metrics(rows,threshold,lead=30.)
    m['early_over3000']=sum(r['timely'] and r['lead_ms']>3000 for r in m['predictions'])
    m['timely_accuracy']=(m['tp']+m['n_safe']-m['fp'])/len(rows)
    del m['predictions']
    return m


def build_adapter(base, *, readout, feedback, context=False):
    model_type = LifecycleAdapter
    if context:
        from ..context_memory_20261028.model import ContextAdapter
        model_type = ContextAdapter
    return model_type(base, readout=readout, feedback=feedback).to(base.center.device).eval()


def choose_order(ids, policy):
    if policy == 'legacy_proxy':
        return order_stream(ids)
    if policy == 'current_only':
        from ..current_order_20261031.order import order_stream as current_order
        return current_order(ids)
    if policy == 'current_stratified':
        from ..current_order_20261031.stratified import order_stream as stratified_order
        return stratified_order(ids)
    raise ValueError('explicit legacy_proxy, current_only or current_stratified order required')


def main(root,target_loss='all_points',readout='shape',source_memory=None,feedback='delayed',direction_memory=None,event_memory=None,context_memory=None,order_policy='legacy_proxy',endpoint_policy='full_window'):
    if root.exists():raise FileExistsError(root)
    if endpoint_policy not in ('full_window','prefix16'):
        raise ValueError('explicit full_window or prefix16 endpoint policy required')
    if endpoint_policy=='prefix16':
        # Import before the code snapshot: this numerical path is not optional
        # provenance merely because target_example imports it lazily.
        from ..short_history_20261031 import sampling
    if sum(v is not None for v in (source_memory,direction_memory,event_memory,context_memory))>1:
        raise ValueError('choose one explicit source memory contract')
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.mha.set_fastpath_enabled(False)
    start=time.monotonic();contract=load_test_contract('EAST')
    order=choose_order(contract['shot_ids'],order_policy);ids=[r['shot'] for r in order]
    ledger=AccessLedger(ids,20)
    base,parent=load_parent(DEFAULT_PARENT);base.cuda()
    threshold=float(parent['threshold'])
    models={k:build_adapter(base,readout=readout,feedback=feedback,context=context_memory is not None)
            for k in ('cold','inter','both')}
    memory_receipt=None
    if source_memory is not None:
        from ..source_task_memory_20261026.checkpoint import load_source_memory
        memory_receipt=load_source_memory(models['cold'],source_memory)
    if direction_memory is not None:
        from ..direction_teaching_20261026.checkpoint import load_direction_memory
        memory_receipt=load_direction_memory(models['cold'],direction_memory)
    if event_memory is not None:
        from ..event_teaching_20261026.checkpoint import load_event_memory
        memory_receipt=load_event_memory(models['cold'],event_memory)
    if context_memory is not None:
        from ..context_memory_20261028.checkpoint import load_context_memory
        # A scaled checkpoint installs its TRAIN-only metric buffer and memory
        # class in EVERY arm before strict state copying. Never drop that buffer.
        for name in ('cold','inter','both'):
            receipt=load_context_memory(models[name],context_memory)
            if name=='cold':memory_receipt=receipt
            elif receipt!=memory_receipt:raise ValueError('context source changed between arm loads')
    # All arms share exact immutable source tensors and the same random key,
    # not independent initializations or an old EAST-fitted memory checkpoint.
    for k in ('inter','both'):models[k].load_state_dict(models['cold'].state_dict())
    source_stats=json.loads(STATISTICS.read_text())
    if source_stats['device']!='J-TEXT' or set(source_stats['shot_ids'])!=set(simple_data.source_roles()['train']):
        raise ValueError('source statistic cache is not canonical TRAIN1353')
    root.mkdir(parents=True);(root/'shots').mkdir();(root/'models').mkdir()
    _durable_json(root/'order.json',order)
    files=set(Path(__file__).parent.glob('*.py'))
    if order_policy == 'current_only':
        files.add(io23.ROOT/'CURRENT_ORDER_PLAN_20261031.md')
    if order_policy == 'current_stratified':
        files.add(io23.ROOT/'STRATIFIED_COLD20_PLAN_20261031.md')
    if endpoint_policy=='prefix16':
        files.add(Path(sampling.__file__))
        files.add(io23.ROOT/'SHORT_HISTORY_PLAN_20261031.md')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    protocol=dict(scope='DEVELOPMENT_ONLY_COMPLETE_COMMISSIONING360',source_sha256=PARENT_SHA256,
        source_checkpoint=str(DEFAULT_PARENT),contract=contract,order_sha256=sha(root/'order.json'),warmup=20,
        order_policy=order_policy,
        endpoint_policy=endpoint_policy,
        endpoint_plan_sha256=(sha(io23.ROOT/'SHORT_HISTORY_PLAN_20261031.md')
                             if endpoint_policy=='prefix16' else None),
        ordering_plan_sha256=(sha(io23.ROOT/'CURRENT_ORDER_PLAN_20261031.md')
                              if order_policy=='current_only' else
                              sha(io23.ROOT/'STRATIFIED_COLD20_PLAN_20261031.md')
                              if order_policy=='current_stratified' else None),
        warmup_in_primary_score=True,external_east_labels=0,threshold=threshold,
        arms=ARMS,source_statistics_sha256=sha(STATISTICS),plan_sha256=sha(PLAN),
        target_loss=target_loss,label_plan_sha256=sha(LABEL_PLAN) if target_loss!='all_points' else None,
        readout=readout,readout_plan_sha256=sha(READOUT_PLAN) if readout!='shape' else None,
        source_memory=memory_receipt,seed=131,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        all_outcomes_naturally_revealed_after_seal=True,learned_alarm_updates=0,
        sorting_is_offline_prefix_proxy=True,measured_damage=False,independent_confirmation=False,
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files})
    if feedback!='delayed':
        protocol.update(feedback=feedback,feedback_plan_sha256=sha(FEEDBACK_PLAN),
                        source_memory_training_feedback=('progressive' if direction_memory is not None or event_memory is not None or context_memory is not None
                                                         else 'delayed' if memory_receipt else None),
                        source_deployment_feedback_match=(True if direction_memory is not None or event_memory is not None or context_memory is not None
                                                          else False if memory_receipt else None))
    _durable_json(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
    source,events=source_replay(models['cold'])
    _durable_json(root/'source_replay.json',events)
    pending=[];map_records=[];mapping=None;replays={k:deque(maxlen=8) for k in ('inter','both')}
    rows={k:[] for k in ARMS};receipts=[];source_input_hash={}
    for i,sid in enumerate(ids):
        if time.monotonic()-start>5400:raise TimeoutError('declared90-minute runtime exceeded')
        raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        signal={k:raw[k] for k in ('values','available')}
        original=original_predictions(base,signal)
        if mapping is None:
            scores={k:original.copy() for k in ARMS};fast_info={'updates':0};delta=None
        else:
            signal=transform(signal,mapping)
            scores,delta,fast_info=predict(models,signal);scores['source']=original
        path=root/'shots'/f'{i+1:04d}_{sid}.npz'
        with path.open('xb') as f:
            np.savez_compressed(f,times=raw['times'],**scores);f.flush();os.fsync(f.fileno())
        digest=sha(path);ledger.seal(sid,digest)
        _durable_json(root/'shots'/f'{i+1:04d}_{sid}_seal.json',dict(shot=sid,index=i+1,sha256=digest,
            model_version_before_shot=len(ledger.history),outcome_available=False))
        # The one and only online outcome read follows all arm seals.
        ledger.reveal(sid)
        outcome=simple_data.attach_outcome(raw,'EAST',{})
        for name in ARMS:
            v=outcome['valid'];rows[name].append(dict(shot=sid,times=raw['times'][v],scores=scores[name][v],
                is_disrupt=outcome['is_disrupt'],down=outcome['down']))
        receipt=dict(index=i+1,shot=sid,prediction_sha256=digest,fast=fast_info,updates={})
        source_input_hash[str(sid)]=sha(io23.path_for('EAST',sid))
        if i<20:
            pending.append((outcome,scores['source']))
            map_records.append({k:raw[k] for k in ('shot','device','values','available')})
        if i==19:
            mapping=fit_prefix_map(source_stats,map_records,order=ids,count=20,policy='density_gain')
            _durable_json(root/'input_map.json',mapping)
            examples=[]
            for s,p in pending:
                sig=transform({k:s[k] for k in ('values','available')},mapping)
                d,_=models['cold'].memory_delta(sig)
                examples.append(target_example(models['cold'],s,p,sig,d,endpoint_policy=endpoint_policy))
            if endpoint_policy=='prefix16':
                receipt['cold_training_endpoints']=[dict(shot=e['shot'],flag=e['flag'],
                    ends=e['ends'],labels=e['y'].cpu().tolist()) for e in examples]
            receipt['cold_fit']=update(models['cold'],examples,[],source,fast='mixed',steps=20,guarded=False,policy=target_loss)
            state=models['cold'].adapted_state()
            for k in ('inter','both'):models[k].restore(state);replays[k].extend(examples)
            torch.save({k:v.cpu() for k,v in state.items()},root/'models'/'cold.pt')
            pending=[];map_records=[]
        elif i>=20:
            for name in ('inter','both'):
                e=target_example(models[name],outcome,scores[name],signal,delta,endpoint_policy=endpoint_policy)
                if endpoint_policy=='prefix16':
                    receipt.setdefault('training_endpoints',{})[name]=dict(shot=e['shot'],flag=e['flag'],
                        ends=e['ends'],labels=e['y'].cpu().tolist())
                receipt['updates'][name]=update(models[name],[e],list(replays[name]),source,
                    fast=name=='both',steps=2,guarded=True,policy=target_loss)
                replays[name].append(e)
        receipts.append(receipt)
        _durable_json(root/'shots'/f'{i+1:04d}_{sid}_audit.json',receipt)
        if (i+1)%10==0 or i==19:
            progress=dict(shots=i+1,total=360,seconds=round(time.monotonic()-start,2))
            _durable_json(root/'progress.json',progress);print(json.dumps(progress),flush=True)
    results={};per_shot={}
    for name,arm in rows.items():
        full=score_fixed_test('EAST',arm,threshold);per_shot[name]=full['predictions']
        results[name]=dict(full360=compact(arm,threshold),warmup20=compact(arm[:20],threshold),
            suffix340=compact(arm[20:],threshold),high_proxy90=compact(arm[-90:],threshold),
            blocks20=[compact(arm[i:i+20],threshold) for i in range(0,360,20)])
    if len(ledger.history)!=360 or sha(DEFAULT_PARENT)!=PARENT_SHA256:raise AssertionError('incomplete stream or changed source')
    for name in ('inter','both'):
        torch.save({k:v.cpu() for k,v in models[name].adapted_state().items()},root/'models'/f'{name}.pt')
    _durable_json(root/'per_shot.json',per_shot);_durable_json(root/'input_hashes.json',source_input_hash)
    _durable_json(root/'receipts.json',receipts)
    m=results['intra']['full360'];b=results['cold']['full360']
    passed=bool(m['J']>b['J'] and m['fpr']<=.20)
    done=dict(status='COMPLETED',metrics=results,seconds=time.monotonic()-start,
        intra_acceptance_met=passed,scientific_goal_achieved=False,
        source_bitwise_checkpoint_unchanged=True,independent_confirmation=False,
        accepted_inter_updates={k:sum(r['updates'].get(k,{}).get('accepted',False) for r in receipts) for k in ('inter','both')})
    _durable_json(root/'completed.json',done)
    print(json.dumps({k:v['full360'] for k,v in results.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--target-loss',choices=('all_points','disruptive_positive_only'),default='all_points')
    p.add_argument('--readout',choices=('shape','full','level'),default='shape')
    p.add_argument('--source-memory',type=Path)
    p.add_argument('--direction-memory',type=Path,help='audited matched-feedback direction_teaching_v1 artifact')
    p.add_argument('--event-memory',type=Path,help='audited source-only event_teaching_v1 memory; no deployed event head')
    p.add_argument('--context-memory',type=Path,help='audited source-only context8_v1 checkpoint; eight-variable fast state')
    p.add_argument('--feedback',choices=('delayed','progressive'),default='delayed')
    p.add_argument('--order-policy',choices=('legacy_proxy','current_only','current_stratified'),default='legacy_proxy')
    p.add_argument('--endpoint-policy',choices=('full_window','prefix16'),default='full_window')
    args=p.parse_args();main(args.output,args.target_loss,args.readout,args.source_memory,args.feedback,args.direction_memory,args.event_memory,args.context_memory,args.order_policy,args.endpoint_policy)
