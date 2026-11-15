"""Sequential predict -> durable seal -> natural outcome -> persistent learning."""
import argparse
import copy
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..diverse_support_20261023.run import READ,make_parent
from ..event_prefix_probe import read_events,DEFAULT_LABELS,LABEL_SHA256
from ..heldout_threshold_20261023.run import REFERENCE,REFERENCE_SHA,models_for_calibration
from ..persistence_control_20261023.run import CALIBRATION,CALIBRATION_SHA
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics,sha
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .learning import Access,episode_indices,select_pair,trim_memory,after_shot_step,predict

PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/逐炮经验训练炮内适应_实验约定_20261023.md'
ARMS=('static','fixed_intra','inter_only','meta_static','meta_intra','meta_hold')


def outcome_record(sid,outcome,digest,index):
    flag,down=outcome
    if flag and (down is None or not np.isfinite(down)):
        raise ValueError('a disruptive outcome needs its actual event time')
    # HDF5 safety outcomes legitimately have no disruption timestamp. Keep
    # that absence explicit instead of inventing a terminal time or a zero.
    return dict(shot=sid,is_disrupt=bool(flag),down=None if down is None else float(down),
                prediction_sha256=digest,index=index)


def labeled(s, outcome, mapping):
    flag,down=outcome;y,valid=io23.labels_at(s['times'],flag,down,125.)
    signal=transform({k:s[k] for k in ('values','available')},mapping)
    return dict(signal,shot=s['shot'],times=s['times'],is_disrupt=flag,down=down,y=y,valid=valid)


def optimizer(model):
    names=set(model.encoder_names())
    return torch.optim.Adam([p for k,p in model.named_parameters() if k in names],lr=3e-5)


def train_current(meta,inter,opts,memory,sid,index):
    rng=np.random.default_rng(131+index);rows=[]
    for step in range(4):
        episodes=select_pair(memory,sid,rng)
        if not episodes:
            rows.append(dict(step=step,reason='no_authorized_ordered_episode'));continue
        assert all(s['shot'] in memory for s,e in episodes)
        pair={name:after_shot_step(net,opts[name],episodes,adapt=name=='meta',device='cuda')
              for name,net in (('inter',inter),('meta',meta))}
        rows.append(dict(step=step,**pair))
    return rows


def save_checkpoint(path,index,meta,inter,opts,memory):
    torch.save(dict(next_index=index,meta=state_cpu(meta),inter=state_cpu(inter),
        optimizers={k:v.state_dict() for k,v in opts.items()},memory_ids=list(memory)),path)


def evaluate(root,protocol,outcomes):
    rows={a:[] for a in ARMS}
    for row in outcomes:
        sid=row['shot'];flag=row['is_disrupt'];down=row['down']
        with np.load(root/'shots'/f'{sid}.npz') as z:
            _,valid=io23.labels_at(z['times'],flag,down,125.)
            for arm in ARMS:rows[arm].append(dict(shot=sid,times=z['times'][valid],scores=z[arm][valid],is_disrupt=flag,down=down))
    results={};blocks=[]
    for label,threshold in (('calibrated',protocol['threshold']),('source_threshold',protocol['source_threshold'])):
        for arm in ARMS:
            m=metrics(rows[arm],threshold,lead=30.)
            m['very_early_over3000']=sum(r['timely'] and r['lead_ms']>3000 for r in m['predictions'])
            results[arm+'/'+label]=m
    for hi in range(20,len(outcomes)+20,20):
        hi=min(hi,len(outcomes));lo=max(0,hi-20)
        blocks.append(dict(start=lo,end=hi,cumulative={a:public(metrics(rows[a][:hi],protocol['threshold'],lead=30.)) for a in ARMS},
            block={a:public(metrics(rows[a][lo:hi],protocol['threshold'],lead=30.)) for a in ARMS}))
    _durable_json(root/'per_shot.json',results);_durable_json(root/'blocks.json',blocks)
    return {k:public(v) for k,v in results.items()}


def main(root,limit=None):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    assert sha(REFERENCE/'completed.json')==REFERENCE_SHA
    assert sha(CALIBRATION/'completed.json')==CALIBRATION_SHA
    rp=READ(REFERENCE/'protocol.json');cp=READ(CALIBRATION/'protocol.json')
    for base in (REFERENCE,CALIBRATION):
        done=READ(base/'completed.json')
        for name,digest in done['artifact_sha256'].items():assert sha(base/name)==digest,name
    order=rp['shot_ids'] if limit is None else rp['shot_ids'][:limit]
    assert order==sorted(order) and not set(order)&set(cp['calibration_ids'])
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    parent,source_threshold=make_parent();origin=models_for_calibration(parent)['mapped_fixed']
    untouched=state_cpu(origin);meta=copy.deepcopy(origin);inter=copy.deepcopy(origin)
    opts=dict(meta=optimizer(meta),inter=optimizer(inter));mapping=READ(REFERENCE/'input_map.json')
    gate=Access(order,rp['selected']);memory={};reader=io23.read_outcome
    def prepare_guard(device,sid):
        assert device=='EAST' and sid in rp['selected'],'only authorized preparation labels'
        return reader(device,sid)
    with patch.object(io23,'read_outcome',side_effect=prepare_guard):
        for sid in rp['selected']:
            raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            s=labeled(raw,io23.read_outcome('EAST',sid),mapping)
            memory[sid]=(s,episode_indices(s))
    assert sha(DEFAULT_LABELS)==LABEL_SHA256
    events,metadata=read_events(DEFAULT_LABELS);train=set(io23.source_roles()['train'])
    event_info=dict(total_labeled_shots=len(metadata),total_events=sum(map(len,events.values())),
        train_event_shots=len(set(events)&train),train_events=sum(len(es) for sid,es in events.items() if sid in train),
        used_for_training=False,sha256=LABEL_SHA256)
    root.mkdir(parents=True);(root/'shots').mkdir();(root/'checkpoints').mkdir()
    files=set(Path(__file__).parent.glob('*.py'));files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_continual_episode.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    p=dict(shot_ids=order,preparation_ids=rp['selected'],calibration_ids=cp['calibration_ids'],
        threshold=READ(CALIBRATION/'thresholds.json')['mapped_fixed']['0.2']['threshold'],source_threshold=source_threshold,
        source_sha256=PARENT_SHA256,reference_sha256=REFERENCE_SHA,calibration_sha256=CALIBRATION_SHA,
        plan_sha256=sha(PLAN),code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        input_sha256={str(io23.path_for('EAST',s)):sha(io23.path_for('EAST',s)) for s in order+rp['selected']},
        event_labels=event_info,arms=ARMS,inter_steps=4,inter_learning_rate=3e-5,seed=131,
        intra_learning_rate=.001,alarm_updates=0,reveal_mode='all outcomes after each sealed shot; no query saving claim',
        parameter_names=meta.encoder_names(),parameter_count=sum(p.numel() for k,p in meta.named_parameters() if k in meta.encoder_names()),
        scope='DEVELOPMENT_ONLY; historical preparation plus ascending prequential stream; shadow arms share meta parent',
        independent_confirmation=False,gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    _durable_json(root/'protocol.json',p)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
    save_checkpoint(root/'checkpoints'/'before000.pt',0,meta,inter,opts,memory)
    receipts=[];outcomes=[]
    def deny(*args,**kwargs):raise AssertionError('no outcome access before prediction seal')
    for index,sid in enumerate(order):
        gate.begin(sid)
        if time.monotonic()-started>1800:raise TimeoutError('declared 30-minute experiment budget exceeded')
        with patch.object(io23,'read_outcome',side_effect=deny):
            raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=transform({k:raw[k] for k in ('values','available')},mapping)
            curves,updates=predict(meta,inter,signal)
            with np.load(REFERENCE/'fold0/shots'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(raw['times'],z['times'])
                curves.update(static=z['mapped_fixed_static'].copy(),fixed_intra=z['mapped_fixed_intra'].copy())
                if index==0:
                    np.testing.assert_array_equal(curves['meta_static'],curves['static'])
                    np.testing.assert_array_equal(curves['meta_intra'],curves['fixed_intra'])
            path=root/'shots'/f'{sid}.npz';up=root/'shots'/f'{sid}_intra.json'
            np.savez_compressed(path,times=raw['times'],**curves);_durable_json(up,updates)
            digest=sha(path);gate.seal(digest)
            _durable_json(root/'shots'/f'{sid}_seal.json',dict(shot=sid,index=index,sha256=digest,model_version=index))
        # The reader can be called exactly once by the gate and only after
        # the current prediction file and its hash seal have been written.
        assert sha(path)==gate.digest
        outcome=gate.reveal(lambda current:reader('EAST',current))
        s=labeled(raw,outcome,mapping);memory[sid]=(s,episode_indices(s));trim_memory(memory)
        assert set(memory)<=gate.allowed
        row=outcome_record(sid,outcome,digest,index)
        outcomes.append(row);_durable_json(root/'shots'/f'{sid}_outcome.json',row)
        with patch.object(io23,'read_outcome',side_effect=deny):
            training=train_current(meta,inter,opts,memory,sid,index)
        tr=root/'shots'/f'{sid}_inter.json'
        _durable_json(tr,dict(shot=sid,affected_shots='strictly later stream positions',memory_ids=list(memory),steps=training))
        receipts.append(dict(shot=sid,index=index,prediction_sha256=digest,intra_sha256=sha(up),inter_sha256=sha(tr)))
        gate.finish()
        if (index+1)%20==0 or index+1==len(order):
            save_checkpoint(root/'checkpoints'/f'before{index+1:03}.pt',index+1,meta,inter,opts,memory)
            _durable_json(root/'progress.json',dict(completed_shots=index+1,seconds=time.monotonic()-started,last_shot=sid))
            print(dict(shots=index+1,seconds=time.monotonic()-started,memory=len(memory)),flush=True)
    _durable_json(root/'prediction_receipts.json',receipts);_durable_json(root/'outcomes.json',outcomes)
    for net in (meta,inter):assert_unchanged(net,untouched,meta.encoder_names())
    assert_unchanged(origin,untouched);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=evaluate(root,p,outcomes)
    done=dict(status='COMPLETED_PREQUENTIAL_EPISODE_EXPERIMENT',metrics=result,seconds=time.monotonic()-started,
        source_unchanged=True,online_labels_revealed=len(order),event_labels=event_info,
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',done)
    print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--limit',type=int)
    args=parser.parse_args();main(args.root,args.limit)
