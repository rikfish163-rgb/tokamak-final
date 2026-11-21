"""Preparation20 then full causal360; learn policy, never a score correction."""
import argparse
import copy
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..continual_episode_20261023.learning import Access, episode_indices, select_pair, trim_memory
from ..continual_episode_20261023.run import labeled, outcome_record
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..diverse_support_20261023.run import READ, make_parent
from ..heldout_threshold_20261023.run import REFERENCE, models_for_calibration
from ..paired_feedback_20261022.model import initial_encoder
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics, sha
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .learning import ARMS, WeightPolicy, observed, train_policy, predict

PREVIOUS = io23.ROOT/'experiments/risk_aligned_online/runs/continual_episode360_20261023'
PREVIOUS_SHA = '5d6b902050c0787f040426c1a4fa2e2a90d509147de3d291fcee9c03f5a8b118'
PLAN = io23.ROOT/'docs/research/source_event_rebuild_20261022/轨迹条件反馈权重_事前约定_20261023.md'
ALL_ARMS = ('static', 'uniform') + ARMS


def read_signal(sid):
    return apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))


def normalization(model, memory, *, device):
    contexts, rows = [], []
    fast = initial_encoder(model)
    for sid,(s,es) in memory.items():
        for count in sorted({e['count'] for e in es}):
            prefix = {k:s[k][:count] for k in ('values','available')}
            *_, context = observed(model,prefix,count,fast,device=device)
            contexts.append(context); rows.append(dict(shot=sid,count=count))
    if not contexts:
        raise ValueError('preparation needs observed ordered episodes')
    x = np.asarray(contexts)
    return x.mean(0), np.maximum(x.std(0),1e-4), rows


def prepare_pair(memory, rng):
    episodes = []
    for label in (False, True):
        ids = [sid for sid,(s,es) in memory.items() if s['is_disrupt'] == label and es]
        if not ids:
            raise ValueError('both preparation outcome classes required')
        sid = int(rng.choice(ids)); s,es = memory[sid]
        targets = sorted({e['label'] for e in es}); target = int(rng.choice(targets))
        options = [e for e in es if e['label'] == target]
        episodes.append((s,options[int(rng.integers(len(options)))]))
    return episodes


def train_ended(model, policy, optimizer, memory, sid, index):
    rng = np.random.default_rng(131+index)
    records = []
    for k in range(4):
        pair = select_pair(memory,sid,rng)
        if not pair:
            records.append(dict(step=k,reason='no_authorized_ordered_episode'))
        else:
            records.append(dict(step=k,**train_policy(model,policy,optimizer,pair,device='cuda')))
    return records


def checkpoint(path, index, fixed, online, optimizer, memory):
    torch.save(dict(next_index=index,fixed=state_cpu(fixed),online=state_cpu(online),
                    optimizer=optimizer.state_dict(),memory_ids=list(memory)),path)


def evaluate(root, protocol, outcomes):
    rows = {a:[] for a in ALL_ARMS}
    for r in outcomes:
        with np.load(root/'shots'/f'{r["shot"]}.npz') as z:
            _,valid = io23.labels_at(z['times'],r['is_disrupt'],r['down'],125.)
            for a in rows:
                rows[a].append(dict(shot=r['shot'],times=z['times'][valid],scores=z[a][valid],
                                    is_disrupt=r['is_disrupt'],down=r['down']))
    results, blocks = {}, []
    for label,threshold in (('calibrated',protocol['threshold']),('source_threshold',protocol['source_threshold'])):
        for a in rows:
            m = metrics(rows[a],threshold,lead=30.)
            m['very_early_over3000'] = sum(r['timely'] and r['lead_ms']>3000 for r in m['predictions'])
            results[a+'/'+label] = m
    for end in range(20,len(outcomes)+20,20):
        end = min(end,len(outcomes))
        blocks.append(dict(end=end,cumulative={a:public(metrics(rs[:end],protocol['threshold'],lead=30.)) for a,rs in rows.items()}))
    _durable_json(root/'per_shot.json',results); _durable_json(root/'blocks.json',blocks)
    return {k:public(v) for k,v in results.items()}


def main(root, limit=None):
    if root.exists():
        raise FileExistsError(root)
    if not torch.cuda.is_available():
        raise RuntimeError('rl_env CUDA required')
    assert sha(PREVIOUS/'completed.json') == PREVIOUS_SHA
    prior = READ(PREVIOUS/'protocol.json')
    for f,h in READ(PREVIOUS/'completed.json')['artifact_sha256'].items():
        assert sha(PREVIOUS/f) == h, f
    assert READ(PREVIOUS/'verification.json')['completed_sha256'] == PREVIOUS_SHA
    torch.set_num_threads(2); torch.manual_seed(131)
    started = time.monotonic()
    parent,_ = make_parent(); model = models_for_calibration(parent)['mapped_fixed']
    original = state_cpu(model); mapping = READ(REFERENCE/'input_map.json')
    memory = {}; reader = io23.read_outcome
    def prep_reader(device,sid):
        assert device=='EAST' and sid in prior['preparation_ids']
        return reader(device,sid)
    with patch.object(io23,'read_outcome',side_effect=prep_reader):
        for sid in prior['preparation_ids']:
            s = labeled(read_signal(sid),io23.read_outcome('EAST',sid),mapping)
            memory[sid] = (s,episode_indices(s))
    center,scale,norm_rows = normalization(model,memory,device='cuda')
    online = WeightPolicy(center,scale).cuda()
    optimizer = torch.optim.Adam(online.parameters(),lr=.01)
    root.mkdir(parents=True); (root/'shots').mkdir(); (root/'checkpoints').mkdir()
    order = prior['shot_ids'] if limit is None else prior['shot_ids'][:limit]
    assert not set(order)&set(prior['calibration_ids'])
    files = set(Path(__file__).parent.glob('*.py'))
    files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_contextual_feedback.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):
            files.add(Path(module.__file__))
    protocol = dict(shot_ids=order,preparation_ids=prior['preparation_ids'],calibration_ids=prior['calibration_ids'],
        threshold=prior['threshold'],source_threshold=prior['source_threshold'],source_sha256=PARENT_SHA256,
        previous_sha256=PREVIOUS_SHA,plan_sha256=sha(PLAN),seed=131,preparation_steps=256,post_shot_steps=4,
        policy_learning_rate=.01,intra_learning_rate=.001,arms=ALL_ARMS,
        policy_parameter_count=sum(v.numel() for v in online.parameters()),
        persistent_classifier_updates=0,temporary_encoder_parameters=model.encoder_names(),
        label_access='all360 naturally revealed after current prediction seal',event_labels_used=False,
        normalization=dict(center=center.tolist(),scale=scale.tolist(),rows=norm_rows),
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        input_sha256=prior['input_sha256'],scope='DEVELOPMENT_ONLY; contextual rule only, frozen persistent risk model',
        independent_confirmation=False,gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    _durable_json(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(files):archive.write(f,str(f.relative_to(io23.ROOT)))
    rng = np.random.default_rng(131); preparation=[]
    def deny(*args,**kwargs):raise AssertionError('no current or future outcomes in prediction/training')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for step in range(256):
            preparation.append(train_policy(model,online,optimizer,prepare_pair(memory,rng),device='cuda'))
            if (step+1)%64==0:print(dict(stage='prepare',step=step+1,seconds=time.monotonic()-started),flush=True)
    _durable_json(root/'preparation.json',preparation)
    fixed = copy.deepcopy(online); frozen_policy = state_cpu(fixed)
    checkpoint(root/'checkpoints/before000.pt',0,fixed,online,optimizer,memory)
    gate = Access(order,prior['preparation_ids']); outcomes=[]; receipts=[]
    for index,sid in enumerate(order):
        if time.monotonic()-started>1800:raise TimeoutError('declared30min exceeded')
        gate.begin(sid)
        with patch.object(io23,'read_outcome',side_effect=deny):
            raw = read_signal(sid)
            signal = transform({k:raw[k] for k in ('values','available')},mapping)
            curves,updates = predict(model,fixed,online,signal)
            with np.load(PREVIOUS/'shots'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(raw['times'],z['times'])
                curves.update(static=z['static'].copy(),uniform=z['fixed_intra'].copy())
            if index==0:np.testing.assert_array_equal(curves['fixed'],curves['online'])
            for a in ARMS:np.testing.assert_array_equal(curves[a][:256],curves['static'][:256])
            path=root/'shots'/f'{sid}.npz'; up=root/'shots'/f'{sid}_intra.json'
            np.savez_compressed(path,times=raw['times'],**curves);_durable_json(up,updates)
            digest=sha(path);gate.seal(digest)
            _durable_json(root/'shots'/f'{sid}_seal.json',dict(shot=sid,index=index,sha256=digest,policy_version=index))
        assert sha(path)==gate.digest
        outcome=gate.reveal(lambda current:reader('EAST',current))
        s=labeled(raw,outcome,mapping);memory[sid]=(s,episode_indices(s));trim_memory(memory)
        assert set(memory)<=gate.allowed
        row=outcome_record(sid,outcome,digest,index);outcomes.append(row)
        _durable_json(root/'shots'/f'{sid}_outcome.json',row)
        with patch.object(io23,'read_outcome',side_effect=deny):
            training=train_ended(model,online,optimizer,memory,sid,index)
        tr=root/'shots'/f'{sid}_inter.json'
        _durable_json(tr,dict(shot=sid,memory_ids=list(memory),steps=training))
        receipts.append(dict(shot=sid,prediction_sha256=digest,intra_sha256=sha(up),inter_sha256=sha(tr)))
        gate.finish()
        if (index+1)%20==0 or index+1==len(order):
            checkpoint(root/'checkpoints'/f'before{index+1:03}.pt',index+1,fixed,online,optimizer,memory)
            status=dict(shots=index+1,last_shot=sid,seconds=time.monotonic()-started)
            _durable_json(root/'progress.json',status);print(status,flush=True)
    _durable_json(root/'outcomes.json',outcomes);_durable_json(root/'prediction_receipts.json',receipts)
    assert_unchanged(model,original);assert_unchanged(fixed,frozen_policy);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    results=evaluate(root,protocol,outcomes)
    done=dict(status='COMPLETED_CONTEXTUAL_FEEDBACK_EXPERIMENT',metrics=results,source_unchanged=True,
        persistent_classifier_unchanged=True,seconds=time.monotonic()-started,
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',done)
    print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--limit',type=int)
    args=parser.parse_args();main(args.root,args.limit)
