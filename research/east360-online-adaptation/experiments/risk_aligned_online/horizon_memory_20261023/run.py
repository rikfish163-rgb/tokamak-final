"""Four-fold whole-shot development evaluation, not a new TEST360 result."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..learned_update_loss_20261023.learning import folds
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent,load_source_train,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics,sha
from ..warmstart_alignment_20261022.alignment import fit_statistics,transform,SIGNAL_KEYS
from ..warmstart_alignment_20261022.run import public
from .prepare import prepare
from .learning import ARMS,FoldAccess,make_memory,predict,gate

ROOT=io23.ROOT
SELECTION=ROOT/'experiments/risk_aligned_online/runs/matched_preparation20_test360_20261023/selection.json'
PLAN=ROOT/'docs/research/source_event_rebuild_20261022/分炮准备与后续经验检索_事前约定_20261023.md'


def snapshot(output):
    files=set(Path(__file__).parent.glob('*.py'))
    # Existing source imports are captured transitively after module loading.
    import sys
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):
            p=Path(module.__file__)
            if p.suffix=='.py':files.add(p)
    files.add(ROOT/'experiments/risk_aligned_online/tests/test_horizon_memory.py')
    hashes={}
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            name=str(p.relative_to(ROOT));archive.write(p,name);hashes[name]=sha(p)
    return hashes


def score(root,plans,threshold):
    rows={a:[] for a in ARMS};label_access=[]
    for plan in plans:
        for sid in plan['held']:
            s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
            label_access.append(dict(fold=plan['fold'],shot=sid,phase='score_after_all_prediction_seals'))
            with np.load(root/f'fold{plan["fold"]}'/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
                np.testing.assert_array_equal(trace['times'],s['times'])
                for a in ARMS:
                    rows[a].append(dict(shot=sid,times=s['times'][s['valid']],scores=trace[a][s['valid']],
                                        is_disrupt=s['is_disrupt'],down=s['down']))
    results={a:metrics(v,threshold,lead=30.) for a,v in rows.items()}
    _durable_json(root/'evaluation_access.json',label_access);_durable_json(root/'per_shot.json',results)
    return results


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env local CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    selected=json.loads(SELECTION.read_text())['selected'];plans=folds(selected)
    base,info=load_parent(DEFAULT_PARENT);parent=PairedFeedback(base,np.ones(23)).cuda()
    parent.set_stage('frozen');original=state_cpu(parent);threshold=float(info['threshold'])
    source_ids=io23.source_roles()['train'];root.mkdir(parents=True)
    protocol=dict(source_checkpoint=str(DEFAULT_PARENT),source_sha256=PARENT_SHA256,
        selection_sha256=sha(SELECTION),selected_ids=sorted(r['shot'] for r in selected),
        plan_path=str(PLAN),plan_sha256=sha(PLAN),code_sha256=snapshot(root),folds=plans,arms=ARMS,
        threshold=threshold,seed=131,torch_version=torch.__version__,cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),independent_confirmation=False,
        scope='four-fold DEVELOPMENT whole-shot evaluation; not TEST360; metadata-stratified nonchronological',
        input_mapping='source TRAIN / fold fit15 only; empirical density_gain, not certified units',
        preparation='512 conditional-moments first-encoder steps +256 forecast-head-only steps',
        memory='same480 labeled endpoints; key ends e vs e-25; no held labels or signals in fold fit',
        primary='lag_rank',no_inter=True,no_alarm_adjustment=True,
        intra='cumulative first encoder, reset each shot; interval128; first after256; original head frozen',
        source_training_role_ids=source_ids)
    _durable_json(root/'protocol.json',protocol)
    read_outcome,read_signal=io23.read_outcome,simple_data.read_shot;source_access=[]
    def source_guard(device,sid):
        if device!='J-TEXT' or sid not in source_ids:raise AssertionError('source TRAIN outcomes only')
        source_access.append(sid);return read_outcome(device,sid)
    with patch.object(io23,'read_outcome',side_effect=source_guard):source,_=load_source_train()
    _durable_json(root/'source_label_access.json',source_access)
    source_stats=fit_statistics([{k:s[k] for k in SIGNAL_KEYS} for s in source],device='J-TEXT',allowed_ids=source_ids)
    _durable_json(root/'source_statistics.json',source_stats)
    seals=[];total_updates=0
    for plan in plans:
        f=plan['fold'];out=root/f'fold{f}';out.mkdir();(out/'shots').mkdir()
        access=FoldAccess(plan,source_ids)
        def allowed_signal(device,sid,**kwargs):
            access.check('signal',device,sid);return read_signal(device,sid,**kwargs)
        def allowed_outcome(device,sid):
            access.check('outcome',device,sid);return read_outcome(device,sid)
        with patch.object(simple_data,'read_shot',side_effect=allowed_signal),patch.object(io23,'read_outcome',side_effect=allowed_outcome):
            model,mapping,records=prepare(parent,plan['fit'],source,source_stats,out)
            frozen=state_cpu(model);memory,manifest=make_memory(model,records)
            torch.save({k:torch.as_tensor(v).cpu() for k,v in memory.items()},out/'memory.pt')
            _durable_json(out/'memory_manifest.json',manifest)
            access.phase='predict'
            for sid in plan['held']:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal=transform({k:s[k] for k in ('values','available')},mapping)
                predictions,updates=predict(model,signal,memory)
                path=out/'shots'/f'{sid}.npz';up=out/'shots'/f'{sid}_updates.json'
                np.savez_compressed(path,times=s['times'],**predictions);_durable_json(up,updates)
                total_updates+=sum(r['gradient_steps'] for rows in updates.values() for r in rows)
                seals.append(dict(fold=f,shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up)))
                progress=dict(stage='held_stream',fold=f,shot=sid,sealed=len(seals),seconds=time.monotonic()-started)
                _durable_json(root/'progress.json',progress);print(progress,flush=True)
            assert_unchanged(model,frozen)
        _durable_json(out/'access.json',access.log);assert_unchanged(parent,original)
        del model,memory,records
    _durable_json(root/'prediction_receipts.json',seals)
    results=score(root,plans,threshold)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    summary={a:dict(**public(r),very_early_over3000=sum(p['timely'] and p['lead_ms']>3000 for p in r['predictions'])) for a,r in results.items()}
    done=dict(status='COMPLETED',metrics=summary,recommend_test360=gate(results),seconds=time.monotonic()-started,
        neural_steps=total_updates,source_unchanged=True,fold_models_unchanged_during_stream=True,
        independent_confirmation=False,test360_accessed=False,
        artifact_sha256={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()})
    _durable_json(root/'completed.json',done);print(json.dumps({k:v for k,v in done.items() if k!='artifact_sha256'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args().output)
