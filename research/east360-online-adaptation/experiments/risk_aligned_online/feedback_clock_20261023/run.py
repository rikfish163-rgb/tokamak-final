"""Full fixed TEST replay of feedback timing; no new preparation/label budget."""
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
from ..fixed_test_protocol import score_fixed_test
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..target_feedback_meta_20261023.run import scored_row
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .learning import predict_clock

REFERENCE=Path(__file__).parents[1]/'runs/peak_balanced15_holdout5_test360_20261023'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.output
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('local CUDA required')
    read=lambda path:json.loads(path.read_text())
    verified=read(REFERENCE/'verification.json');old=read(REFERENCE/'protocol.json')
    assert verified['status']=='VERIFIED' and sha(REFERENCE/'completed.json')==verified['completed_sha256']
    assert sha(REFERENCE/'protocol.json')==verified['protocol_sha256']
    for f,h in old['code_sha256'].items():assert sha(io23.ROOT/f)==h,f
    prep=read(REFERENCE/'preparation_receipts.json')
    assert sha(REFERENCE/'warm.pt')==prep['warm']['sha256']
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    base,source=load_parent(DEFAULT_PARENT);model=PairedFeedback(base,np.ones(23)).cuda()
    model.load_state_dict(torch.load(REFERENCE/'warm.pt',map_location='cpu',weights_only=True)['model'])
    model.set_stage('frozen');frozen=state_cpu(model);threshold=float(source['threshold'])
    assert threshold==old['threshold']
    mapping=read(REFERENCE/'input_map.json');contract=old['contract']
    root.mkdir(parents=True);(root/'shots').mkdir()
    files={io23.ROOT/f for f in old['code_sha256']};files.update(Path(__file__).parent.glob('*.py'))
    files.add(Path(__file__).parents[1]/'tests/test_feedback_clock.py')
    hashes={str(f.relative_to(io23.ROOT)):sha(f) for f in sorted(files)}
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
    plan=io23.ROOT/'docs/research/source_event_rebuild_20261022/反馈时钟与目标新鲜度_事前约定_20261023.md'
    arms=['static']+[f'clock{i}_{t}' for i in (128,25) for t in ('four','latest')]
    protocol=dict(source_sha256=PARENT_SHA256,reference=str(REFERENCE),
        reference_protocol_sha256=sha(REFERENCE/'protocol.json'),reference_completed_sha256=sha(REFERENCE/'completed.json'),
        reference_warm_sha256=sha(REFERENCE/'warm.pt'),reference_mapping_sha256=sha(REFERENCE/'input_map.json'),
        reference_receipts_sha256=sha(REFERENCE/'prediction_receipts.json'),code_sha256=hashes,
        plan_path=str(plan),plan_sha256=sha(plan),contract=contract,threshold=threshold,arms=arms,
        intervals=[128,25],targets=[4,1],first_update_after_count=256,step_rule='.001*interval/128; gradient clip1',
        seed=131,new_supervised_training=False,new_labels=0,persistent_parameters_optimized=[],
        preparation_ids=old['selected_ids'],cached_static=True,threshold_selection=False,
        independent_confirmation=False,scope='development exposed full TEST; causal intra, no inter',
        equal_compute=False,equal_gradient_norm_step_bound_per_time_approximately=True)
    _durable_json(root/'protocol.json',protocol);_durable_json(root/'input_map.json',mapping)
    refs={x['shot']:x for x in read(REFERENCE/'prediction_receipts.json')};seals=[]
    def deny(*args,**kwargs):raise AssertionError('no HDF outcome access before all predictions sealed')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for index,sid in enumerate(contract['shot_ids']):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=transform({k:s[k] for k in ('values','available')},mapping)
            previous=REFERENCE/'shots'/f'{sid}.npz'
            assert sha(previous)==refs[sid]['prediction_sha256']
            with np.load(previous,allow_pickle=False) as trace:
                np.testing.assert_array_equal(trace['times'],s['times'])
                pred={'static':trace['static'].copy()};control=trace['warm_intra'].copy()
            updates={}
            for interval in (128,25):
                values,records=predict_clock(model,signal,interval=interval)
                for name in values:
                    arm=f'clock{interval}_{name}';pred[arm]=values[name];updates[arm]=records[name]
            np.testing.assert_array_equal(pred['clock128_four'],control)
            path=root/'shots'/f'{sid}.npz';up=root/'shots'/f'{sid}_updates.json'
            np.savez_compressed(path,times=s['times'],**pred);_durable_json(up,updates)
            seals.append(dict(shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up)))
            if (index+1)%20==0:
                progress=dict(shots=index+1,seconds=time.monotonic()-started)
                _durable_json(root/'progress.json',progress);print(json.dumps(progress),flush=True)
        _durable_json(root/'prediction_receipts.json',seals)
        _durable_json(root/'pre_eval_label_access.json',[])
    rows={a:[] for a in arms}
    for sid in contract['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
            np.testing.assert_array_equal(trace['times'],s['times'])
            for arm in arms:rows[arm].append(scored_row(s,trace[arm]))
    results={k:score_fixed_test('EAST',v,threshold) for k,v in rows.items()}
    assert_unchanged(model,frozen);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(root/'per_shot.json',results)
    metrics={k:dict(**public(v),very_early_over3000=sum(r['timely'] and r['lead_ms']>3000 for r in v['predictions']))
             for k,v in results.items()}
    _durable_json(root/'completed.json',dict(status='COMPLETED',metrics=metrics,seconds=time.monotonic()-started,
        source_unchanged=True,persistent_model_unchanged=True,original_intra_exact=True,temporary_resets=True))
    print(json.dumps(metrics),flush=True)


if __name__=='__main__':main()
