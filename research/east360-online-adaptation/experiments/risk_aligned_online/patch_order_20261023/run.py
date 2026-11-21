"""All-shot fixed-endpoint patch-order sensitivity; frozen neural parameters."""
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
from ..fixed_test_protocol import load_test_contract,score_fixed_test
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..temporal_dependence_20261023.run import load_model,REFERENCE
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .control import predict_reversed

TEMPORAL=io23.ROOT/'experiments/risk_aligned_online/runs/temporal_dependence20_test360_20261023'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/固定末端的时间片顺序_诊断约定_20261023.md'
ARMS=['raw_original','raw_reversed','mapped_original','mapped_reversed']


def main(output):
    if output.exists():raise FileExistsError(output)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    read=lambda p:json.loads(p.read_text())
    tp=read(TEMPORAL/'protocol.json');tv=read(TEMPORAL/'verification.json')
    assert tv['status']=='VERIFIED' and sha(TEMPORAL/'completed.json')==tv['completed_sha256']
    assert sha(TEMPORAL/'protocol.json')==tv['protocol_sha256']
    for name,h in tp['code_sha256'].items():assert sha(io23.ROOT/name)==h,name
    for name,h in tp['reference_hashes'].items():assert sha(REFERENCE/name)==h,name
    model,threshold=load_model();frozen=state_cpu(model);contract=load_test_contract('EAST')
    assert tp['contract']==contract and tp['threshold']==threshold
    mapping=read(TEMPORAL/'input_map.json')
    assert mapping==read(REFERENCE/'input_map.json')
    assert not set(mapping['calibration_ids'])&set(contract['shot_ids'])
    seals={r['shot']:r for r in read(TEMPORAL/'prediction_receipts.json')}
    output.mkdir(parents=True);(output/'shots').mkdir()
    files=[io23.ROOT/name for name in tp['code_sha256']]+list(Path(__file__).parent.glob('*.py'))
    files += [io23.ROOT/'experiments/risk_aligned_online/tests/test_patch_order.py']
    hashes={str(f.relative_to(io23.ROOT)):sha(f) for f in files}
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in hashes:z.write(io23.ROOT/name,name)
    protocol=dict(scope='POSTHOC ORDER SENSITIVITY; NOT ADAPTATION OR PHYSICAL CAUSAL EVIDENCE',
        source_sha256=PARENT_SHA256,threshold=threshold,contract=contract,arms=ARMS,
        plan_path=str(PLAN),plan_sha256=sha(PLAN),code_sha256=hashes,
        temporal_reference=str(TEMPORAL),preparation_reference=str(REFERENCE),
        temporal_hashes={n:sha(TEMPORAL/n) for n in ('protocol.json','completed.json','verification.json',
            'input_map.json','prediction_receipts.json','per_shot.json')},
        formula='patches [0,...,14,15] -> [14,...,0,15], within-patch order fixed',
        neural_updates=0,threshold_changes=0,reference_originals='cached sealed curves, not full fresh reinference',
        gate='full128 record points and temporally constant per-channel availability; others unchanged',
        future_window_access=False)
    _durable_json(output/'protocol.json',protocol);_durable_json(output/'input_map.json',mapping)
    receipts=[]
    with patch.object(io23,'read_outcome',side_effect=AssertionError('no outcome until curves sealed')):
        for index,sid in enumerate(contract['shot_ids']):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            raw={k:s[k] for k in ('values','available')};mapped=transform(raw,mapping)
            path=TEMPORAL/'shots'/f'{sid}.npz';assert sha(path)==seals[sid]['prediction_sha256']
            curves={}
            with np.load(path,allow_pickle=False) as old:
                np.testing.assert_array_equal(s['times'],old['times'])
                for view,signal in (('raw',raw),('mapped',mapped)):
                    r=predict_reversed(model,signal)
                    curves[view+'_original']=old[view+'_original'].copy()
                    curves[view+'_reversed']=r['probability']
                    curves[view+'_eligible']=r['eligible'];curves[view+'_changed']=r['changed']
                    np.testing.assert_array_equal(r['probability'][~r['eligible']],old[view+'_original'][~r['eligible']])
            np.testing.assert_array_equal(curves['raw_eligible'],curves['mapped_eligible'])
            path=output/'shots'/f'{sid}.npz';np.savez_compressed(path,times=s['times'],**curves)
            receipts.append(dict(shot=sid,prediction_sha256=sha(path)))
            if (index+1)%30==0:
                progress=dict(shots=index+1,seconds=time.monotonic()-start)
                _durable_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
        _durable_json(output/'prediction_receipts.json',receipts)
    rows={a:[] for a in ARMS}
    for sid in contract['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(output/'shots'/f'{sid}.npz',allow_pickle=False) as z:
            for a in ARMS:rows[a].append(dict(shot=sid,times=s['times'][s['valid']],scores=z[a][s['valid']],
                                              is_disrupt=s['is_disrupt'],down=s['down']))
    results={a:score_fixed_test('EAST',r,threshold) for a,r in rows.items()}
    assert_unchanged(model,frozen);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(output/'per_shot.json',results)
    complete=dict(status='COMPLETED',seconds=time.monotonic()-start,source_unchanged=True,neural_updates=0,
        metrics={a:dict(**public(r),very_early_over3000=sum(v['timely'] and v['lead_ms']>3000 for v in r['predictions']))
                 for a,r in results.items()})
    _durable_json(output/'completed.json',complete);print(json.dumps(complete),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args().output)
