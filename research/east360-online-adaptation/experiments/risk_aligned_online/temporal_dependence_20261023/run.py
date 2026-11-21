"""Full TEST360 retrospective temporal-dependence diagnosis; no training."""
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
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .controls import predict_controls

REFERENCE=io23.ROOT/'experiments/risk_aligned_online/runs/labeled_alignment20_test360_20261023'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/时间变化与水平依赖_诊断约定_20261023.md'


def load_model(reference=REFERENCE):
    prep=json.loads((reference/'preparation_receipts.json').read_text())
    assert sha(reference/'input_only.pt')==prep['input_only']['checkpoint_sha256']
    base,parent=load_parent(DEFAULT_PARENT);model=PairedFeedback(base,np.ones(23))
    source=state_cpu(model.base)
    model.load_state_dict(torch.load(reference/'input_only.pt',map_location='cpu',weights_only=True)['model'])
    assert_unchanged(model.base,source,('forecast_head.weight','forecast_head.bias'))
    model.cuda();model.set_stage('frozen')
    return model,float(parent['threshold'])


def main(output):
    if output.exists():raise FileExistsError(output)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    read=lambda p:json.loads(p.read_text())
    rp=read(REFERENCE/'protocol.json');rv=read(REFERENCE/'verification.json')
    assert rv['status']=='VERIFIED' and sha(REFERENCE/'completed.json')==rv['completed_sha256']
    model,threshold=load_model();frozen=state_cpu(model);contract=load_test_contract('EAST')
    mapping=read(REFERENCE/'input_map.json');assert rp['contract']==contract
    assert not set(mapping['calibration_ids'])&set(contract['shot_ids'])
    seals={r['shot']:r for r in read(REFERENCE/'prediction_receipts.json')}
    output.mkdir(parents=True);(output/'shots').mkdir()
    files=list(Path(__file__).parent.glob('*.py'))+[io23.ROOT/'experiments/risk_aligned_online'/n for n in (
        'simple23.py','simple_data.py','io23.py','causal_input.py','simple_online.py','simple_train.py',
        'fixed_test_protocol.py','paired_feedback_20261022/model.py','masked_state_20261022/model.py',
        'redesign_20261004/prepare_prior.py','warmstart_alignment_20261022/alignment.py',
        'warmstart_alignment_20261022/run.py','deepcoral_prefix_20261022/run.py','tests/test_temporal_dependence.py')]
    hashes={str(f.relative_to(io23.ROOT)):sha(f) for f in files}
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,str(f.relative_to(io23.ROOT)))
    arms=[f'{a}_{b}' for a in ('raw','mapped') for b in ('original','density_flat','all_flat')]
    protocol=dict(scope='POSTHOC MODEL SENSITIVITY, NOT A TRAINED METHOD OR PHYSICAL CAUSAL CLAIM',
        source_sha256=PARENT_SHA256,threshold=threshold,contract=contract,arms=arms,
        reference=str(REFERENCE),reference_hashes={name:sha(REFERENCE/name) for name in (
            'protocol.json','completed.json','verification.json','input_only.pt','input_map.json','prediction_receipts.json')},
        plan_path=str(PLAN),plan_sha256=sha(PLAN),code_sha256=hashes,
        formula='within each current W128 window, observed values -> per-channel median',
        neural_updates=0,threshold_changes=0,future_window_access=False,
        caveat='moving window medians retain long-time evolution; artificial windows may be out of distribution')
    _durable_json(output/'protocol.json',protocol);_durable_json(output/'input_map.json',mapping)
    receipts=[]
    with patch.object(io23,'read_outcome',side_effect=AssertionError('no outcome before all curves sealed')):
        for index,sid in enumerate(contract['shot_ids']):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            raw={k:s[k] for k in ('values','available')};mapped=transform(raw,mapping)
            curves={f'{a}_{b}':v for a,signal in (('raw',raw),('mapped',mapped))
                    for b,v in predict_controls(model,signal).items()}
            reference=REFERENCE/'shots'/f'{sid}.npz';assert sha(reference)==seals[sid]['prediction_sha256']
            with np.load(reference,allow_pickle=False) as old:
                np.testing.assert_array_equal(s['times'],old['times'])
                np.testing.assert_array_equal(curves['raw_original'],old['raw_source'])
                np.testing.assert_array_equal(curves['mapped_original'],old['input_only_static'])
            path=output/'shots'/f'{sid}.npz';np.savez_compressed(path,times=s['times'],**curves)
            receipts.append(dict(shot=sid,prediction_sha256=sha(path)))
            if (index+1)%30==0:
                progress=dict(shots=index+1,seconds=time.monotonic()-start)
                _durable_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
        _durable_json(output/'prediction_receipts.json',receipts)
    rows={a:[] for a in arms}
    for sid in contract['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(output/'shots'/f'{sid}.npz',allow_pickle=False) as z:
            for a in arms:rows[a].append(dict(shot=sid,times=s['times'][s['valid']],scores=z[a][s['valid']],
                                             is_disrupt=s['is_disrupt'],down=s['down']))
    results={a:score_fixed_test('EAST',r,threshold) for a,r in rows.items()}
    assert_unchanged(model,frozen);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(output/'per_shot.json',results)
    complete=dict(status='COMPLETED',seconds=time.monotonic()-start,neural_updates=0,source_unchanged=True,
        metrics={a:dict(**public(r),very_early_over3000=sum(p['timely'] and p['lead_ms']>3000 for p in r['predictions']))
                 for a,r in results.items()})
    _durable_json(output/'completed.json',complete);print(json.dumps(complete),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args();main(args.output)
