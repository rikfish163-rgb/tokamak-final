"""Fixed-TEST360 comparison of two source-taught feedback objectives."""
import argparse
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_train import sha,metrics
from ..simple_online import _durable_json
from ..fixed_test_protocol import load_test_contract,score_fixed_test
from ..deepcoral_prefix_20261022.run import predict_pairs
from ..warmstart_alignment_20261022.run import snapshot,public
from ..warmstart_alignment_20261022.alignment import transform

REFERENCE=io23.ROOT/'experiments/risk_aligned_online/runs/head_operator_test360_20261022'
MAP_PATH=io23.ROOT/'experiments/risk_aligned_online/runs/deepcoral_prefix_test360_v2_20261022/density_map.json'
ARMS=('static','original','forecast','meta')


def load_models(prepared):
    done=json.loads((prepared/'prepare_completed.json').read_text());assert done['status']=='COMPLETED'
    protocol=json.loads((prepared/'protocol.json').read_text())
    assert protocol['source_sha256']==PARENT_SHA256
    for path,h in protocol['code_sha256'].items():assert sha(io23.ROOT/path)==h
    assert sha(protocol['plan_path'])==protocol['plan_sha256']
    models={}
    for arm in ('forecast','meta'):
        path=prepared/f'{arm}.pt';assert sha(path)==done['head_sha256'][arm]
        payload=torch.load(path,map_location='cpu',weights_only=True)
        assert payload['parent_sha256']==PARENT_SHA256
        base,parent=load_parent(DEFAULT_PARENT)
        base.forecast_head.load_state_dict(payload['forecast_head'],strict=True)
        model=PairedFeedback(base,np.ones(23)).cuda();model.set_stage('frozen');models[arm]=model
    return models,parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    models,parent=load_models(args.prepared);threshold=float(parent['threshold'])
    states={k:{n:v.detach().cpu().clone() for n,v in m.state_dict().items()} for k,m in models.items()}
    contract=load_test_contract('EAST');order=contract['shot_ids']
    old=json.loads((REFERENCE/'protocol.json').read_text())
    assert old['contract']==contract and threshold==old['threshold']
    refs={r['shot']:r for r in json.loads((REFERENCE/'prediction_receipts.json').read_text())}
    mapping=json.loads(MAP_PATH.read_text());assert mapping['calibration_ids']==order[:20]
    args.output.mkdir(parents=True);(args.output/'shots').mkdir()
    hashes=snapshot(args.output)
    deps=[Path(__file__),Path(__file__).parents[1]/'deepcoral_prefix_20261022/run.py',
          Path(__file__).parents[1]/'deepcoral_prefix_20261022/learning.py']
    with zipfile.ZipFile(args.output/'code_snapshot.zip','a',zipfile.ZIP_DEFLATED) as z:
        for f in deps:z.write(f,str(f.relative_to(io23.ROOT)));hashes[str(f.relative_to(io23.ROOT))]=sha(f)
    _durable_json(args.output/'protocol.json',dict(contract=contract,threshold=threshold,arms=ARMS,
        source_sha256=PARENT_SHA256,code_sha256=hashes,prepared=str(args.prepared.resolve()),
        preparation_protocol_sha256=sha(args.prepared/'protocol.json'),
        preparation_completion_sha256=sha(args.prepared/'prepare_completed.json'),
        reference=str(REFERENCE),reference_protocol_sha256=sha(REFERENCE/'protocol.json'),
        mapping_path=str(MAP_PATH),mapping_sha256=sha(MAP_PATH),
        reused=['static','original'],first20_common_static=True,
        first_affected_shot_index=20,no_target_outcomes_in_updates=True,
        all_outcomes_read_after360_prediction_seals=True,independent_confirmation=False))
    seals=[]
    for i,sid in enumerate(order):
        shot=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        path=REFERENCE/'shots'/f'{sid}.npz';assert sha(path)==refs[sid]['prediction_sha256']
        with np.load(path,allow_pickle=False) as ref:
            predictions=dict(static=ref['static'].copy(),original=ref['encoder_original'].copy())
            if i<20:
                predictions.update({k:predictions['static'].copy() for k in models});updates={}
            else:
                signal=transform({k:shot[k] for k in ('values','available')},mapping)
                fresh,updates=predict_pairs(models,signal)
                for arm in models:
                    np.testing.assert_array_equal(fresh[f'{arm}_static'],predictions['static'])
                    predictions[arm]=fresh[f'{arm}_intra']
                    np.testing.assert_array_equal(predictions[arm][:256],predictions['static'][:256])
        out=args.output/'shots'/f'{sid}.npz';uout=args.output/'shots'/f'{sid}_updates.json'
        np.savez_compressed(out,times=shot['times'],**predictions);_durable_json(uout,updates)
        seals.append(dict(shot=sid,prediction_sha256=sha(out),updates_sha256=sha(uout)))
        if i==19:_durable_json(args.output/'activation_receipt.json',dict(after_seals=seals.copy(),mapping_sha256=sha(MAP_PATH)))
        if (i+1)%40==0:
            progress=dict(shots=i+1,seconds=time.monotonic()-start)
            _durable_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    _durable_json(args.output/'prediction_receipts.json',seals)
    rows={k:[] for k in ARMS}
    for sid in order:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(args.output/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
            for arm in ARMS:
                v=s['valid'];rows[arm].append(dict(shot=sid,times=s['times'][v],scores=trace[arm][v],
                                                 is_disrupt=s['is_disrupt'],down=s['down']))
    full={k:score_fixed_test('EAST',r,threshold) for k,r in rows.items()}
    result={k:dict(full360=public(full[k]),common_suffix340=public(metrics(r[20:],threshold,lead=30.))) for k,r in rows.items()}
    for k,m in models.items():
        for n,v in m.state_dict().items():torch.testing.assert_close(v.cpu(),states[k][n],rtol=0,atol=0)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(args.output/'per_shot.json',full)
    _durable_json(args.output/'completed.json',dict(status='COMPLETED',metrics=result,
        persistent_models_bitwise_unchanged=True,seconds=time.monotonic()-start))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
