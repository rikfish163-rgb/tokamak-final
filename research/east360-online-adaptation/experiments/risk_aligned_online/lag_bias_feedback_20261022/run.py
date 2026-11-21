"""Fixed-parent, fixed-alarm OLD-bias/NEW-target intra-shot experiment."""
import argparse
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..paired_feedback_20261022.model import PairedFeedback, initial_encoder
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha, metrics
from ..fixed_test_protocol import load_test_contract, score_fixed_test
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import predict_many, public, snapshot
from .learning import adapt_arrived

REFERENCE = io23.ROOT/'experiments/risk_aligned_online/runs/head_operator_test360_20261022'
MAP_PATH = io23.ROOT/'experiments/risk_aligned_online/runs/deepcoral_prefix_test360_v2_20261022/density_map.json'
ARMS = ('static', 'original4', 'recent', 'lag_bias')


def predict_arms(model, signal, *, device='cuda'):
    if (set(signal) != {'values', 'available'} or signal['values'].ndim != 2
            or signal['values'].shape != signal['available'].shape
            or signal['values'].shape[1] != 23 or not len(signal['values'])):
        raise ValueError('nonempty signal-only Nx23 record required')
    if model.cfg.window != 128 or model.cfg.horizon != 25:
        raise ValueError('fixed W128/H25 parent required')
    n = len(signal['values'])
    states = {k:initial_encoder(model) for k in ('static', 'recent', 'lag_bias')}
    predictions = {k:[] for k in states}
    updates = {k:[] for k in ('recent', 'lag_bias')}
    for lo in range(0,n,128):
        hi = min(lo+128,n)
        for start in range(lo,hi,32):
            stop = min(hi,start+32)
            windows = [simple_data.window(signal,t,128) for t in range(start,stop)]
            x,m = [torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
            for name,state in states.items():
                h = model.predict_fast(x,m,state,mode='risk_state')['risk_state']
                predictions[name].append(model.base.risk_head(h).squeeze(-1).sigmoid().detach().cpu().numpy())
        # No step after the last prediction; all updates use completed targets
        # and affect ONLY later endpoints. Fast state is fixed within a block.
        if hi < 256 or hi == n:
            continue
        for name in updates:
            states[name], receipt = adapt_arrived(model,signal,hi,states[name],
                                                  device=device,correct=name=='lag_bias')
            updates[name].append(receipt)
    result = {k:np.concatenate(v) for k,v in predictions.items()}
    if not all(np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('nonfinite risk trace')
    for name in updates:
        np.testing.assert_array_equal(result[name][:256],result['static'][:256])
    return result, updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2); torch.manual_seed(131)
    start = time.monotonic()
    base,parent = load_parent(DEFAULT_PARENT)
    model = PairedFeedback(base,np.ones(23)).cuda(); model.set_stage('frozen')
    original = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    contract = load_test_contract('EAST'); order = contract['shot_ids']
    threshold = float(parent['threshold'])
    ref_protocol = json.loads((REFERENCE/'protocol.json').read_text())
    assert threshold == ref_protocol['threshold'] and order == ref_protocol['contract']['shot_ids']
    assert ref_protocol['source_sha256'] == PARENT_SHA256
    assert ref_protocol['density_mapping_sha256'] == sha(MAP_PATH)
    mapping = json.loads(MAP_PATH.read_text())
    assert mapping['calibration_ids'] == order[:20]
    old_seals = {r['shot']:r for r in json.loads((REFERENCE/'prediction_receipts.json').read_text())}
    assert list(old_seals) == order
    args.output.mkdir(parents=True); (args.output/'shots').mkdir()
    hashes = snapshot(args.output)
    dependencies = list(Path(__file__).parent.glob('*.py')) + [
        Path(__file__).parents[1]/'deepcoral_prefix_20261022/learning.py',
        Path(__file__).parents[1]/'tests/test_lag_bias_feedback.py']
    with zipfile.ZipFile(args.output/'code_snapshot.zip','a',zipfile.ZIP_DEFLATED) as z:
        for f in dependencies:
            z.write(f,str(f.relative_to(io23.ROOT)))
            hashes[str(f.relative_to(io23.ROOT))] = sha(f)
    plan = io23.ROOT/'docs/research/source_event_rebuild_20261022/历史偏差与动态误差分离_事前约定.md'
    _durable_json(args.output/'protocol.json',dict(contract=contract,threshold=threshold,arms=ARMS,
        source_checkpoint=str(DEFAULT_PARENT),source_sha256=PARENT_SHA256,
        reference=str(REFERENCE),reference_protocol_sha256=sha(REFERENCE/'protocol.json'),
        reference_receipts_sha256=sha(REFERENCE/'prediction_receipts.json'),
        density_mapping_path=str(MAP_PATH),density_mapping_sha256=sha(MAP_PATH),
        code_sha256=hashes,plan_path=str(plan),plan_sha256=sha(plan),
        parameter_scope=model.encoder_names(),
        feedback='older two H25 residual means detached; newer two H25 gradient targets',
        update='temporary encoder SGD0.001; global gradient norm cap1; every128points; first affects256',
        controls='recent uses identical eligible NEW targets without OLD bias subtraction; original4 is archived',
        information_budget='both compute OLD+NEW; matched gradient targets, not a claim that OLD calibration is free information',
        warmup20='common original predictions; density mapping activates after sealing first20',
        target_labels_accessed_only_after_all360_predictions=True,
        no_new_persistent_training=True,independent_confirmation=False))
    seals = []
    for index,sid in enumerate(order):
        s = apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        signal = {k:s[k] for k in ('values','available')}
        ref_path = REFERENCE/'shots'/f'{sid}.npz'
        assert sha(ref_path) == old_seals[sid]['prediction_sha256']
        with np.load(ref_path,allow_pickle=False) as ref:
            np.testing.assert_array_equal(s['times'],ref['times'])
            if index < 20:
                raw = predict_many(model,dict(raw=signal),device='cuda')['raw']
                predictions = {k:raw.copy() for k in ARMS}; updates = {}
                np.testing.assert_array_equal(raw,ref['encoder_original'])
            else:
                predictions,updates = predict_arms(model,transform(signal,mapping))
                predictions['original4'] = ref['encoder_original'].copy()
            np.testing.assert_array_equal(predictions['static'],ref['static'])
        path = args.output/'shots'/f'{sid}.npz'
        update_path = args.output/'shots'/f'{sid}_updates.json'
        np.savez_compressed(path,times=s['times'],**predictions)
        _durable_json(update_path,updates)
        seals.append(dict(shot=sid,prediction_sha256=sha(path),updates_sha256=sha(update_path)))
        if index == 19:
            _durable_json(args.output/'activation_receipt.json',dict(after_sealed_ids=order[:20],
                preceding_seals=seals.copy(),mapping_sha256=sha(MAP_PATH)))
        if (index+1)%40 == 0:
            progress = dict(shots=index+1,seconds=time.monotonic()-start)
            _durable_json(args.output/'progress.json',progress); print(json.dumps(progress),flush=True)
    _durable_json(args.output/'prediction_receipts.json',seals)
    # Evaluation labels are read only AFTER the complete prediction seal.
    rows = {k:[] for k in ARMS}
    for sid in order:
        s = simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(args.output/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
            for name in ARMS:
                v = s['valid']
                rows[name].append(dict(shot=sid,times=s['times'][v],scores=trace[name][v],
                    is_disrupt=s['is_disrupt'],down=s['down']))
    full = {k:score_fixed_test('EAST',v,threshold) for k,v in rows.items()}
    result = {k:dict(full360=public(full[k]),common_suffix340=public(metrics(v[20:],threshold,lead=30.)))
              for k,v in rows.items()}
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v.cpu(),original[k],rtol=0,atol=0)
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    _durable_json(args.output/'per_shot.json',full)
    _durable_json(args.output/'completed.json',dict(status='COMPLETED',metrics=result,
        persistent_model_bitwise_unchanged=True,seconds=time.monotonic()-start))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
