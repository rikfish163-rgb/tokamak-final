"""EFD-inspired numerical-scale diagnostic with the qualified source model.

Target median/IQR statistics are estimated from the first 30 signal records
in the locked shot-number order. This matches the model's existing asinh
coordinates, not the thesis z-score pipeline. No neural parameter is fitted.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import io23, simple_data
from .alignment import fit_ended_shot_statistics
from .causal_input import apply_input_contract
from .redesign_20261004.embedding_model import EmbeddingModel
from .redesign_20261004.intra import predict_shot
from .scale_policy import POLICIES, scale_variant
from .simple23 import Config, Simple23
from .simple_online import _hash, _durable_json, seal_predictions
from .simple_train import metrics
from .source_gate import qualify_source


def split_ids(order, cohort, count=30):
    ids = [r['shot'] for r in order]
    if (type(count) is not int or count < 1 or count >= len(ids)
            or any(type(s) is not int for s in ids) or len(ids) != len(set(ids))
            or not set(cohort) <= set(ids) or len(cohort) != len(set(cohort))):
        raise ValueError('explicit unique locked order and cohort required')
    calibration = ids[:count]
    held = [sid for sid in ids[count:] if sid in set(cohort)]
    if not held:
        raise ValueError('empty held cohort')
    return calibration, held


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--qa-convention',choices=('storage','source_ratio'),default='storage',
                   help='pointwise source QA convention; not geometric/physical q calibration')
    p.add_argument('--scale-policies', nargs='+', choices=POLICIES,
                   default=['target_robust_scale'])
    args = p.parse_args()
    if (args.output.exists() or args.limit < 0
            or len(args.scale_policies) != len(set(args.scale_policies))):
        raise ValueError('new output directory, nonnegative limit and distinct policies required')
    root = Path(__file__).parent
    ckpt_path = root/'runs/gradient_aligned512_20260929/source.pt'
    order_path = root/'runs/east_all_nominal_stream_20261004/order.json'
    gate = qualify_source(ckpt_path)
    if not gate['qualified'] or not torch.cuda.is_available():
        raise RuntimeError('qualified parent and local CUDA required')
    torch.set_num_threads(4)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if state['model_config']['input_view'] != 'source':
        raise ValueError('existing source-coordinate model required')
    source = Simple23(Config(**state['model_config']), state['source_center'], state['source_scale'])
    source.load_state_dict(state['model'], strict=True)
    qa_receipt=None
    if args.qa_convention=='source_ratio':
        from .qa_convention import fit_source_qa_convention
        qa_receipt=fit_source_qa_convention(
            simple_data.read_shot('J-TEXT',sid) for sid in simple_data.source_roles()['train'])
    order = json.loads(order_path.read_text())
    calibration, held = split_ids(order, io23.cohort_ids('EAST'))
    selected = held[:args.limit] if args.limit else held
    stats = fit_ended_shot_statistics([
        apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        for sid in calibration], ended_ids=calibration)
    # The archived reference uses this wrapper and differentiable eval route.
    # Its persistent/temporary deltas stay zero here; using no_grad instead
    # selects a different Transformer kernel and changes near-threshold scores.
    models = {name: EmbeddingModel(deepcopy(source)).to('cuda').eval()
              for name in ('source_mask_matched', *args.scale_policies)}
    with torch.no_grad():
        for name in args.scale_policies:
            variant = scale_variant(state['source_center'], state['source_scale'], stats, name)
            models[name].center.copy_(torch.from_numpy(variant['center']))
            models[name].scale.copy_(torch.from_numpy(variant['scale']))
    args.output.mkdir(parents=True)
    for name in models:
        (args.output/name).mkdir()
    _durable_json(args.output/'statistics.json',
                  {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in stats.items()})
    protocol = dict(scope='DEVELOPMENT SCALE DIAGNOSTIC; zero classifier training, 30 past records for statistics',
        calibration_ids=calibration, evaluation_ids=selected,
        held_total=len(held), source_sha256=gate['checkpoint_sha256'], order_sha256=_hash(order_path),
        threshold=float(state['threshold']), rate_contract='nominal',
        neural_training_steps=0, statistics='median/IQR, approximate equal sampling per shot',
        scale_policies=args.scale_policies,
        interventions='numeric diagnostics; no physical unit equivalence inferred',
        mask_matched_channels=[io23.RAW23[i] for i in np.flatnonzero(~stats['usable'])],
        code_sha256={name: _hash(root/name) for name in ('paper131_scale_diagnostic.py', 'alignment.py',
            'causal_input.py', 'scale_policy.py', 'simple_data.py', 'io23.py', 'simple23.py', 'simple_online.py', 'simple_train.py',
            'redesign_20261004/embedding_model.py', 'redesign_20261004/intra.py')},
        warning='local cohort, not certified thesis roster; source weights and threshold fixed')
    protocol.update(qa_convention=args.qa_convention,qa_source_receipt=qa_receipt)
    protocol['code_sha256']['qa_convention.py']=_hash(root/'qa_convention.py')
    _durable_json(args.output/'protocol.json', protocol)
    receipts = {}
    static_dir = root/'runs/redesign_static_all2953_20261004/static'
    for path in static_dir.glob('*_seal.json'):
        r = json.loads(path.read_text())
        if r['shot'] in set(selected):
            receipts[r['shot']] = (path, r)
    if set(receipts) != set(selected):
        raise ValueError('missing existing static reference')
    rows = {name: [] for name in ('original_static', *models)}
    start_time = time.monotonic()
    for index, sid in enumerate(selected, 1):
        original = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        shot = dict(original, available=original['available'] & stats['usable'][None, :])
        if qa_receipt is not None:
            from .qa_convention import rebuild_qa
            shot=rebuild_qa(shot,qa_receipt['coefficient'])
        path, receipt = receipts[sid]
        assert receipt['threshold'] == state['threshold']
        npz = path.parent/receipt['npz']
        assert _hash(npz) == receipt['sha256']
        with np.load(npz, allow_pickle=False) as seal:
            assert np.array_equal(seal['times'], shot['times']) and int(seal['shot']) == sid
            reference = seal['scores'].copy()
        # Reproduce the frozen parent at a real shot before comparing scales.
        if index == 1:
            probe_model = EmbeddingModel(deepcopy(source)).to('cuda').eval()
            scores, audit = predict_shot(probe_model, original, intra=False, device='cuda')
            assert audit['accepted_updates'] == 0
            difference = float(np.max(np.abs(scores-reference)))
            if difference > 2e-6:
                raise ValueError(f'parent replay differs from sealed reference: {difference}')
            protocol['parent_probe_max_abs_error'] = difference
            protocol['numerical_path'] = audit['numerical_path']
            del probe_model
        emitted = dict(original_static=reference)
        for name, model in models.items():
            scores, audit = predict_shot(model, shot, intra=False, device='cuda')
            assert audit['accepted_updates'] == 0
            seal_predictions(args.output/name, index, sid, shot['times'], scores,
                             arm=name, threshold=float(state['threshold']))
            emitted[name] = scores
        outcome = simple_data.attach_outcome(original, 'EAST', {})
        valid = outcome['valid'].copy()
        if outcome['is_disrupt']:
            valid &= original['times'] <= outcome['down']
        for name, scores in emitted.items():
            rows[name].append(dict(shot=sid, is_disrupt=outcome['is_disrupt'], down=outcome['down'],
                times=original['times'][valid], scores=scores[valid]))
        if index % 10 == 0 or index == len(selected):
            print(json.dumps(dict(observed=index, total=len(selected), elapsed_s=time.monotonic()-start_time)),flush=True)
    final = {}
    for name, arm in rows.items():
        final[name] = metrics(arm, state['threshold'], lead=30.)
    for model in models.values():
        for name, param in model.base.named_parameters():
            assert torch.equal(param.detach().cpu(), state['model'][name])
        assert torch.count_nonzero(model.delta_weight) == 0
        assert torch.count_nonzero(model.delta_bias) == 0
    assert _hash(ckpt_path) == gate['checkpoint_sha256']
    _durable_json(args.output/'protocol.json', protocol)
    _durable_json(args.output/'result.json', dict(metrics=final, elapsed_s=time.monotonic()-start_time,
        neural_parameters_bitwise_unchanged=True, parent_sha256=gate['checkpoint_sha256']))
    print(json.dumps({name:{k:v for k,v in m.items() if k != 'predictions'} for name,m in final.items()}),flush=True)


if __name__ == '__main__':
    main()
