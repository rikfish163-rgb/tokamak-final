"""Fixed frozen-source controls; no neural/threshold update or target-label fit."""
import argparse
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..fixed_test_protocol import load_test_contract, score_fixed_test
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics, sha
from ..warmstart_alignment_20261022 import alignment as scalar
from ..warmstart_alignment_20261022.run import predict_many, public, snapshot
from .alignment import summarize, combine, fit_maps, make_block_map, transform, POLICIES, BLOCKS

REFERENCE = io23.ROOT/'experiments/risk_aligned_online/runs/warmstart_alignment_test360_v2_20261022'


def stability(source, target_rows, mapping):
    result = {}
    for name, original in mapping['blocks'].items():
        distances = []; failed = []
        for row in target_rows:
            kept = [r for r in target_rows if r['shot'] != row['shot']]
            stats = combine(kept, device='EAST', allowed_ids=mapping['calibration_ids'])
            estimate = make_block_map(source['blocks'][name], stats['blocks'][name], mapping['policy'])
            if estimate['status'] != 'mapped' or original['status'] != 'mapped':
                failed.append(row['shot']); continue
            a, b = np.array(original['matrix']), np.array(estimate['matrix'])
            distances.append(dict(omitted_shot=row['shot'],
                relative_matrix_difference=float(np.linalg.norm(b-a)/np.linalg.norm(a)),
                target_mean_change_l2=float(np.linalg.norm(np.array(estimate['target_mean'])-original['target_mean']))))
        result[name] = dict(leave_one_shot_out=distances, unidentifiable_omissions=failed)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('rl_env CUDA required')
    torch.set_num_threads(2); torch.manual_seed(131); start = time.monotonic()
    contract = load_test_contract('EAST'); order = contract['shot_ids']
    base, parent = load_parent(DEFAULT_PARENT)
    model = PairedFeedback(base, np.ones(23)).cuda(); model.set_stage('frozen')
    original = {k:v.detach().cpu().clone() for k,v in model.base.state_dict().items()}
    center, scale = base.center.cpu().numpy(), base.scale.cpu().numpy()
    threshold = float(parent['threshold'])
    arms = {f'{policy}_n{n}':(n, policy) for n in (10, 20) for policy in ('density_gain', *POLICIES)}
    args.output.mkdir(parents=True); (args.output/'shots').mkdir(); (args.output/'maps').mkdir()
    code_hashes = snapshot(args.output)
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'a', zipfile.ZIP_DEFLATED) as z:
        for f in Path(__file__).parent.glob('*.py'):
            rel = str(f.relative_to(io23.ROOT)); z.write(f, rel); code_hashes[rel] = sha(f)
    plan = io23.ROOT/'docs/research/source_event_rebuild_20261022/协方差对照_事前约定_20261022.md'
    protocol = dict(format='reverse-coral-radiation-diagnostic-v1', contract=contract, arms=arms,
        source_checkpoint=str(DEFAULT_PARENT), source_sha256=PARENT_SHA256, threshold=threshold,
        coordinate=dict(name='source_asinh', center=center.tolist(), scale=scale.tolist()),
        blocks=BLOCKS, target_access='exact ended prefix signals; prediction sealed first; no outcome/event fitting',
        base='density_gain fitted to same 10/20 prefix; unchanged static before prefix ends',
        covariance='equal-shot joint-time rows, max128 per shot; total and within-shot alternatives',
        correlation_ridge=1., neural_parameter_updates=0, alarm_parameter_updates=0,
        incomplete_block='identity whole block at that timestamp; do not impute or change mask',
        reference=str(REFERENCE), reference_protocol_sha256=sha(REFERENCE/'protocol.json'),
        independent_confirmation=False, physical_calibration_certified=False,
        reproduction='reverse target-to-source CORAL-inspired diagnostic; not original source-retrain protocol',
        plan=str(plan), plan_sha256=sha(plan), code_sha256=code_hashes)
    _durable_json(args.output/'protocol.json', protocol)
    roles = io23.source_roles(); records = [{k:s[k] for k in scalar.SIGNAL_KEYS} for s in
        (simple_data.read_shot('J-TEXT', sid) for sid in roles['train'])]
    source_rows = [summarize(r, center=center, scale=scale) for r in records]
    source = combine(source_rows, device='J-TEXT', allowed_ids=roles['train'])
    source_scalar = scalar.fit_statistics(records, device='J-TEXT', allowed_ids=roles['train'])
    _durable_json(args.output/'source_shot_statistics.json', source_rows)
    _durable_json(args.output/'source_statistics.json', source)
    _durable_json(args.output/'source_scalar_statistics.json', source_scalar)
    del records, source_rows
    pending, summaries, receipts = [], [], []
    maps, density_maps = {}, {}; rows = {k:[] for k in ('static', *arms)}
    max_reference_error = 0.; aggregate = {}
    for index, sid in enumerate(order):
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        signal = {k:raw[k] for k in ('values', 'available')}; inputs = dict(static=signal)
        shot_audit = {}
        for n, dm in density_maps.items(): inputs[f'density_gain_n{n}'] = scalar.transform(signal, dm)
        for name, mapping in maps.items():
            n = arms[name][0]
            inputs[name], shot_audit[name] = transform(inputs[f'density_gain_n{n}'], mapping,
                                                       center=center, scale=scale)
            for block, vals in shot_audit[name].items():
                ag = aggregate.setdefault(name, {}).setdefault(block, {})
                for k, v in vals.items():
                    if type(v) is int: ag[k] = ag.get(k, 0)+v
        predictions = predict_many(model, inputs, device='cuda')
        for name in arms:
            if name not in predictions: predictions[name] = predictions['static'].copy()
        with np.load(REFERENCE/'shots'/f'{sid}.npz', allow_pickle=False) as ref:
            for name in ('static', 'density_gain_n10', 'density_gain_n20'):
                err = float(abs(ref[name]-predictions[name]).max())
                max_reference_error = max(max_reference_error, err)
                if err > 2e-6 or not np.array_equal(ref[name] >= threshold, predictions[name] >= threshold):
                    raise AssertionError(f'reference behavior changed: {sid} {name} {err}')
        path = args.output/'shots'/f'{sid}.npz'
        np.savez_compressed(path, times=raw['times'], **predictions)
        receipt = dict(shot=sid, prediction_sha256=sha(path), active_maps=sorted(k for k in inputs if k!='static'),
                       mapping_audit=shot_audit)
        receipts.append(receipt); _durable_json(args.output/'shots'/f'{sid}_receipt.json', receipt)
        # Neither labels nor these new statistics can affect already-sealed predictions.
        if index < 20:
            r = {k:raw[k] for k in scalar.SIGNAL_KEYS}; pending.append(r)
            summaries.append(summarize(r, center=center, scale=scale))
        if index+1 in (10, 20):
            n = index+1; seals = {str(r['shot']):r['prediction_sha256'] for r in receipts}
            dm = scalar.fit_prefix_map(source_scalar, pending, order=order, count=n, policy='density_gain')
            dm['preceding_prediction_seals'] = seals; density_maps[n] = dm
            _durable_json(args.output/'maps'/f'density_gain_n{n}.json', dm)
            _durable_json(args.output/f'target_shot_statistics_n{n}.json', summaries)
            for policy in POLICIES:
                name = f'{policy}_n{n}'
                m = fit_maps(source, summaries, order=order, count=n, policy=policy)
                m['preceding_prediction_seals'] = seals
                m['stability'] = stability(source, summaries, m)
                maps[name] = m; _durable_json(args.output/'maps'/f'{name}.json', m)
            if n == 20: pending, summaries = [], []
        outcome = simple_data.attach_outcome(raw, 'EAST', {})
        for name in rows:
            v = outcome['valid']
            rows[name].append(dict(shot=sid, times=raw['times'][v], scores=predictions[name][v],
                                  is_disrupt=outcome['is_disrupt'], down=outcome['down']))
        if (index+1)%30 == 0:
            progress = dict(shots=index+1, total=len(order), seconds=time.monotonic()-start)
            _durable_json(args.output/'progress.json', progress); print(json.dumps(progress), flush=True)
    full = {name:score_fixed_test('EAST', records, threshold) for name,records in rows.items()}
    result = {name:dict(full360=public(full[name]),
                       common_suffix340=public(metrics(records[20:], threshold, lead=30.)))
              for name,records in rows.items()}
    if any(not torch.equal(v, model.base.state_dict()[k].cpu()) for k,v in original.items()):
        raise AssertionError('source state changed')
    if sha(DEFAULT_PARENT) != PARENT_SHA256: raise AssertionError('parent changed')
    _durable_json(args.output/'prediction_receipts.json', receipts)
    _durable_json(args.output/'per_shot.json', full)
    _durable_json(args.output/'mapping_audit.json', aggregate)
    completed = dict(status='COMPLETED', metrics=result, source_bitwise_unchanged=True,
                     reference_max_abs_error=max_reference_error, neural_parameter_updates=0,
                     seconds=time.monotonic()-start, independent_confirmation=False)
    _durable_json(args.output/'completed.json', completed)
    print(json.dumps(completed), flush=True)


if __name__ == '__main__': main()
