"""Replay the exact original learning history; intervene only in diagnostic reads."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import load_context_memory
from ..delayed_recurrent_memory_20261025.learning import batch
from ..east_context_operator_20261031.run import configure, read
from ..fixed_test_protocol import score_fixed_test
from ..lifecycle_stream_20261025.learning import source_replay, target_example, update
from ..lifecycle_stream_20261025.protocol import AccessLedger
from ..lifecycle_stream_20261025.run import build_adapter, compact, STATISTICS
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import fit_prefix_map, transform
from .factors import logits_with_components

REFERENCE = io23.ROOT/'probe_outputs/stratified_cold20_context360_v1'
REFERENCE_SHA = 'ef6e2fccd56f7a8b7bc0f93c44315cf77028ee6e122d74758e5f0678baa937ef'
PLAN = io23.ROOT/'SLOW_FAST_ATTRIBUTION_PLAN_20261031.md'
ARMS = ('cold_off', 'cold_on', 'both_off', 'both_on', 'both_slow_cold_readout',
        'cold_slow_both_readout', 'inter_off', 'inter_slow_cold_readout')
MATCHED = dict(cold_off='cold', cold_on='intra', both_off='both_shadow',
               both_on='both', inter_off='inter')
CASES = (60239, 64562, 83376, 94402)


def states_equal(model, state):
    actual = model.adapted_state()
    return actual.keys() == state.keys() and all(torch.equal(actual[k], state[k].to(actual[k])) for k in actual)


@torch.no_grad()
def predict(models, signal, *, case):
    cold = models['cold']
    delta, info = cold.memory_delta(signal)
    c, b, i = [models[k].adapted_state() for k in ('cold', 'both', 'inter')]
    assert torch.equal(i['fast_output'], c['fast_output'])
    specs = dict(cold_off=(c, c, False), cold_on=(c, c, True),
                 both_off=(b, b, False), both_on=(b, b, True),
                 both_slow_cold_readout=(b, c, True), cold_slow_both_readout=(c, b, True),
                 inter_off=(i, i, False), inter_slow_cold_readout=(i, c, True))
    logs = {arm:[] for arm in ARMS}
    curves = {arm:[] for arm in ARMS}
    internals = []
    for lo in range(0, len(signal['values']), 32):
        ends = np.arange(lo, min(lo+32, len(signal['values'])))
        x, available = batch(signal, ends, cold.base.center.device)
        h, valid = cold.tokens(x, available)
        d = delta[ends]
        for arm, (slow, output, on) in specs.items():
            action = d if on else torch.zeros_like(d)
            z = logits_with_components(cold, h, valid, action, slow['slow'], output['fast_output'])
            if not torch.isfinite(z).all():
                raise FloatingPointError('nonfinite intervention logits')
            logs[arm].append(z.cpu().numpy())
            curves[arm].append(z.sigmoid().cpu().numpy())
        if case:
            fast_c = d@c['fast_output'].T * valid[:, -2, None]
            fast_b = d@b['fast_output'].T * valid[:, -2, None]
            slow_c = h[:, -2]@c['slow'].T * valid[:, -2, None]
            slow_b = h[:, -2]@b['slow'].T * valid[:, -2, None]
            internals.append(torch.stack((fast_c.norm(dim=1), fast_b.norm(dim=1),
                                          slow_c.norm(dim=1), slow_b.norm(dim=1)), 1).cpu().numpy())
    logs = {arm:np.concatenate(values) for arm, values in logs.items()}
    curves = {arm:np.concatenate(values) for arm, values in curves.items()}
    return curves, logs, delta, dict(fast=info, internal_norms=np.concatenate(internals) if case else None)


def main(root):
    if root.exists():
        raise FileExistsError(root)
    configure()
    start = time.monotonic()
    assert sha(REFERENCE/'completed.json') == REFERENCE_SHA
    p = read(REFERENCE/'protocol.json')
    assert p['target_loss'] == 'disruptive_positive_only' and p['endpoint_policy'] == 'full_window'
    order = read(REFERENCE/'order.json')
    ids = [row['shot'] for row in order]
    assert len(ids) == len(set(ids)) == 360
    old_receipts = read(REFERENCE/'receipts.json')
    raw_hashes = read(REFERENCE/'input_hashes.json')
    ledger = AccessLedger(ids, 20)
    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda()
    threshold = float(parent['threshold'])
    assert threshold == p['threshold']
    models = {name:build_adapter(base, readout='full', feedback='progressive', context=True)
              for name in ('cold', 'inter', 'both')}
    for model in models.values():
        assert load_context_memory(model, Path(p['source_memory']['path'])) == p['source_memory']
    for name in ('inter', 'both'):
        models[name].load_state_dict(models['cold'].state_dict())
    assert sha(STATISTICS) == p['source_statistics_sha256']
    source_stats = read(STATISTICS)
    root.mkdir(parents=True)
    for name in ('shots', 'states', 'updates'):
        (root/name).mkdir()
    files = set(Path(__file__).parent.glob('*.py')) | {PLAN}
    for name, module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module, '__file__', None):
            files.add(Path(module.__file__))
    protocol = dict(scope='POSTHOC_COMPONENT_INTERVENTION_EXACT_ORIGINAL_LEARNING_HISTORY',
                    reference=str(REFERENCE), reference_completed_sha256=REFERENCE_SHA,
                    source_checkpoint=str(DEFAULT_PARENT), source_sha256=PARENT_SHA256,
                    order=order, threshold=threshold, arms=ARMS, matched= MATCHED,
                    counterfactual_curves_used_for_training=False, case_ids=CASES,
                    warmup=20, full_stream_count=360, first20_cost_included=True,
                    plan_sha256=sha(PLAN), torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
                    independent_confirmation=False, no_new_online_policy_claim=True)
    _durable_json(root/'protocol.json', protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))
    source, events = source_replay(models['cold'])
    assert events == read(REFERENCE/'source_replay.json')
    _durable_json(root/'source_replay.json', events)
    rows = {arm:[] for arm in ARMS}
    histories = {arm:deque(maxlen=8) for arm in ('inter', 'both')}
    pending, map_records, mapping = [], [], None
    match_count, update_count = 0, 0
    cases, state_rows = [], []
    for index, sid in enumerate(ids):
        assert sha(io23.path_for('EAST', sid)) == raw_hashes[str(sid)]
        prefix = f'{index+1:04d}_{sid}'
        saved = REFERENCE/'shots'/f'{prefix}.npz'
        assert sha(saved) == read(REFERENCE/'shots'/f'{prefix}_seal.json')['sha256']
        with np.load(saved, allow_pickle=False) as data:
            originals = {arm:data[arm].copy() for arm in ('source', *set(MATCHED.values()))}
            old_times = data['times'].copy()
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        assert np.array_equal(raw['times'], old_times)
        signal = {k:raw[k] for k in ('values', 'available')}
        info, logs, delta = {}, {}, None
        if index < 20:
            curves = {arm:originals['source'].copy() for arm in ARMS}
        else:
            signal = transform(signal, mapping)
            states = {name:{k:v.cpu() for k, v in model.adapted_state().items()}
                      for name, model in models.items()}
            state_path = root/'states'/f'{prefix}.pt'
            torch.save(states, state_path)
            curves, logs, delta, info = predict(models, signal, case=sid in CASES)
            state_rows.append(dict(shot=sid, index=index+1, state_sha256=sha(state_path),
                norm={name:{k:float(v.norm()) for k, v in state.items()} for name, state in states.items()},
                cold_distance={name:{k:float((v-states['cold'][k]).norm()) for k, v in state.items()}
                               for name, state in states.items()}))
        for arm, old_arm in MATCHED.items():
            assert np.array_equal(curves[arm], originals[old_arm]), (sid, arm, 'original curve drift')
            match_count += 1
        output_path = root/'shots'/f'{prefix}.npz'
        payload = dict(times=raw['times'], **curves, **{'logit_'+k:v for k, v in logs.items()})
        if info.get('internal_norms') is not None:
            payload['internal_norms_fastC_fastB_slowC_slowB'] = info['internal_norms']
        with output_path.open('xb') as f:
            np.savez_compressed(f, **payload)
            f.flush()
            os.fsync(f.fileno())
        digest = sha(output_path)
        ledger.seal(sid, digest)
        _durable_json(root/'shots'/f'{prefix}_seal.json', dict(shot=sid, index=index+1,
                      sha256=digest, outcome_available=False))
        ledger.reveal(sid)
        outcome = simple_data.attach_outcome(raw, 'EAST', {})
        for arm in ARMS:
            valid = outcome['valid']
            rows[arm].append(dict(shot=sid, times=raw['times'][valid], scores=curves[arm][valid],
                                  is_disrupt=outcome['is_disrupt'], down=outcome['down']))
        if sid in CASES and index >= 20:
            end = int(curves['both_on'].argmax())
            interaction = logs['both_on']-logs['both_slow_cold_readout']-logs['cold_slow_both_readout']+logs['cold_on']
            cases.append(dict(shot=sid, index=index+1, reference_endpoint=end,
                reference_endpoint_rule='original both peak, posthoc diagnostic only',
                at_same_endpoint={arm:dict(logit=float(logs[arm][end]), probability=float(curves[arm][end])) for arm in ARMS},
                maxima={arm:float(curves[arm].max()) for arm in ARMS},
                first_alarms={arm:int(np.flatnonzero(curves[arm]>=threshold)[0])
                              if (curves[arm]>=threshold).any() else None for arm in ARMS},
                internal_norms_at_reference=info['internal_norms'][end].tolist(),
                nonlinear_logit_interaction_at_reference=float(interaction[end]),
                nonlinear_logit_interaction_mean_abs=float(abs(interaction).mean())))
        if index < 20:
            pending.append((outcome, originals['source']))
            map_records.append({k:raw[k] for k in ('shot', 'device', 'values', 'available')})
        if index == 19:
            mapping = fit_prefix_map(source_stats, map_records, order=ids, count=20, policy='density_gain')
            assert mapping == read(REFERENCE/'input_map.json')
            _durable_json(root/'input_map.json', mapping)
            examples = []
            for shot, scores in pending:
                mapped = transform({k:shot[k] for k in ('values', 'available')}, mapping)
                d, _ = models['cold'].memory_delta(mapped)
                examples.append(target_example(models['cold'], shot, scores, mapped, d))
            fitted = update(models['cold'], examples, [], source, fast='mixed', steps=20,
                            guarded=False, policy=p['target_loss'])
            assert fitted == old_receipts[index]['cold_fit'], 'cold learning receipts differ'
            old_state = torch.load(REFERENCE/'models/cold.pt', map_location='cuda', weights_only=True)
            assert states_equal(models['cold'], old_state), 'cold parameters differ'
            _durable_json(root/'cold_fit.json', fitted)
            for name in histories:
                models[name].restore(models['cold'].adapted_state())
                histories[name].extend(examples)
            pending, map_records = [], []
        elif index >= 20:
            receipts = {}
            for name in histories:
                example = target_example(models[name], outcome, originals[name], signal, delta)
                receipt = update(models[name], [example], list(histories[name]), source,
                                 fast=name=='both', steps=2, guarded=True, policy=p['target_loss'])
                assert receipt == old_receipts[index]['updates'][name], (sid, name, 'learning replay drift')
                receipts[name] = receipt
                histories[name].append(example)
                update_count += 1
            _durable_json(root/'updates'/f'{prefix}.json', receipts)
        if (index+1) % 20 == 0:
            progress = dict(shots=index+1, total=360, seconds=time.monotonic()-start,
                            exact_original_curves=match_count, exact_update_receipts=update_count)
            _durable_json(root/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    for name in histories:
        expected = torch.load(REFERENCE/'models'/f'{name}.pt', map_location='cuda', weights_only=True)
        assert states_equal(models[name], expected), ('final state', name)
    assert sha(DEFAULT_PARENT) == PARENT_SHA256 and len(ledger.history) == 360
    results, per_shot = {}, {}
    for arm, values in rows.items():
        per_shot[arm] = score_fixed_test('EAST', values, threshold)['predictions']
        results[arm] = dict(full360=compact(values, threshold), warmup20=compact(values[:20], threshold),
                            suffix340=compact(values[20:], threshold))
    _durable_json(root/'per_shot.json', per_shot)
    _durable_json(root/'cases.json', cases)
    _durable_json(root/'state_evolution.json', state_rows)
    completed = dict(status='COMPLETED', seconds=time.monotonic()-start, metrics=results,
                     exact_original_curves=match_count, exact_update_receipts=update_count,
                     exact_cold_fit_steps=20, exact_final_inter_and_both_states=True,
                     source_checkpoint_unchanged=True, independent_confirmation=False,
                     scope='DIAGNOSTIC_SWAPS_ON_ORIGINAL_HISTORY_NOT_NEW_POLICY', scientific_goal_achieved=False)
    _durable_json(root/'completed.json', completed)
    print(json.dumps({k:v['full360'] for k, v in results.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
