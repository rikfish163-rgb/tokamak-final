"""Four independent prequential histories; only acceptance criteria differ."""
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
from torch.nn import functional as F

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import load_context_memory
from ..delayed_recurrent_memory_20261025.learning import batch
from ..east_context_operator_20261031.run import configure, read
from ..fixed_test_protocol import score_fixed_test
from ..lifecycle_stream_20261025.learning import source_replay, target_example
from ..lifecycle_stream_20261025.protocol import AccessLedger
from ..lifecycle_stream_20261025.run import build_adapter, compact, STATISTICS
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..slow_fast_attribution_20261031.run import REFERENCE, REFERENCE_SHA
from ..warmstart_alignment_20261022.alignment import fit_prefix_map, transform
from .learning import update_with_guards

POLICIES = ('original', 'dual', 'paired', 'half')
ARMS = ('cold_off', 'cold_on', *(p+'_'+mode for p in POLICIES for mode in ('on','off')), 'paired_reverse')
MATCHED = dict(cold_off='cold', cold_on='intra', original_on='both', original_off='both_shadow')
PLAN = io23.ROOT/'PAIRED_GAP_FORWARD_PLAN_20261031.md'
COLD_SHA = 'a06377a218a0f06def1eb3652cc091b5c514653681bffcf5dbfda7906f9db344'


@torch.no_grad()
def predict(models, signal):
    cold = models['cold']
    delta, fast_info = cold.memory_delta(signal)
    reverse, _ = cold.memory_delta(signal, 'reverse')
    curves = {arm:[] for arm in ARMS}
    action_norms = {name:[] for name in ('cold', *POLICIES)}
    for lo in range(0, len(signal['values']), 32):
        hi = min(lo+32, len(signal['values']))
        ends = np.arange(lo, hi)
        x, available = batch(signal, ends, cold.base.center.device)
        h, valid = cold.tokens(x, available)
        action = delta[ends]
        for name, model in models.items():
            for mode in ('on','off'):
                d = action if mode == 'on' else torch.zeros_like(action)
                z = model.risk(h, valid, d)
                if not torch.isfinite(z).all():
                    raise FloatingPointError('nonfinite prequential risk')
                curves[name+'_'+mode].append(z.sigmoid().cpu().numpy())
            change = F.linear(action, model.memory.output.weight)*valid[:, -2, None]
            action_norms[name].append(change.norm(dim=1).cpu().numpy())
        z = models['paired'].risk(h, valid, reverse[ends])
        curves['paired_reverse'].append(z.sigmoid().cpu().numpy())
    curves = {arm:np.concatenate(values) for arm, values in curves.items()}
    effects = {}
    for name, values in action_norms.items():
        norms = np.concatenate(values)
        difference = abs(curves[name+'_on']-curves[name+'_off'])
        effects[name] = dict(mean_action_norm=float(norms.mean()), max_action_norm=float(norms.max()),
                             nonzero_action_points=int((norms>0).sum()),
                             mean_abs_probability_change=float(difference.mean()),
                             max_abs_probability_change=float(difference.max()),
                             nonzero_probability_points=int((difference>0).sum()),
                             samples=len(norms), scope='all_observed_record_points_no_outcome_filter')
    return curves, delta, dict(fast=fast_info, effects=effects)


def main(root):
    if root.exists():
        raise FileExistsError(root)
    configure()
    start = time.monotonic()
    assert sha(REFERENCE/'completed.json') == REFERENCE_SHA
    assert sha(REFERENCE/'models/cold.pt') == COLD_SHA
    p = read(REFERENCE/'protocol.json')
    order = read(REFERENCE/'order.json')
    ids = [r['shot'] for r in order]
    assert len(ids) == len(set(ids)) == 360
    assert p['endpoint_policy'] == 'full_window' and p['target_loss'] == 'disruptive_positive_only'
    originals = read(REFERENCE/'receipts.json')
    raw_hashes = read(REFERENCE/'input_hashes.json')
    ledger = AccessLedger(ids, 20)
    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda()
    threshold = float(parent['threshold'])
    assert threshold == p['threshold']
    models = {name:build_adapter(base, readout='full', feedback='progressive', context=True)
              for name in ('cold', *POLICIES)}
    for model in models.values():
        assert load_context_memory(model, Path(p['source_memory']['path'])) == p['source_memory']
    assert sha(STATISTICS) == p['source_statistics_sha256']
    source_stats = read(STATISTICS)
    source, events = source_replay(models['cold'])
    assert events == read(REFERENCE/'source_replay.json')
    root.mkdir(parents=True)
    for name in ('shots','states','models'):
        (root/name).mkdir()
    files = set(Path(__file__).parent.glob('*.py')) | {PLAN}
    for name, module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module, '__file__', None):
            files.add(Path(module.__file__))
    protocol = dict(scope='DEVELOPMENT_COMPLETE360_INDEPENDENT_FORWARD_GUARD_POLICIES',
        original_parent=str(REFERENCE), original_completed_sha256=REFERENCE_SHA,
        cold_sha256=COLD_SHA, source_sha256=PARENT_SHA256, source_memory=p['source_memory'],
        contract=p['contract'], order_policy=p['order_policy'], order=order,
        threshold=threshold, lead_strictly_greater_ms=30, warmup=20, warmup_in_primary_score=True,
        policies=POLICIES, arms=ARMS, main_candidate='paired', tolerance=1e-7,
        steps_per_proposal=2, learning_rate=.001, gradient_clip=1.,
        target_loss=p['target_loss'], endpoint_policy=p['endpoint_policy'],
        half_rule='keep odd numbered own original-eligible proposals, no future budget access',
        half_control_exactly_acceptance_matched=False, all_proposal_four_losses_logged=True,
        shadows_do_not_train=True, paired_reverse_uses_same_persistent_state=True,
        cold_preparation_reused_only_after20=True, external_target_shots=0,
        plan_sha256=sha(PLAN), source_statistics_sha256=sha(STATISTICS), seed=131,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), independent_confirmation=False,
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files})
    _durable_json(root/'protocol.json', protocol)
    _durable_json(root/'source_replay.json', events)
    with zipfile.ZipFile(root/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))
    rows = {arm:[] for arm in ARMS}
    histories = {name:deque(maxlen=8) for name in POLICIES}
    eligible_counts = {name:0 for name in POLICIES}
    accepted_counts = {name:0 for name in POLICIES}
    pending, map_records, mapping = [], [], None
    matched_curves, matched_updates, receipts = 0, 0, []
    for index, sid in enumerate(ids):
        assert sha(io23.path_for('EAST', sid)) == raw_hashes[str(sid)]
        prefix = f'{index+1:04d}_{sid}'
        old_path = REFERENCE/'shots'/f'{prefix}.npz'
        assert sha(old_path) == read(REFERENCE/'shots'/f'{prefix}_seal.json')['sha256']
        with np.load(old_path, allow_pickle=False) as data:
            reference_curves = {arm:data[arm].copy() for arm in ('source', *set(MATCHED.values()))}
            old_times = data['times'].copy()
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        assert np.array_equal(raw['times'], old_times)
        signal = {k:raw[k] for k in ('values','available')}
        state_hash = None
        if index < 20:
            curves = {arm:reference_curves['source'].copy() for arm in ARMS}
            delta, info = None, dict(fast=dict(updates=0), effects={})
        else:
            signal = transform(signal, mapping)
            state_path = root/'states'/f'{prefix}.pt'
            torch.save({name:{key:value.cpu() for key,value in model.adapted_state().items()}
                        for name,model in models.items()}, state_path)
            state_hash = sha(state_path)
            curves, delta, info = predict(models, signal)
        for arm, previous in MATCHED.items():
            assert np.array_equal(curves[arm], reference_curves[previous]), (sid, arm, 'original drift')
            matched_curves += 1
        path = root/'shots'/f'{prefix}.npz'
        with path.open('xb') as f:
            np.savez_compressed(f, times=raw['times'], **curves)
            f.flush()
            os.fsync(f.fileno())
        digest = sha(path)
        ledger.seal(sid, digest)
        _durable_json(root/'shots'/f'{prefix}_seal.json', dict(shot=sid, index=index+1,
                      sha256=digest, state_sha256=state_hash, outcome_available=False))
        ledger.reveal(sid)
        outcome = simple_data.attach_outcome(raw, 'EAST', {})
        for arm in ARMS:
            valid = outcome['valid']
            rows[arm].append(dict(shot=sid, times=raw['times'][valid], scores=curves[arm][valid],
                                  is_disrupt=outcome['is_disrupt'], down=outcome['down']))
        receipt = dict(shot=sid, index=index+1, prediction_sha256=digest, state_sha256=state_hash,
                       info=info, updates={}, endpoints={})
        if index < 20:
            pending.append((outcome, reference_curves['source']))
            map_records.append({k:raw[k] for k in ('shot','device','values','available')})
        if index == 19:
            mapping = fit_prefix_map(source_stats, map_records, order=ids, count=20, policy='density_gain')
            assert mapping == read(REFERENCE/'input_map.json')
            _durable_json(root/'input_map.json', mapping)
            cold = torch.load(REFERENCE/'models/cold.pt', map_location='cuda', weights_only=True)
            for model in models.values():
                model.restore(cold)
            examples = []
            for shot, scores in pending:
                mapped = transform({k:shot[k] for k in ('values','available')}, mapping)
                d, _ = models['cold'].memory_delta(mapped)
                examples.append(target_example(models['cold'], shot, scores, mapped, d))
            for history in histories.values():
                history.extend(examples)
            _durable_json(root/'cold_reuse.json', dict(available_after_stream_rank=20,
                fit_ids=ids[:20], cold_sha256=COLD_SHA, no_new_cold_training=True,
                original_cold_replay_verified_in='slow_fast_attribution_v1'))
            pending, map_records = [], []
        elif index >= 20:
            for name in POLICIES:
                example = target_example(models[name], outcome, curves[name+'_on'], signal, delta)
                before_count = eligible_counts[name]
                result = update_with_guards(models[name], [example], list(histories[name]), source,
                    rule=name, eligible_so_far=before_count, policy=p['target_loss'], steps=2)
                if name == 'original':
                    assert {k:v for k,v in result.items() if k!='paired_guard'} == originals[index]['updates']['both']
                    matched_updates += 1
                eligible_counts[name] = result['paired_guard']['eligible_count_after']
                accepted_counts[name] += int(result['accepted'])
                receipt['updates'][name] = result
                receipt['endpoints'][name] = example['ends']
                histories[name].append(example)
        receipts.append(receipt)
        _durable_json(root/'shots'/f'{prefix}_audit.json', receipt)
        if (index+1) % 20 == 0:
            progress = dict(shots=index+1, total=360, seconds=time.monotonic()-start,
                            original_curves_exact=matched_curves, original_updates_exact=matched_updates,
                            accepted=accepted_counts, eligible=eligible_counts)
            _durable_json(root/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    assert len(ledger.history) == 360 and sha(DEFAULT_PARENT) == PARENT_SHA256
    original_final = torch.load(REFERENCE/'models/both.pt', map_location='cuda', weights_only=True)
    assert all(torch.equal(value, original_final[key]) for key,value in models['original'].adapted_state().items())
    for name, model in models.items():
        torch.save({k:v.cpu() for k,v in model.adapted_state().items()}, root/'models'/f'{name}.pt')
    results, per_shot = {}, {}
    for arm, values in rows.items():
        per_shot[arm] = score_fixed_test('EAST', values, threshold)['predictions']
        results[arm] = dict(full360=compact(values,threshold), warmup20=compact(values[:20],threshold),
                           suffix340=compact(values[20:],threshold),
                           blocks20=[compact(values[i:i+20],threshold) for i in range(0,360,20)])
    _durable_json(root/'per_shot.json', per_shot)
    _durable_json(root/'receipts.json', receipts)
    done = dict(status='COMPLETED', seconds=time.monotonic()-start, metrics=results,
                accepted=accepted_counts, original_eligible=eligible_counts,
                original_reference_curves_exact=matched_curves, original_update_receipts_exact=matched_updates,
                original_final_state_exact=True, source_unchanged=True,
                independent_confirmation=False, scientific_goal_achieved=False)
    _durable_json(root/'completed.json', done)
    print(json.dumps({arm:value['full360'] for arm,value in results.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
