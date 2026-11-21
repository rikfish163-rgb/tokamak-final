"""Rescore every trace; rebuild frozen policy and selected full-shot replays."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23, simple_data
from ..deepcoral_prefix_20261022.audit import paired_flips
from ..fixed_test_protocol import load_test_contract, score_fixed_test
from ..lag_bias_feedback_20261022.audit import independent_counts
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha, metrics
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .execution import predict_policy, collect_shot, eligible_query
from .policy import fit_policy, choose_action, ACTIONS
from .run import ARMS, REFERENCE, load_signal, validate_prefix_seals


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('directory', type=Path)
    root = p.parse_args().directory
    done = json.loads((root / 'completed.json').read_text()); assert done['status'] == 'COMPLETED'
    protocol = json.loads((root / 'protocol.json').read_text())
    order = load_test_contract('EAST')['shot_ids']; assert protocol['contract']['shot_ids'] == order
    assert protocol['arms'] == list(ARMS)
    for path, digest in protocol['code_sha256'].items(): assert sha(io23.ROOT / path) == digest
    assert sha(protocol['plan_path']) == protocol['plan_sha256']
    assert sha(protocol['mapping_path']) == protocol['mapping_sha256']
    assert sha(REFERENCE / 'protocol.json') == protocol['reference_protocol_sha256']
    assert sha(REFERENCE / 'prediction_receipts.json') == protocol['reference_receipts_sha256']
    mapping = json.loads(Path(protocol['mapping_path']).read_text())
    seals = json.loads((root / 'prediction_receipts.json').read_text())
    assert [s['shot'] for s in seals] == order
    validate_prefix_seals(root, seals[:20], order)
    gate = json.loads((root / 'label_access_gate.json').read_text())
    activation = json.loads((root / 'activation_receipt.json').read_text())
    assert gate['permitted_ids'] == activation['after_sealed_ids'] == order[:20]
    assert gate['after_all20_sealed'] == activation['preceding_seals'] == seals[:20]
    assert sha(root / 'training_rows.json') == activation['training_rows_sha256']
    assert sha(root / 'policy.json') == activation['policy_sha256']
    train = json.loads((root / 'training_rows.json').read_text())
    policy = json.loads((root / 'policy.json').read_text())
    recomputed = fit_policy(train, allowed_ids=order[:20])
    assert recomputed == policy
    prefix_outcomes = {r['shot']: r for r in json.loads((root / 'prefix_outcomes.json').read_text())}
    assert list(prefix_outcomes) == order[:20]
    contexts = np.asarray([r['context'] for r in train]); weights = np.asarray(policy['weights'])
    observed_classes = Counter()
    for sid in order[:20]:
        s = simple_data.attach_outcome(load_signal(sid), 'EAST', {})
        assert prefix_outcomes[sid]['is_disrupt'] == s['is_disrupt']
        assert prefix_outcomes[sid]['down'] == s['down']
        observed_classes[str(s['is_disrupt'])] += 1
        for r in (r for r in train if r['shot'] == sid):
            assert r['is_disrupt'] == s['is_disrupt']
            count = r['count']; stop = min(count + 128, len(s['values']))
            selected = eligible_query(s, count, stop)
            assert int(selected.sum()) == r['eligible_query_points'] > 0
            for a in ACTIONS:
                assert count <= r['peak_indices'][a] < stop and selected[r['peak_indices'][a] - count]
            np.testing.assert_array_equal(r['utility'], [r['losses'][a] - r['losses']['hold'] for a in ACTIONS[1:]])
    assert dict(observed_classes) == policy['class_counts']
    full = json.loads((root / 'per_shot.json').read_text())
    rows = {a: [] for a in ARMS}; summaries = {a: [] for a in ARMS[3:]}
    old_seals = json.loads((REFERENCE / 'prediction_receipts.json').read_text())
    for index, (sid, seal, old_seal) in enumerate(zip(order, seals, old_seals)):
        path = root / 'shots' / f'{sid}.npz'; upath = root / 'shots' / f'{sid}_updates.json'
        assert sha(path) == seal['prediction_sha256'] and sha(upath) == seal['updates_sha256']
        refpath = REFERENCE / 'shots' / f'{sid}.npz'
        assert sha(refpath) == old_seal['prediction_sha256']
        s = simple_data.attach_outcome(load_signal(sid), 'EAST', {})
        with np.load(path, allow_pickle=False) as trace, np.load(refpath, allow_pickle=False) as ref:
            np.testing.assert_array_equal(trace['times'], s['times'])
            for a in ARMS[:3]: np.testing.assert_array_equal(trace[a], ref[a])
            for a in ARMS:
                assert trace[a].shape == s['times'].shape and np.isfinite(trace[a]).all()
                assert np.all((trace[a] >= 0) & (trace[a] <= 1))
                if index < 20: np.testing.assert_array_equal(trace[a], trace['static'])
                np.testing.assert_array_equal(trace[a][:256], trace['static'][:256])
                v = s['valid']
                rows[a].append(dict(shot=sid, times=s['times'][v], scores=trace[a][v].copy(),
                                    is_disrupt=s['is_disrupt'], down=s['down']))
            updates = json.loads(upath.read_text())
            if index < 20:
                assert updates == {}; continue
            assert set(updates) == set(ARMS[3:])
            for a, records in updates.items():
                assert [r['first_affected_prediction'] for r in records] == list(range(256, len(s['values']), 128))
                for r in records:
                    count = r['first_affected_prediction']; assert r['last_observation'] == count - 1
                    action, utility = choose_action(policy, r['context'], mean_context=a == 'mean_context')
                    assert r['action'] == action and r['predicted_utility'] == utility
                    assert r['gradient_steps'] in (0, 1) and 0 <= r['parameter_delta_norm'] <= .00101
                    if action == 'hold': assert r['gradient_steps'] == 0 and r['parameter_delta_norm'] == 0
                    elif r['gradient_steps']:
                        assert r['calibration_origins'] == [count-101, count-76]
                        assert r['gradient_origins'] == [count-51, count-26]
                        assert r['observed_gradient_targets'] > 0
                    else:
                        assert r['reason'] == 'no_shared_observed_targets'
                    prior_alarm = bool(np.any(trace[a][:count][s['valid'][:count]] >= protocol['threshold']))
                    summaries[a].append(dict(r, shot=sid, already_alarm=prior_alarm))
    counts = {}
    for a, rs in rows.items():
        assert score_fixed_test('EAST', rs, protocol['threshold']) == full[a]
        assert public(full[a]) == done['metrics'][a]['full360']
        assert public(metrics(rs[20:], protocol['threshold'], lead=30.)) == done['metrics'][a]['common_suffix340']
        counts[a] = independent_counts(rs, protocol['threshold'])
        for k in ('tp', 'fp', 'n_disrupt', 'n_safe'): assert counts[a][k] == full[a][k]
    summary = {}
    for a, records in summaries.items():
        summary[a] = dict(events=len(records), gradient_steps=sum(r['gradient_steps'] for r in records),
            actions=dict(Counter(r['action'] for r in records)),
            actions_before_alarm=dict(Counter(r['action'] for r in records if not r['already_alarm'])),
            actions_after_alarm=dict(Counter(r['action'] for r in records if r['already_alarm'])),
            steps_loss_down=sum(r['gradient_steps'] and r['loss_after'] < r['loss_before'] for r in records),
            steps_loss_up=sum(r['gradient_steps'] and r['loss_after'] > r['loss_before'] for r in records),
            context_any_over5sd=sum(r['context_components_over5sd'] > 0 for r in records),
            context_max_standardized=float(max(r['scaled_context_abs_max'] for r in records)))
    torch.set_num_threads(2); torch.manual_seed(131)
    base, _ = load_parent(DEFAULT_PARENT); model = PairedFeedback(base, np.ones(23)).cuda(); model.set_stage('frozen')
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    replayed = []; train_replayed = []
    # Predeclared positional selection: no choosing improved or failed shots.
    for index in (0, 9, 19):
        sid = order[index]; s = load_signal(sid)
        signal = transform({k: s[k] for k in ('values', 'available')}, mapping)
        outcome = simple_data.attach_outcome(s, 'EAST', {})
        actual, _ = collect_shot(model, signal, outcome, sid, device='cuda')
        assert actual == [r for r in train if r['shot'] == sid]
        train_replayed.append(sid)
    for index in (20, 179, 359):
        sid = order[index]; s = load_signal(sid)
        signal = transform({k: s[k] for k in ('values', 'available')}, mapping)
        actual, updates = predict_policy(model, signal, policy)
        with np.load(root / 'shots' / f'{sid}.npz', allow_pickle=False) as saved:
            for a, v in actual.items(): np.testing.assert_array_equal(v, saved[a])
        assert updates == json.loads((root / 'shots' / f'{sid}_updates.json').read_text())
        replayed.append(sid)
    for k, v in model.state_dict().items(): torch.testing.assert_close(v.cpu(), frozen[k], rtol=0, atol=0)
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    comparisons = {f'{base}_to_{a}': paired_flips(full[base]['predictions'], full[a]['predictions'])
                   for base in ('static', 'recent', 'lag_bias', 'mean_context') for a in ('contextual',)
                   if base != a}
    report = dict(status='VERIFIED', rescored_traces=len(order)*len(ARMS),
        first20_cost_unchanged=True, policy_exactly_refitted=True,
        prefix_counterfactual_shots_replayed=train_replayed, deployed_shots_replayed=replayed,
        deployed_full_traces_replayed=len(replayed)*2, max_replay_error=0.,
        source_archive_unchanged=True, independent_counts=counts, update_summary=summary,
        paired_comparisons=comparisons, training_rows=len(train), training_shots=len(order[:20]),
        training_class_counts=dict(observed_classes),
        training_context_rank=int(np.linalg.matrix_rank((contexts-policy['mean']) / policy['scale'])),
        training_class_weight={str(k):float(sum(w for r,w in zip(train,weights) if r['is_disrupt']==k))
                               for k in (False,True)},
        scope='development experiment; no significance/independent-confirmation claim',
        audit_code_sha256=sha(Path(__file__)))
    _durable_json(root / 'verification.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
