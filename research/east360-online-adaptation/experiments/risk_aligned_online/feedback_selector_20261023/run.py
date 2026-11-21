"""Learn one feedback selector after20 ended shots; freeze it for suffix340."""
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
from ..lag_bias_feedback_20261022.run import MAP_PATH
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha, metrics
from ..warmstart_alignment_20261022.alignment import fit_prefix_map, transform
from ..warmstart_alignment_20261022.run import public, snapshot
from .execution import collect_shot, predict_policy
from .policy import fit_policy

REFERENCE = io23.ROOT / 'experiments/risk_aligned_online/runs/lag_bias_feedback_test360_20261022'
ARMS = ('static', 'recent', 'lag_bias', 'contextual', 'mean_context')


def validate_prefix_seals(root, seals, order):
    if len(seals) != 20 or [r['shot'] for r in seals] != list(order[:20]):
        raise ValueError('all first20 must be sealed before any training outcome access')
    for r in seals:
        if sha(root / 'shots' / f"{r['shot']}.npz") != r['prediction_sha256']:
            raise ValueError('prefix prediction changed before label reveal')


def load_signal(sid):
    return apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))


def prepare_policy(model, prefix, mapping, root, seals, order):
    validate_prefix_seals(root, seals, order)
    if [s['shot'] for s in prefix] != order[:20]:
        raise ValueError('exact original prefix required')
    _durable_json(root / 'label_access_gate.json', dict(permitted_ids=order[:20],
        after_all20_sealed=seals, action='reveal completed prefix only'))
    rows = []; exclusions = []; outcomes = []
    for s in prefix:
        outcome = simple_data.attach_outcome(s, 'EAST', {})
        signal = transform({k: s[k] for k in ('values', 'available')}, mapping)
        collected, excluded = collect_shot(model, signal, outcome, s['shot'], device='cuda')
        rows.extend(collected); exclusions.extend(excluded)
        outcomes.append(dict(shot=s['shot'], is_disrupt=outcome['is_disrupt'], down=outcome['down']))
        print(json.dumps(dict(stage='prefix_counterfactual', shot=s['shot'],
                              n_rows=len(collected))), flush=True)
    _durable_json(root / 'training_rows.json', rows)
    _durable_json(root / 'training_exclusions.json', exclusions)
    _durable_json(root / 'prefix_outcomes.json', outcomes)
    policy = fit_policy(rows, allowed_ids=order[:20])
    _durable_json(root / 'policy.json', policy)
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    root = parser.parse_args().output
    if root.exists(): raise FileExistsError(root)
    if not torch.cuda.is_available(): raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2); torch.manual_seed(131)
    start = time.monotonic()
    base, parent = load_parent(DEFAULT_PARENT)
    model = PairedFeedback(base, np.ones(23)).cuda(); model.set_stage('frozen')
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    contract = load_test_contract('EAST'); order = contract['shot_ids']
    threshold = float(parent['threshold'])
    ref_protocol = json.loads((REFERENCE / 'protocol.json').read_text())
    assert json.loads((REFERENCE / 'verification.json').read_text())['status'] == 'VERIFIED'
    assert threshold == ref_protocol['threshold'] and order == ref_protocol['contract']['shot_ids']
    assert ref_protocol['source_sha256'] == PARENT_SHA256
    assert ref_protocol['density_mapping_sha256'] == sha(MAP_PATH)
    mapping = json.loads(MAP_PATH.read_text())
    assert mapping['calibration_ids'] == order[:20]
    assert mapping['source_statistics']['shot_ids'] == io23.source_roles()['train']
    assert mapping['first_affected_index'] == 20
    old_seals = json.loads((REFERENCE / 'prediction_receipts.json').read_text())
    assert [r['shot'] for r in old_seals] == order
    # Verify the mapping's original prefix seal chain, not just its final hash.
    assert [r['shot'] for r in mapping['preceding_prediction_seals']] == order[:20]
    for r in mapping['preceding_prediction_seals']:
        assert sha(MAP_PATH.parent / 'shots' / f"{r['shot']}.npz") == r['prediction_sha256']
    root.mkdir(parents=True); (root / 'shots').mkdir()
    hashes = snapshot(root)
    dependencies = list(Path(__file__).parent.glob('*.py')) + [
        Path(__file__).parents[1] / n for n in (
            'deepcoral_prefix_20261022/learning.py', 'lag_bias_feedback_20261022/learning.py',
            'lag_bias_feedback_20261022/run.py', 'tests/test_feedback_selector.py')]
    with zipfile.ZipFile(root / 'code_snapshot.zip', 'a', zipfile.ZIP_DEFLATED) as archive:
        for f in dependencies:
            relative = str(f.relative_to(io23.ROOT)); archive.write(f, relative); hashes[relative] = sha(f)
    plan = io23.ROOT / 'docs/research/source_event_rebuild_20261022/反馈选择器_20炮准备_事前约定_20261023.md'
    _durable_json(root / 'protocol.json', dict(contract=contract, threshold=threshold, arms=ARMS,
        source_checkpoint=str(DEFAULT_PARENT), source_sha256=PARENT_SHA256,
        reference=str(REFERENCE), reference_protocol_sha256=sha(REFERENCE / 'protocol.json'),
        reference_receipts_sha256=sha(REFERENCE / 'prediction_receipts.json'),
        mapping_path=str(MAP_PATH), mapping_sha256=sha(MAP_PATH),
        plan_path=str(plan), plan_sha256=sha(plan), code_sha256=hashes,
        parameter_scope=model.encoder_names(), prefix_outcome_budget=20,
        update='same archived temporary SGD0.001 cap1 W128/H25 interval128',
        policy='linear ridge .01; fixed after20; class/shot/trajectory equal weighting',
        proxy='BCE of next128 peak logit; safe all valid, disruptive only strict lead>30',
        first20='archived raw predictions retained; mapped counterfactuals only for training',
        archived_control_arms=['static', 'recent', 'lag_bias'],
        suffix_outcomes_read_only_after_all360_sealed=True,
        no_cross_shot_encoder_persistence=True, independent_confirmation=False))
    seals = []; prefix = []; policy = None; policy_hash = None
    for index, (sid, old_seal) in enumerate(zip(order, old_seals)):
        s = load_signal(sid); refpath = REFERENCE / 'shots' / f'{sid}.npz'
        assert sha(refpath) == old_seal['prediction_sha256']
        with np.load(refpath, allow_pickle=False) as ref:
            np.testing.assert_array_equal(s['times'], ref['times'])
            predictions = {k: ref[k].copy() for k in ARMS[:3]}
        if index < 20:
            predictions.update({k: predictions['static'].copy() for k in ARMS[3:]})
            records = {}
            prefix.append(s)
        else:
            assert sha(root / 'policy.json') == policy_hash
            signal = transform({k: s[k] for k in ('values', 'available')}, mapping)
            new, records = predict_policy(model, signal, policy)
            predictions.update(new)
            for k in ARMS[3:]:
                np.testing.assert_array_equal(predictions[k][:256], predictions['static'][:256])
        path = root / 'shots' / f'{sid}.npz'; updatepath = root / 'shots' / f'{sid}_updates.json'
        np.savez_compressed(path, times=s['times'], **predictions)
        _durable_json(updatepath, records)
        seals.append(dict(shot=sid, prediction_sha256=sha(path), updates_sha256=sha(updatepath)))
        if index == 19:
            validate_prefix_seals(root, seals, order)
            signals = [{k: s[k] for k in ('shot', 'device', 'values', 'available')} for s in prefix]
            rebuilt = fit_prefix_map(mapping['source_statistics'], signals,
                                     order=order, count=20, policy='density_gain')
            assert rebuilt == {k: v for k, v in mapping.items() if k != 'preceding_prediction_seals'}
            policy = prepare_policy(model, prefix, mapping, root, seals.copy(), order)
            policy_hash = sha(root / 'policy.json')
            _durable_json(root / 'activation_receipt.json', dict(after_sealed_ids=order[:20],
                preceding_seals=seals.copy(), mapping_recomputed_from_prefix=True,
                policy_sha256=policy_hash, training_rows_sha256=sha(root / 'training_rows.json')))
            prefix = []
        if (index + 1) % 40 == 0:
            progress = dict(stage='stream', shots=index + 1, seconds=time.monotonic() - start)
            _durable_json(root / 'progress.json', progress); print(json.dumps(progress), flush=True)
    _durable_json(root / 'prediction_receipts.json', seals)
    rows = {k: [] for k in ARMS}
    # Future340 labels have no access path in model/context/selection above.
    for sid in order:
        s = simple_data.attach_outcome(load_signal(sid), 'EAST', {})
        with np.load(root / 'shots' / f'{sid}.npz', allow_pickle=False) as trace:
            for name in ARMS:
                v = s['valid']
                rows[name].append(dict(shot=sid, times=s['times'][v], scores=trace[name][v],
                                      is_disrupt=s['is_disrupt'], down=s['down']))
    full = {k: score_fixed_test('EAST', v, threshold) for k, v in rows.items()}
    result = {k: dict(full360=public(full[k]),
                     common_suffix340=public(metrics(v[20:], threshold, lead=30.))) for k, v in rows.items()}
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v.cpu(), frozen[k], rtol=0, atol=0)
    assert sha(DEFAULT_PARENT) == PARENT_SHA256 and sha(root / 'policy.json') == policy_hash
    _durable_json(root / 'per_shot.json', full)
    _durable_json(root / 'completed.json', dict(status='COMPLETED', metrics=result,
        persistent_model_bitwise_unchanged=True, policy_frozen_after20=True,
        seconds=time.monotonic() - start))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
