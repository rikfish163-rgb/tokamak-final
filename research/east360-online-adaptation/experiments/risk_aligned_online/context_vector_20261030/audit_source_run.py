"""Independent saved-array recount; optional full source-VAL neural replay.

This audits a component experiment, never a full online stream or shot TPR.
It does not invoke the training runner's summary or re-select a checkpoint.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .. import io23
from ..simple_online import _durable_json
from ..simple_train import sha

ARMS = ('real', 'reverse', 'mean_feedback', 'constant', 'risk_axis')
GROUPS = ('safe', 'danger', 'early')


def recount(records, arrays):
    n = len(records)
    for key, value in arrays.items():
        if np.shape(value) != (n,):
            raise ValueError('prediction shape mismatch: '+key)
        if not np.isfinite(value).all():
            raise ValueError('prediction must be finite: '+key)
    label = np.asarray([r['label'] for r in records], dtype=np.float64)
    parent = np.asarray(arrays['baseline'], dtype=np.float64)
    parent_loss = np.logaddexp(0., parent)-parent*label
    buckets = defaultdict(list)
    for i, record in enumerate(records):
        buckets[record['group'], record['shot']].append(i)
    rows, groups = [], {}
    for arm in ARMS:
        z = np.asarray(arrays[arm+'_logit'], dtype=np.float64)
        losses = np.logaddexp(0., z)-z*label
        groups[arm] = {}
        for group in GROUPS:
            local = []
            for (kind, sid), take in sorted(buckets.items()):
                if kind != group:
                    continue
                row = dict(arm=arm, group=group, shot=int(sid), windows=len(take),
                    mean_before_bce=float(np.mean(parent_loss[take])),
                    mean_after_bce=float(np.mean(losses[take])),
                    mean_delta_bce=float(np.mean(losses[take]-parent_loss[take])),
                    mean_logit_change=float(np.mean(z[take]-parent[take])),
                    mean_action_norm=float(np.mean(arrays[arm+'_norm'][take])),
                    mean_cos_real=float(np.mean(arrays[arm+'_cos_real'][take])))
                rows.append(row)
                local.append(row)
            delta = np.asarray([r['mean_delta_bce'] for r in local])
            groups[arm][group] = dict(shots=len(local), windows=sum(r['windows'] for r in local),
                mean_delta_bce=float(delta.mean()) if len(local) else None,
                improved_shots=int((delta < -1e-6).sum()), worsened_shots=int((delta > 1e-6).sum()),
                mean_logit_change=float(np.mean([r['mean_logit_change'] for r in local])) if local else None)
    reasons = []
    if min(groups['real'][g]['shots'] for g in ('safe', 'danger')) < 10:
        reasons.append('insufficient_validation_shots')
    else:
        for group in ('safe', 'danger'):
            real = groups['real'][group]['mean_delta_bce']
            if real >= 0:
                reasons.append(group+': real does not improve baseline')
            for arm in ('reverse', 'mean_feedback'):
                if real >= groups[arm][group]['mean_delta_bce']:
                    reasons.append(group+': real does not outperform '+arm)
    return dict(groups=groups, per_shot=rows, reasons=reasons, source_screen_passed=not reasons)


def descriptive_intervals(rows):
    """Shot-resampled descriptive intervals; no IID window or confirmatory claim."""
    rng = np.random.default_rng(131)
    result = {}
    for group in GROUPS:
        subset = [r for r in rows if r['group'] == group]
        ids = sorted({r['shot'] for r in subset})
        if not ids:
            continue
        values = {(r['arm'], r['shot']): r['mean_delta_bce'] for r in subset}
        take = rng.integers(0, len(ids), (10000, len(ids)))
        result[group] = {}
        for arm in ('baseline', 'reverse', 'mean_feedback', 'constant', 'risk_axis'):
            diff = np.array([values['real', sid] - (0. if arm == 'baseline' else values[arm, sid]) for sid in ids])
            result[group][arm] = dict(shots=len(ids), real_minus_control=float(diff.mean()),
                percentile95=np.quantile(diff[take].mean(1), [.025, .975]).tolist())
    return dict(method='10000 paired shot bootstrap draws; seed131; unadjusted descriptive intervals',
                not_confirmatory=True, between_seed_variation_not_measured=True, groups=result)


def read(path):
    return json.loads(path.read_text())


def verify(directory):
    protocol, complete = read(directory/'protocol.json'), read(directory/'complete.json')
    assert complete['complete'] and complete['source_unchanged']
    for name, digest in complete['artifact_sha256'].items():
        assert sha(directory/name) == digest, name
    for name, digest in protocol['code_sha256'].items():
        assert sha(io23.ROOT/name) == digest, name
    roles = io23.source_roles()
    assert protocol['train_ids'] == roles['train'] and protocol['val_ids'] == roles['val']
    train_records = read(directory/'train_records.json')
    val_records = read(directory/'val_records.json')
    for role, records in [('train', train_records), ('val', val_records)]:
        hashes = read(directory/f'{role}_input_hashes.json')
        assert {int(sid) for sid in hashes} == set(roles[role])
        for sid, digest in hashes.items():
            assert sha(io23.path_for('J-TEXT', int(sid))) == digest, sid
        population = read(directory/f'{role}_population.json')
        assert [r['shot'] for r in population['shots']] == roles[role]
        for group in GROUPS:
            selected = [r for r in records if r['group'] == group]
            assert population['groups'][group] == dict(shots=len({r['shot'] for r in selected}), windows=len(selected))
        counts = defaultdict(int)
        identities = set()
        for r in records:
            assert r['shot'] in roles[role]
            assert r['origin'] >= 127 and (r['origin']-127) % 25 == 0
            assert r['origin']+25 == r['maturity'] < r['query'] <= r['origin']+50
            assert not r['disruptive'] or r['lead_ms'] > 10.
            assert r['feedback_observed'] and r['label'] == float(r['group'] == 'danger')
            assert (r['shot'], r['origin']) not in identities
            identities.add((r['shot'], r['origin']))
            counts[r['shot'], r['group']] += 1
        for row in population['shots']:
            assert sum(row['before'].values())+sum(row['excluded'].values()) == row['origins']
            for group in GROUPS:
                assert row['after'][group] == min(8, row['before'][group]) == counts[row['shot'], group]
    history = read(directory/'training.json')
    assert len(history['history']) == len(history['sampled_indices']) == 512
    for indices in history['sampled_indices']:
        assert len(indices) == 12
        assert [train_records[i]['group'] for i in indices] == ['safe']*6+['danger']*6
    with np.load(directory/'validation_arrays.npz', allow_pickle=False) as handle:
        arrays = dict(handle)
    recalculated = recount(val_records, arrays)
    recorded = read(directory/'validation.json')
    assert recalculated['groups'] == recorded['groups']
    def by_id(rows):
        return {(r['arm'], r['group'], r['shot']): r for r in rows}
    assert by_id(recalculated['per_shot']) == by_id(recorded['per_shot'])
    assert recalculated['reasons'] == recorded['reasons']
    assert recalculated['source_screen_passed'] == complete['source_screen_passed']
    return dict(verified=True, raw_inputs=len(roles['train'])+len(roles['val']),
        validation_windows=len(val_records), per_shot_rows=len(recalculated['per_shot']),
        source_screen_passed=recalculated['source_screen_passed'], reasons=recalculated['reasons'],
        component_not_stream=True, inference_replayed=False,
        descriptive=descriptive_intervals(recalculated['per_shot']))


def neural_replay(directory):
    import torch
    from .. import simple_data
    from ..attention_basis_20261029.learning import MatrixAttention
    from ..delayed_recurrent_memory_20261025.learning import batch
    from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
    from .learning import ContextTarget
    from .observation import observation
    from .prepare import evaluate
    from .source_run import source_gate

    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent, _ = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(parent.cuda()).eval().requires_grad_(False)
    original = {k: v.detach().clone() for k, v in net.state_dict().items()}
    policy = torch.load(directory/'controller.pt', weights_only=True, map_location='cuda')
    controller = ContextTarget().cuda().eval().requires_grad_(False)
    controller.load_state_dict(policy['state'])
    records = read(directory/'val_records.json')
    roles, advanced = io23.source_roles(), io23.advanced_times()
    by_shot = defaultdict(list)
    for r in records:
        by_shot[r['shot']].append(r)
    chunks = []
    with source_gate(roles['val']):
        for sid in roles['val']:
            shot = simple_data.attach_outcome(simple_data.read_shot('J-TEXT', sid), 'J-TEXT', advanced)
            signal = {k: shot[k] for k in ('values', 'available')}
            for r in by_shot[sid]:
                lead = round((shot['down']-shot['times'][r['query']])*1000., 9) if shot['is_disrupt'] else None
                assert lead == r['lead_ms'] and shot['y'][r['query']] == r['label']
                assert shot['valid'][r['query']]
                obs = observation(net, signal, r['origin'])
                x, mask = batch(signal, [r['query']], 'cuda')
                obs.pop('reference')
                obs.update(later_x=x, later_mask=mask)
                chunks.append(obs)
    data = {k: torch.cat([c[k] for c in chunks]) for k in chunks[0]}
    actual = evaluate(net, None, controller, data, policy['mean_context'])
    diffs = {}
    with np.load(directory/'validation_arrays.npz', allow_pickle=False) as expected:
        assert set(actual) == set(expected.files)
        for name, value in actual.items():
            diffs[name] = float(np.max(np.abs(value-expected[name])))
            assert np.array_equal(value, expected[name]), (name, diffs[name])
    assert all(torch.equal(net.state_dict()[k], v) for k, v in original.items())
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    return dict(verified=True, windows=len(records), groups='all saved validation groups',
                exact_array_match=True, max_abs_differences=diffs,
                source_unchanged=True, training_not_repeated=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = neural_replay(args.run) if args.replay else verify(args.run)
    _durable_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
