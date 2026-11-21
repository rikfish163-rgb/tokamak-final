"""Recount all five arms and replay the sealed readout/scope experiment.

The warning counter is independent of the runner's metric routine. Neural
replay uses the same implementation; it is not independent replication.
"""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..horizon_memory_20261023.audit import independently_score, aggregate
from ..horizon_memory_20261023.learning import FoldAccess
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent, load_source_train, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import ARMS, PatchForecast
from .run import prepare

READ = lambda p: json.loads(p.read_text())


def contrasts(stats):
    if set(stats) != {'static', *ARMS} or any(
        r['n_disrupt'] != 4 or r['n_safe'] != 16 for r in stats.values()
    ):
        raise ValueError('five arms on four disruptive and sixteen safe shots required')

    def delta(after, before):
        return {f'delta_{key}': after[key] - before[key] for key in ('tp', 'fp', 'J')}

    base = stats['static']
    vs_static = {}
    for arm in ARMS:
        r = stats[arm]
        user = r['tp'] >= base['tp'] and r['J'] > base['J'] and r['fpr'] <= .20
        vs_static[arm] = dict(**delta(r, base), user_target_pass=bool(user),
                             strict_screen_pass=bool(user and r['fp'] <= base['fp']))
    return dict(
        vs_static=vs_static,
        readout_effect={s: delta(stats[f'patch_{s}'], stats[f'full_{s}'])
                        for s in ('all', 'observation')},
        scope_effect={h: delta(stats[f'{h}_observation'], stats[f'{h}_all'])
                      for h in ('full', 'patch')},
    )


def check_draws(folder, reference, source, target):
    groups = {'source': {r['shot']: r for r in source},
              'target': {r['shot']: r for r in target}}
    count = 0
    for head in ('full', 'patch'):
        path = folder / f'{head}_manifest.jsonl'
        assert path.read_bytes() == (reference / 'forecast_manifest.jsonl').read_bytes()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 256
        for step, row in enumerate(rows):
            assert row['step'] == step
            for domain, records in groups.items():
                assert len(row[domain]) == 32
                for item in row[domain]:
                    signal = records[item['shot']]
                    assert 127 <= item['end'] < len(signal['values']) - 25
                    count += 1
    return count


def audit(root):
    if (root / 'verification.json').exists():
        raise FileExistsError(root / 'verification.json')
    p, done = READ(root / 'protocol.json'), READ(root / 'completed.json')
    assert done['status'] == 'COMPLETED' and not done['test360_accessed']
    for name, digest in done['artifact_sha256'].items():
        assert sha(root / name) == digest, name
    for name, digest in p['code_sha256'].items():
        assert sha(io23.ROOT / name) == digest, name
    assert sha(Path(p['plan_path'])) == p['plan_sha256']
    assert sha(DEFAULT_PARENT) == PARENT_SHA256 == p['source_sha256']
    reference = Path(p['reference'])
    assert sha(reference / 'completed.json') == p['reference_sha256']
    ref_hashes = READ(reference / 'completed.json')['artifact_sha256']
    held = [s for f in p['folds'] for s in f['held']]
    assert len(held) == len(set(held)) == 20
    assert all(set(f['fit']) == set(held) - set(f['held']) for f in p['folds'])

    torch.set_num_threads(2)
    torch.manual_seed(131)
    base, info = load_parent(DEFAULT_PARENT)
    parent = PairedFeedback(base, np.ones(23)).cuda()
    parent.set_stage('frozen')
    original = state_cpu(parent)
    assert float(info['threshold']) == p['threshold']
    source_ids = io23.source_roles()['train']
    read_signal, read_outcome = simple_data.read_shot, io23.read_outcome

    def source_guard(device, sid):
        assert device == 'J-TEXT' and sid in source_ids
        return read_outcome(device, sid)

    with patch.object(io23, 'read_outcome', side_effect=source_guard):
        source, _ = load_source_train()
    assert READ(root / 'source_label_access.json') == source_ids
    seals = {(r['fold'], r['shot']): r for r in READ(root / 'prediction_receipts.json')}
    rows = {a: [] for a in ('static',) + ARMS}
    details, replay, counts = [], [], {}
    update_stats = {a: dict(steps=0, loss_decreased=0, max_summary_step=0., max_step=0.) for a in ARMS}

    # Prediction and update replay complete before this audit reads held labels.
    from .learning import predict
    for plan in p['folds']:
        f = plan['fold']
        folder, ref = root / f'fold{f}', reference / f'fold{f}' / 'legacy'
        for name in ('parent.pt', 'input_map.json', 'forecast_manifest.jsonl', 'forecast_training.json'):
            path = ref / name
            assert sha(path) == ref_hashes[str(path.relative_to(reference))]
        fit = copy.deepcopy(parent)
        fit.load_state_dict(torch.load(ref / 'parent.pt', map_location='cuda', weights_only=True)['model'])
        fit.set_stage('frozen')
        origin = state_cpu(fit)
        models = {}
        for h in ('full', 'patch'):
            model = copy.deepcopy(fit)
            if h == 'patch':
                model.base.forecast_head = PatchForecast(model.base.forecast_head, model.cfg.dim)
            model.load_state_dict(torch.load(folder / f'{h}.pt', map_location='cuda', weights_only=True)['model'])
            model.set_stage('frozen')
            allowed = () if h == 'full' else ('base.forecast_head.weight', 'base.forecast_head.bias')
            assert_unchanged(model, origin, allowed)
            if h == 'patch':
                torch.testing.assert_close(model.base.forecast_head.weight[:, -model.cfg.dim:],
                    parent.base.forecast_head.weight[:, -model.cfg.dim:], rtol=0, atol=0)
            models[h] = model
        frozen = {h: state_cpu(m) for h, m in models.items()}
        mapping = READ(folder / 'input_map.json')
        assert mapping == READ(ref / 'input_map.json')
        access = FoldAccess(plan, source_ids)

        def signal_guard(device, sid, **kwargs):
            access.check('signal', device, sid)
            return read_signal(device, sid, **kwargs)

        def forbid_outcome(device, sid):
            raise AssertionError(f'no outcome needed during forecast preparation/replay: {device}/{sid}')

        with patch.object(simple_data, 'read_shot', side_effect=signal_guard), patch.object(io23, 'read_outcome', side_effect=forbid_outcome):
            records = [apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal')) for sid in plan['fit']]
            target = [dict(r, **transform({k: r[k] for k in ('values', 'available')}, mapping)) for r in records]
            counts[f] = check_draws(folder, ref, source, target)
            access.phase = 'predict'
            for sid in plan['held']:
                s = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
                scores, updates = predict(models, transform({k: s[k] for k in ('values', 'available')}, mapping))
                path, up = folder / 'shots' / f'{sid}.npz', folder / 'shots' / f'{sid}_updates.json'
                assert sha(path) == seals[f, sid]['prediction_sha256']
                assert sha(up) == seals[f, sid]['updates_sha256'] and updates == READ(up)
                with np.load(path, allow_pickle=False) as trace:
                    np.testing.assert_array_equal(trace['times'], s['times'])
                    for a in rows:
                        np.testing.assert_array_equal(scores[a], trace[a])
                        np.testing.assert_array_equal(trace[a][:256], trace['static'][:256])
                for a, receipts in updates.items():
                    st = update_stats[a]
                    for j, r in enumerate(receipts):
                        assert r['last_observation'] == 255 + 128 * j
                        assert r['first_affected_prediction'] == 256 + 128 * j < len(s['times'])
                        assert all(e + 25 <= r['last_observation'] for e in r['forecast_origins'])
                        assert r['parameter_delta_norm'] <= .00101
                        if a.endswith('observation'):
                            assert r['group_delta_norm']['summary'] == 0.
                        st['steps'] += r['gradient_steps']
                        st['loss_decreased'] += int(r.get('loss_after', float('inf')) < r.get('loss_before', -float('inf')))
                        st['max_step'] = max(st['max_step'], r['parameter_delta_norm'])
                        st['max_summary_step'] = max(st['max_summary_step'], r['group_delta_norm']['summary'])
                replay.append(dict(fold=f, shot=sid, arms=5, exact=True))
            assert access.log == READ(folder / 'access.json')
            for h, model in models.items():
                assert_unchanged(model, frozen[h])
            if f == 0:
                out = root / 'audit_refit0'
                out.mkdir()
                refitted = prepare(parent, fit, source, target, out, ref)
                for h, model in refitted.items():
                    assert_unchanged(model, frozen[h])
                    for suffix in ('manifest.jsonl', 'training.json'):
                        name = f'{h}_{suffix}'
                        assert (out / name).read_bytes() == (folder / name).read_bytes()
        print(dict(stage='verified_replay', fold=f, shots=len(replay)), flush=True)

    for plan in p['folds']:
        for sid in plan['held']:
            s = simple_data.attach_outcome(simple_data.read_shot('EAST', sid, rate_contract='nominal'), 'EAST', {})
            with np.load(root / f'fold{plan["fold"]}' / 'shots' / f'{sid}.npz', allow_pickle=False) as trace:
                valid = s['valid']
                timely = valid & (np.round((s['down'] - s['times']) * 1000, 9) > 30.) if s['is_disrupt'] else valid
                for a in rows:
                    r = independently_score(sid, s['is_disrupt'], s['down'], s['times'][valid], trace[a][valid], p['threshold'])
                    rows[a].append(r)
                    details.append(dict(fold=plan['fold'], arm=a, **r,
                        max_valid_probability=float(trace[a][valid].max()),
                        max_timely_probability=float(trace[a][timely].max()) if timely.any() else None))
    results = {a: aggregate(v, p['threshold']) for a, v in rows.items()}
    assert results == done['metrics']
    expected = READ(root / 'per_shot.json')
    for a in rows:
        assert rows[a] == expected[a]['predictions']
    assert sum(r['steps'] for r in update_stats.values()) == done['neural_steps']
    assert_unchanged(parent, original)
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    analysis = dict(contrasts=contrasts(results), per_shot=details, updates=update_stats)
    _durable_json(root / 'analysis.json', analysis)
    report = dict(status='VERIFIED', completed_sha256=sha(root / 'completed.json'),
        audit_sha256=sha(Path(__file__)), analysis_sha256=sha(root / 'analysis.json'),
        test_sha256=sha(io23.ROOT / 'experiments/risk_aligned_online/tests/test_forecast_token_scope_audit.py'),
        recounts=len(details), full_stream_replays=replay, training_draws_checked=counts,
        exact_refits=[dict(fold=0, heads=['full', 'patch'], steps_per_head=256)],
        source_and_classifier_parameters_verified=True, actual_intra_steps=done['neural_steps'],
        metrics=results, scope='development CV20; same-implementation neural replay, not independent confirmation')
    _durable_json(root / 'verification.json', report)
    print(dict(status='VERIFIED', recounts=len(details), shots=len(replay), contrasts=analysis['contrasts']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
