"""Independent counts/loss arithmetic; same-code neural replay is not replication."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

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
from .learning import IncrementFeedback, predict
from .run import fit_heads, HEADS

READ = lambda path: json.loads(path.read_text())


def gate(candidate, static, control):
    rows = (candidate, static, control)
    if any((r['n_disrupt'], r['n_safe']) != (4, 16) for r in rows):
        raise ValueError('matched four disruptive / sixteen safe shots required')
    j = lambda r: r['tp'] / r['n_disrupt'] - r['fp'] / r['n_safe']
    eligible = candidate['tp'] >= static['tp'] and j(candidate) > j(static) and candidate['fp'] / 16 <= .20
    beats_control = candidate['tp'] >= control['tp'] and j(candidate) > j(control)
    return dict(pass_static=eligible, beats_absolute_control=beats_control,
                **{'pass': eligible and beats_control})


def audit(root):
    if (root/'verification.json').exists():
        raise FileExistsError(root/'verification.json')
    p, done = READ(root/'protocol.json'), READ(root/'completed.json')
    assert done['status'] == 'COMPLETED' and not done['test360_accessed']
    for name, digest in done['artifact_sha256'].items():
        assert sha(root/name) == digest, name
    for name, digest in p['code_sha256'].items():
        assert sha(io23.ROOT/name) == digest, name
    assert sha(Path(p['plan_path'])) == p['plan_sha256']
    assert sha(DEFAULT_PARENT) == PARENT_SHA256 == p['source_sha256']
    reference = Path(p['reference'])
    assert sha(reference/'completed.json') == p['reference_sha256']
    for name, digest in READ(reference/'completed.json')['artifact_sha256'].items():
        assert sha(reference/name) == digest, name
    held = [sid for f in p['folds'] for sid in f['held']]
    assert len(set(held)) == len(held) == 20
    assert all(set(f['fit']) == set(held)-set(f['held']) for f in p['folds'])
    torch.set_num_threads(2); torch.manual_seed(131)
    base, info = load_parent(DEFAULT_PARENT)
    parent = PairedFeedback(base, np.ones(23)).cuda(); parent.set_stage('frozen')
    original = state_cpu(parent)
    assert info['threshold'] == p['threshold']
    source_ids = io23.source_roles()['train']
    read_signal, read_outcome = simple_data.read_shot, io23.read_outcome

    def source_guard(device, sid):
        assert device == 'J-TEXT' and sid in source_ids
        return read_outcome(device, sid)

    with patch.object(io23, 'read_outcome', side_effect=source_guard):
        source, _ = load_source_train()
    assert READ(root/'source_label_access.json') == source_ids
    seals = {(r['fold'], r['shot']): r for r in READ(root/'prediction_receipts.json')}
    replays = []; draws = 0
    update_stats = {h: dict(steps=0, loss_decreased=0, max_parameter_step=0., omitted=0) for h in HEADS}
    for plan in p['folds']:
        f = plan['fold']; folder = root/f'fold{f}'; ref = reference/f'fold{f}'
        fit = copy.deepcopy(parent)
        fit.load_state_dict(torch.load(ref/'full.pt', map_location='cuda', weights_only=True)['model'])
        fit.set_stage('frozen'); fit_state = state_cpu(fit); models = {}
        mapping = READ(folder/'input_map.json'); assert mapping == READ(ref/'input_map.json')
        for h in HEADS:
            model = IncrementFeedback(copy.deepcopy(fit.base), h).cuda()
            saved = torch.load(folder/f'{h}.pt', map_location='cuda', weights_only=True)
            assert saved['forecast_mode'] == h
            model.load_state_dict(saved['model']); model.set_stage('frozen')
            assert_unchanged(model, fit_state, ('base.forecast_head.weight', 'base.forecast_head.bias'))
            models[h] = model
        frozen = {h: state_cpu(m) for h, m in models.items()}
        access = FoldAccess(plan, source_ids)

        def signal_guard(device, sid, **kwargs):
            access.check('signal', device, sid)
            return read_signal(device, sid, **kwargs)

        def forbidden(*args, **kwargs):
            raise AssertionError('no EAST outcome in forecast fitting or prediction replay')

        with patch.object(simple_data, 'read_shot', side_effect=signal_guard), patch.object(io23, 'read_outcome', side_effect=forbidden):
            records = [apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal')) for sid in plan['fit']]
            target = [dict(r, **transform({k:r[k] for k in ('values','available')}, mapping)) for r in records]
            groups = {'source': {s['shot']: s for s in source}, 'target': {s['shot']: s for s in target}}
            for h in HEADS:
                manifest = folder/f'{h}_manifest.jsonl'
                assert manifest.read_bytes() == (ref/'full_manifest.jsonl').read_bytes()
                samples = [json.loads(line) for line in manifest.read_text().splitlines()]
                assert len(samples) == 256
                for step, row in enumerate(samples):
                    assert row['step'] == step
                    for domain, pool in groups.items():
                        assert len(row[domain]) == 32
                        for item in row[domain]:
                            assert 127 <= item['end'] < len(pool[item['shot']]['values'])-25
                            draws += 1
            access.phase = 'predict'
            for sid in plan['held']:
                s = apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal = transform({k:s[k] for k in ('values','available')}, mapping)
                path = folder/'shots'/f'{sid}.npz'; up = folder/'shots'/f'{sid}_updates.json'
                assert sha(path) == seals[f,sid]['prediction_sha256'] and sha(up) == seals[f,sid]['updates_sha256']
                cached_updates = READ(up)
                with np.load(path, allow_pickle=False) as cached:
                    np.testing.assert_array_equal(cached['times'], s['times'])
                    with np.load(ref/'shots'/f'{sid}.npz',allow_pickle=False) as previous:
                        np.testing.assert_array_equal(cached['legacy'], previous['full_all'])
                    for h, model in models.items():
                        scores, updates, packets = predict(model, signal)
                        assert updates == cached_updates[h]
                        np.testing.assert_array_equal(scores['intra'], cached[h])
                        np.testing.assert_array_equal(scores['static'], cached['static'])
                        np.testing.assert_array_equal(cached[h][:256],cached['static'][:256])
                        with np.load(folder/'shots'/f'{sid}_{h}_forecast.npz',allow_pickle=False) as pack:
                            for key, value in packets.items(): np.testing.assert_array_equal(value, pack[key])
                        for j, r in enumerate(updates):
                            assert r['last_observation'] == 255+128*j
                            assert r['first_affected_prediction'] == 256+128*j < len(s['times'])
                            assert all(e+25 <= r['last_observation'] for e in r['forecast_origins'])
                            assert r['parameter_delta_norm'] <= .00101
                            st = update_stats[h]; st['steps'] += r['gradient_steps']
                            st['loss_decreased'] += int(r.get('loss_after',np.inf) < r.get('loss_before',-np.inf))
                            st['max_parameter_step'] = max(st['max_parameter_step'], r['parameter_delta_norm'])
                            st['omitted'] += r['omitted_unobserved_last']
                replays.append(dict(fold=f,shot=sid,arms=2,exact=True))
            assert access.log == READ(folder/'access.json')
            for h, model in models.items(): assert_unchanged(model,frozen[h])
            if f == 0:
                out = root/'audit_refit0'; out.mkdir()
                refitted = fit_heads(fit,source,target,out,ref)
                for h, model in refitted.items():
                    assert_unchanged(model,frozen[h])
                    for suffix in ('manifest.jsonl','training.json'):
                        assert (out/f'{h}_{suffix}').read_bytes() == (folder/f'{h}_{suffix}').read_bytes()
        print(dict(stage='verified_replay',fold=f,shots=len(replays)),flush=True)

    # Labels and unseen target values are used only for scoring sealed outputs.
    rows = {a:[] for a in ('static','legacy',*HEADS)}; details = []
    future = READ(root/'future_windows.json')
    lookup = {(r['shot'],r['origin'],r['arm']):r for r in future}
    checked = 0; max_loss_difference = 0.
    center = parent.base.center.cpu().double(); scale = parent.base.scale.cpu().double()
    for plan in p['folds']:
        folder = root/f'fold{plan["fold"]}'; mapping = READ(folder/'input_map.json')
        for sid in plan['held']:
            s = simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
            signal = transform({k:s[k] for k in ('values','available')},mapping); valid = s['valid']
            timely = valid & (np.round((s['down']-s['times'])*1000,9)>30.) if s['is_disrupt'] else valid
            with np.load(folder/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
                for a in rows:
                    r = independently_score(sid,s['is_disrupt'],s['down'],s['times'][valid],trace[a][valid],p['threshold'])
                    rows[a].append(r)
                    details.append(dict(arm=a,**r,max_timely_probability=float(trace[a][timely].max()) if timely.any() else None))
            for h in HEADS:
                with np.load(folder/'shots'/f'{sid}_{h}_forecast.npz',allow_pickle=False) as pack:
                    for j, count in enumerate(pack['counts']):
                        assert count+25 <= len(s['times'])
                        np.testing.assert_array_equal(pack['last_value'][j],signal['values'][count-1])
                        np.testing.assert_array_equal(pack['last_available'][j],signal['available'][count-1])
                        raw = signal['values'][count:count+25]; last = signal['values'][count-1]
                        lm = signal['available'][count-1]&np.isfinite(last)
                        mask = torch.as_tensor(signal['available'][count:count+25]&np.isfinite(raw)&lm[None])
                        truth = torch.asinh((torch.as_tensor(np.where(mask.numpy(),raw,center.numpy())).double()-center)/scale)
                        persistence = torch.asinh((torch.as_tensor(np.where(lm,last,center.numpy())).double()-center)/scale)[None].expand_as(truth)
                        saved = lookup[sid,int(count)-1,h]
                        assert saved['observed'] == int(mask.sum())
                        assert saved['is_disrupt'] == bool(s['is_disrupt'])
                        assert saved['future_all_valid'] == bool(valid[count:count+25].all())
                        assert saved['future_any_positive'] == bool((s['y'][count:count+25]>.5).any())
                        for key,pred in (('static',torch.as_tensor(pack['static'][j]).double()),
                                         ('after',torch.as_tensor(pack['intra'][j]).double()),('persistence',persistence)):
                            loss = F.smooth_l1_loss(pred,truth,reduction='none')
                            if mask.any():
                                error = abs(float(loss[mask].mean())-saved[key]); max_loss_difference=max(max_loss_difference,error)
                                assert error < 2e-12
                            else: assert saved[key] is None
                        assert saved['before'] == saved['static']  # Not the immediately preceding fast model.
                        checked += 1
    results = {a:aggregate(r,p['threshold']) for a,r in rows.items()}
    assert results == done['metrics']
    for a in rows: assert rows[a] == READ(root/'per_shot.json')[a]['predictions']
    assert checked == len(future) and sum(r['steps'] for r in update_stats.values()) == done['neural_steps']
    future_summary = READ(root/'future_summary.json')
    for h in HEADS:
        selected = [r for r in future if r['arm']==h and r['after'] is not None]
        ids = sorted({r['shot'] for r in selected})
        assert len(ids) == future_summary[h]['n_shots'] and len(selected) == future_summary[h]['n_windows']
        for key in ('before','after','static','persistence'):
            mean = float(np.mean([np.mean([r[key] for r in selected if r['shot']==sid]) for sid in ids]))
            assert abs(mean-future_summary[h]['mean_losses'][key]) < 2e-12
    assert_unchanged(parent,original); assert sha(DEFAULT_PARENT)==PARENT_SHA256
    analysis = dict(metrics=results,gate=gate(results['increment'],results['static'],results['absolute']),
        per_shot=details,updates=update_stats,
        forecast_comparison='cumulative adapted versus static; before aliases static, not the previous fast state')
    _durable_json(root/'analysis.json',analysis)
    report = dict(status='VERIFIED',completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        analysis_sha256=sha(root/'analysis.json'),independent_warning_recounts=80,independent_future_recounts=checked,
        max_future_loss_difference=max_loss_difference,complete_replays=replays,manifest_draws_checked=draws,
        refitted_fold=0,refitted_heads=list(HEADS),source_unchanged=True,test360_accessed=False,
        scope='CV20 development; counts and arithmetic independent, neural replays same implementation')
    _durable_json(root/'verification.json',report); print(report,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('root',type=Path); audit(parser.parse_args().root)
