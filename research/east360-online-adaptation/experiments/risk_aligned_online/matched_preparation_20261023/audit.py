"""Independent recount, training-scope checks and sampled CUDA reinference."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.audit import paired_flips
from ..deepcoral_prefix_20261022.run import predict_pairs, state_cpu, assert_unchanged
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..target_feedback_meta_20261023.audit import recount
from ..warmstart_alignment_20261022.alignment import fit_statistics, make_map, transform
from ..warmstart_alignment_20261022.audit import counts
from ..warmstart_alignment_20261022.run import predict_many
from .run import ARMS, NAMES
from .selection import matched_controls


def audit(root, replay):
    read = lambda p: json.loads(p.read_text())
    protocol, done, full = read(root/'protocol.json'), read(root/'completed.json'), read(root/'per_shot.json')
    assert done['status'] == 'COMPLETED'
    assert sha(DEFAULT_PARENT) == protocol['source_sha256'] == PARENT_SHA256
    assert sha(Path(protocol['plan_path'])) == protocol['plan_sha256']
    for file, digest in protocol['code_sha256'].items():
        assert sha(io23.ROOT/file) == digest, file
    ref = Path(protocol['reference'])
    for file, digest in protocol['reference_hashes'].items():
        assert sha(ref/file) == digest, file
    selection = read(root/'selection.json'); ids = protocol['selected_ids']; order = protocol['contract']['shot_ids']
    selected, pairs = matched_controls(selection['audit']['accepted'], protocol['fixed_disruptive_ids'], test_ids=order)
    assert selection['selected'] == selected and selection['pairs'] == pairs
    assert ids == [r['shot'] for r in selected] and len(ids) == 20
    assert len(selection['development_union_ids']) == protocol['old_and_new_signal_identity_union'] == 34
    assert len(order) == len(set(order)) == 360 and not set(ids)&set(order)
    access = read(root/'pre_eval_label_access.json')
    assert {r['device'] for r in access} == {'J-TEXT', 'EAST'}
    source_ids = set(io23.source_roles()['train'])
    assert {r['shot'] for r in access if r['device'] == 'J-TEXT'} == source_ids
    assert {r['shot'] for r in access if r['device'] == 'EAST'} == set(ids)
    for filename, length in (('risk_training_manifest.jsonl',512), ('forecast_manifest.jsonl',256)):
        new = [json.loads(s) for s in (root/filename).read_text().splitlines()]
        old = [json.loads(s) for s in (ref/filename).read_text().splitlines()]
        assert len(new) == len(old) == length
        for a,b in zip(new,old):
            assert a['step'] == b['step'] and a['source'] == b['source']
            if filename.startswith('risk'):
                assert [r for r in a['target'] if r['label'] == 1] == [r for r in b['target'] if r['label'] == 1]
            for domain, allowed in (('source',source_ids), ('target',set(ids))):
                assert all(r['shot'] in allowed for r in a[domain])
                if filename.startswith('risk'):
                    assert [r['label'] for r in a[domain]] == [0]*16+[1]*16
    mapping = read(root/'input_map.json'); records = []
    for sid in ids:
        s = apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        records.append({k:s[k] for k in ('shot','device','values','available')})
    target = fit_statistics(records,device='EAST',allowed_ids=ids)
    assert make_map(mapping['source_statistics'],target,'density_gain') == mapping
    torch.set_num_threads(2); torch.manual_seed(131)
    base, _ = load_parent(DEFAULT_PARENT); parent = PairedFeedback(base,np.ones(23)).cuda(); parent.set_stage('frozen')
    initial = state_cpu(parent); prep = read(root/'preparation_receipts.json'); states = {}
    changed = {}
    for arm in ARMS:
        path = root/f'{arm}.pt'; assert sha(path) == prep[arm]['checkpoint_sha256']
        state = torch.load(path,map_location='cpu',weights_only=True)['model']; states[arm] = state
        changed[arm] = [k for k in state if not torch.equal(state[k],initial[k])]
        assert changed[arm] == prep[arm]['changed_parameters']
        allowed = {'base.forecast_head.weight','base.forecast_head.bias'}
        if arm != 'input_only':
            allowed |= set(protocol['parameter_scope'])
            assert set(changed[arm]) & set(protocol['parameter_scope'])
        assert set(changed[arm]) <= allowed
    seals = read(root/'prediction_receipts.json'); assert [r['shot'] for r in seals] == order
    refseals = {r['shot']:r for r in read(ref/'prediction_receipts.json')}
    totals = {a:dict(tp=0,fp=0,n_safe=0,n_disrupt=0,early=0,leads=[]) for a in NAMES}
    steps = {a:dict(total=0,nonzero=0,loss_decreased=0,max_delta=0.) for a in ARMS}
    for index, seal in enumerate(seals):
        sid = seal['shot']; path, up = root/'shots'/f'{sid}.npz', root/'shots'/f'{sid}_updates.json'
        assert sha(path) == seal['prediction_sha256'] and sha(up) == seal['updates_sha256']
        refpath = ref/'shots'/f'{sid}.npz'
        assert sha(refpath) == seal['reference_sha256'] == refseals[sid]['prediction_sha256']
        d, down = io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as z, np.load(refpath,allow_pickle=False) as old:
            np.testing.assert_array_equal(z['times'],old['times'])
            np.testing.assert_array_equal(z['raw_source'],old['raw_source'])
            n = len(z['times'])
            for a in ARMS:
                np.testing.assert_array_equal(z[a+'_static'][:256],z[a+'_intra'][:256])
            for a in NAMES:
                row = recount(sid,z['times'],z[a],d,down,protocol['threshold'])
                assert row == full[a]['predictions'][index],(sid,a)
                c = totals[a]; c['n_disrupt' if d else 'n_safe'] += 1
                c['tp'] += row['timely']; c['fp'] += not d and row['alarm']
                if row['timely']:
                    c['leads'].append(row['lead_ms']); c['early'] += row['lead_ms'] > 3000
        updates = read(up); assert set(updates) == set(ARMS)
        for a in ARMS:
            assert [u['first_affected_prediction'] for u in updates[a]] == list(range(256,n,128))
            for u in updates[a]:
                count = u['first_affected_prediction']; assert u['last_observation'] == count-1
                assert u['forecast_origins'] == [count-1-25*(j+1) for j in range(4)]
                if u['gradient_steps']:
                    assert u['gradient_steps'] == 1 and u['parameter_delta_norm'] <= .00101
                    c = steps[a]; c['total'] += 1; c['nonzero'] += u['parameter_delta_norm'] > 0
                    c['loss_decreased'] += u['loss_after'] < u['loss_before']
                    c['max_delta'] = max(c['max_delta'],u['parameter_delta_norm'])
    for a,c in totals.items():
        m = done['metrics'][a]
        for key in ('tp','fp','n_safe','n_disrupt'):
            assert c[key] == m[key]
        assert c['n_safe'] == c['n_disrupt'] == 180 and c['early'] == m['very_early_over3000']
        assert float(np.median(c['leads'])) == m['median_lead_ms']
        assert c['tp']/180 == m['tpr'] and c['fp']/180 == m['fpr'] and m['J'] == m['tpr']-m['fpr']
    replays = []
    if replay:
        models = {}
        for a in ARMS:
            models[a] = copy.deepcopy(parent); models[a].load_state_dict(states[a]); models[a].set_stage('frozen')
        before = {a:state_cpu(m) for a,m in models.items()}
        for sid in (order[0],order[len(order)//2],order[-1]):
            s = apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal = {k:s[k] for k in ('values','available')}
            predictions, updates = predict_pairs(models,transform(signal,mapping))
            predictions['raw_source'] = predict_many(parent,dict(raw=signal),device='cuda')['raw']
            with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
                for a in NAMES:
                    np.testing.assert_array_equal(predictions[a],z[a])
            assert updates == read(root/'shots'/f'{sid}_updates.json')
            replays.append(dict(shot=sid,arms=len(NAMES),max_abs_difference=0.,update_receipts_exact=True))
        for a,m in models.items():
            assert_unchanged(m,before[a])
    assert_unchanged(parent,initial)
    oldfull = read(ref/'per_shot.json')
    result = dict(status='VERIFIED', completed_sha256=sha(root/'completed.json'),
        protocol_sha256=sha(root/'protocol.json'),audit_sha256=sha(Path(__file__)),
        source_archive_unchanged=True, all_traces_independently_recounted=len(NAMES)*360,
        source_sampling_exact=True, positive_target_sampling_exact=True,
        input_map_recomputed=True, changed_parameters=changed,
        update_summary=steps,sampled_cuda_replays=replays,full_independent_reinference=False,
        independent_confirmation=False,descriptive_wilson95={a:counts(full[a]['predictions']) for a in NAMES},
        uncertainty_caveat='Independent-shot descriptive intervals only; do not remove temporal dependence, single seed, or development selection',
        paired_intra={a:paired_flips(full[a+'_static']['predictions'],full[a+'_intra']['predictions']) for a in ARMS},
        old_to_new={a:paired_flips(oldfull[a]['predictions'],full[a]['predictions']) for a in NAMES})
    _durable_json(root/'verification.json',result); print(json.dumps(result),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root',type=Path); p.add_argument('--replay',action='store_true')
    args = p.parse_args(); audit(args.root,args.replay)
