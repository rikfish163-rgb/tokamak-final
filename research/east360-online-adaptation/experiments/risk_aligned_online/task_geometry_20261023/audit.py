"""Recount all sealed shots; verify preparation and sampled CUDA replay."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.audit import paired_flips
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged, predict_pairs
from ..fixed_test_protocol import load_test_contract
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..target_feedback_meta_20261023.audit import recount
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.audit import counts
from .learning import ARMS, estimate_importance, importance_weights, select_endpoints, predict
from .run import NAMES, READ, load_model


def audit(root, replay):
    p, done, full = READ(root/'protocol.json'), READ(root/'completed.json'), READ(root/'per_shot.json')
    assert done['status'] == 'COMPLETED'
    assert sha(DEFAULT_PARENT) == p['source_sha256'] == PARENT_SHA256
    assert sha(Path(p['parent'])) == p['parent_sha256']
    assert sha(Path(p['plan_path'])) == p['plan_sha256']
    for file, digest in p['code_sha256'].items():
        assert sha(io23.ROOT/file) == digest, file
    ref = Path(p['reference'])
    for file, digest in p['reference_hashes'].items():
        assert sha(ref/file) == digest, file
    assert p['contract'] == load_test_contract('EAST')
    order, ids = p['contract']['shot_ids'], p['selected_ids']
    assert len(ids) == len(set(ids)) == 20 and not set(ids)&set(order)
    assert READ(root/'pre_eval_label_access.json') == [dict(device='EAST',shot=sid) for sid in ids]
    mapping = READ(ref/'input_map.json')
    calibration = READ(root/'calibration.json')
    assert sha(root/'calibration.json') == done['calibration_sha256']
    assert sha(root/'geometry.pt') == done['geometry_sha256']
    assert len(calibration) == 640
    records = []
    for sid in ids:
        shot = simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),
                                          'EAST',{})
        shot.update(transform({k:shot[k] for k in ('values','available')},mapping)); records.append(shot)
        rows = [r for r in calibration if r['shot'] == sid]
        assert [r['end'] for r in rows] == select_endpoints(shot)
        assert [r['label'] for r in rows] == [float(shot['y'][r['end']]) for r in rows]
    assert sum(s['is_disrupt'] for s in records) == 4
    for d in (False,True):
        assert abs(sum(r['weight'] for r in calibration if r['is_disrupt'] == d)-.5) < 1e-12
    geometry = torch.load(root/'geometry.pt',map_location='cpu',weights_only=True)
    a,b = importance_weights(geometry['fisher'])
    for label, recomputed in (('importance',a),('shuffled',b)):
        for k,v in recomputed.items():
            torch.testing.assert_close(v,geometry[label][k],rtol=0,atol=0)
    for k in a:
        torch.testing.assert_close(a[k].flatten().sort().values,b[k].flatten().sort().values,rtol=0,atol=0)
    fisher_sum = sum(v.sum().item() for v in geometry['fisher'].values())
    parameter_weights = {k:dict(elements=v.numel(),min=float(v.min()),median=float(v.median()),
        mean=float(v.mean()),below_half=int((v < .5).sum()),
        fraction_of_empirical_sensitivity=geometry['fisher'][k].sum().item()/fisher_sum) for k,v in a.items()}
    seals = READ(root/'prediction_receipts.json'); assert [r['shot'] for r in seals] == order
    refseals = {r['shot']:r for r in READ(ref/'prediction_receipts.json')}
    total = {a:dict(tp=0,fp=0,n_disrupt=0,n_safe=0,leads=[],early=0) for a in NAMES}
    steps = {a:dict(total=0,nonzero=0,loss_decreased=0,max_delta=0.,zero_direction=0) for a in ARMS}
    for index,seal in enumerate(seals):
        sid = seal['shot']; path,up = root/'shots'/f'{sid}.npz',root/'shots'/f'{sid}_updates.json'
        assert sha(path) == seal['prediction_sha256'] and sha(up) == seal['updates_sha256']
        refpath = ref/'shots'/f'{sid}.npz'
        assert sha(refpath) == seal['reference_sha256'] == refseals[sid]['prediction_sha256']
        d,down = io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as z, np.load(refpath,allow_pickle=False) as old:
            assert set(z.files) == {'times',*NAMES}
            np.testing.assert_array_equal(z['times'],old['times'])
            np.testing.assert_array_equal(z['static'],old['conditional_moments_static'])
            np.testing.assert_array_equal(z['ordinary'],old['conditional_moments_intra'])
            n = len(z['times'])
            for arm in ARMS:
                np.testing.assert_array_equal(z[arm][:256],z['static'][:256])
            for arm in NAMES:
                row = recount(sid,z['times'],z[arm],d,down,p['threshold'])
                assert row == full[arm]['predictions'][index],(sid,arm)
                c = total[arm]; c['n_disrupt' if d else 'n_safe'] += 1
                c['tp'] += row['timely']; c['fp'] += not d and row['alarm']
                if row['timely']:
                    c['leads'].append(row['lead_ms']); c['early'] += row['lead_ms'] > 3000
        updates = READ(up); assert set(updates) == set(ARMS)
        for arm in ARMS:
            assert [u['first_affected_prediction'] for u in updates[arm]] == list(range(256,n,128))
            for u in updates[arm]:
                count = u['first_affected_prediction']
                assert u['last_observation'] == count-1
                assert u['forecast_origins'] == [count-1-25*(j+1) for j in range(4)]
                c = steps[arm]; c['total'] += u['gradient_steps']
                c['nonzero'] += u['parameter_delta_norm'] > 0
                c['max_delta'] = max(c['max_delta'],u['parameter_delta_norm'])
                if 'zero_direction' in u:
                    c['zero_direction'] += u['zero_direction']
                    expected = 0. if u['zero_direction'] else .001*min(u['reference_norm'],1.)
                    assert abs(u['parameter_delta_norm']-expected) < 1e-7
                    c['loss_decreased'] += u['loss_after'] < u['loss_before']
                else:
                    assert u['reason'] == 'no_observed_targets' and u['gradient_steps'] == 0
    for arm,c in total.items():
        m = done['metrics'][arm]
        for key in ('tp','fp','n_disrupt','n_safe'):
            assert m[key] == c[key]
        assert c['n_disrupt'] == c['n_safe'] == 180
        assert c['early'] == m['very_early_over3000'] and float(np.median(c['leads'])) == m['median_lead_ms']
        assert m['tpr'] == c['tp']/180 and m['fpr'] == c['fp']/180 and m['J'] == m['tpr']-m['fpr']
    replays = []; calibration_recomputed = False
    if replay:
        torch.set_num_threads(2); torch.manual_seed(131)
        model,threshold = load_model(); before = state_cpu(model); assert threshold == p['threshold']
        fisher,rows = estimate_importance(model,records,device='cuda')
        assert rows == calibration
        for k,v in fisher.items():
            torch.testing.assert_close(v.cpu(),geometry['fisher'][k],rtol=0,atol=0)
        calibration_recomputed = True
        weights = {a:{k:v.cuda() for k,v in geometry[a].items()} for a in ('importance','shuffled')}
        for sid in (order[0],order[len(order)//2],order[-1]):
            s = apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal = transform({k:s[k] for k in ('values','available')},mapping)
            predictions,updates = predict(model,signal,weights)
            old,_ = predict_pairs({'m':model},signal)
            predictions.update(static=old['m_static'],ordinary=old['m_intra'])
            with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
                for a in NAMES:
                    np.testing.assert_array_equal(predictions[a],z[a])
            assert updates == READ(root/'shots'/f'{sid}_updates.json')
            replays.append(dict(shot=sid,arms=len(NAMES),max_abs_difference=0.,update_receipts_exact=True))
        assert_unchanged(model,before)
    result = dict(status='VERIFIED',completed_sha256=sha(root/'completed.json'),
        protocol_sha256=sha(root/'protocol.json'),audit_sha256=sha(Path(__file__)),
        all_traces_independently_recounted=360*len(NAMES),sampled_cuda_replays=replays,
        full_independent_reinference=False,calibration_recomputed=calibration_recomputed,
        update_summary=steps,parameter_weights=parameter_weights,independent_confirmation=False,
        descriptive_wilson95={a:counts(full[a]['predictions']) for a in NAMES},
        uncertainty_caveat='Descriptive shot intervals do not remove temporal dependence, single seed or development selection.',
        paired_vs_static={a:paired_flips(full['static']['predictions'],full[a]['predictions']) for a in NAMES[1:]},
        paired_vs_ordinary={a:paired_flips(full['ordinary']['predictions'],full[a]['predictions']) for a in ARMS})
    _durable_json(root/'verification.json',result); print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root',type=Path); parser.add_argument('--replay',action='store_true')
    args = parser.parse_args(); audit(args.root,args.replay)
