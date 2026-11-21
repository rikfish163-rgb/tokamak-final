"""Independently recount completed continual pilots and their causal receipts."""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from . import simple_data
from .continual_decoder_experiment import ARMS, BASELINE, ROOT, REFERENCE
from .io23 import read_outcome
from .peak_bag_experiment import PEAK_ARMS
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from .simple_online import _durable_json, _hash
from .verify_ended_decoder import classify


def changes(left, right):
    if [r['shot'] for r in left] != [r['shot'] for r in right]:
        raise AssertionError('paired shot identities differ')
    pairs = list(zip(left, right))
    return dict(
        rescued=[a['shot'] for a,b in pairs if a['is_disrupt'] and not a['timely'] and b['timely']],
        lost=[a['shot'] for a,b in pairs if a['is_disrupt'] and a['timely'] and not b['timely']],
        removed_fp=[a['shot'] for a,b in pairs if not a['is_disrupt'] and a['alarm'] and not b['alarm']],
        new_fp=[a['shot'] for a,b in pairs if not a['is_disrupt'] and not a['alarm'] and b['alarm']])


def compare_curves(left, right, order, start=0):
    error = 0.
    for index, row in enumerate(order, 1):
        if index <= start:
            continue
        name = f'{index:04d}_{row["shot"]}_sealed.npz'
        with np.load(left/name, allow_pickle=False) as a, np.load(right/name, allow_pickle=False) as b:
            if not np.array_equal(a['times'], b['times']):
                raise AssertionError('control time axis changed')
            error = max(error, float(np.max(np.abs(a['scores']-b['scores']))))
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--arms', choices=(*ARMS,*PEAK_ARMS), nargs='+', default=list(ARMS))
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        parser.error('duplicate arms')
    output = args.run/('verification_'+'_'.join(args.arms)+'.json')
    if output.exists():
        parser.error('new verification receipt required')
    if _hash(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError('qualified source changed')
    order = json.loads((BASELINE/'order.json').read_text())
    reference = json.loads((BASELINE/'comparison.json').read_text())
    threshold = float(reference['static']['all']['threshold'])
    rows = {a:reference[a]['all']['predictions'] for a in ('static','original_intra','meta_after30')}
    results, steps, heads, seals = {}, {}, {}, 0
    for arm in args.arms:
        folder = args.run/arm
        completed = json.loads((folder/'completed.json').read_text())
        protocol = json.loads((folder/'protocol.json').read_text())
        recorded = json.loads((folder/'comparison.json').read_text())
        if (json.loads((folder/'order.json').read_text()) != order
                or protocol['parent_sha256'] != PARENT_SHA256 or protocol['threshold'] != threshold
                or _hash(folder/'code_snapshot.zip') != protocol['code_snapshot_sha256']):
            raise AssertionError('declared order, source, threshold or code identity differs')
        with zipfile.ZipFile(folder/'code_snapshot.zip') as archive:
            for name in archive.namelist():
                if archive.read(name) != (ROOT/name).read_bytes():
                    raise AssertionError('executed core changed after archival: '+name)
        records, receipts, total, logical, previous_head = [], [], 0, 0, None
        emitted_peaks, ended_grids = {}, {}
        for index, item in enumerate(order, 1):
            sid = item['shot']
            raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
            disruptive, down = read_outcome('EAST', sid)
            seal = json.loads((folder/'stream'/f'{index:04d}_{sid}_seal.json').read_text())
            audit = json.loads((folder/'stream'/f'{index:04d}_{sid}_audit.json').read_text())
            if audit['seal_sha256'] != seal['sha256'] or _hash(folder/'stream'/seal['npz']) != seal['sha256']:
                raise AssertionError('durable prediction identity changed')
            with np.load(folder/'stream'/seal['npz'], allow_pickle=False) as trace:
                if (int(trace['shot']) != sid or not np.array_equal(trace['times'], raw['times'])
                        or trace['scores'].shape != raw['times'].shape
                        or not np.isfinite(trace['scores']).all()):
                    raise AssertionError('raw identity, time, score contract differs')
                records.append(dict(shot=sid, **classify(trace['times'], trace['scores'],
                    np.ones(len(trace['times']), bool), threshold=threshold,
                    disruptive=disruptive, down=down)))
                eligible=np.ones(len(trace['times']),bool)
                if disruptive:eligible &= np.round((down-trace['times'])*1000.,9)>30.
                indices=np.flatnonzero(eligible)
                emitted_peaks[sid]=int(indices[np.argmax(trace['scores'][indices])])
                ended_grids[sid]=(trace['times'].copy(),disruptive,down)
            if (not audit['fast_discarded'] or audit['model_parameter_change_norm'] != 0
                    or audit['model_buffer_change_norm'] != 0):
                raise AssertionError('current-shot disposable update changed persistent state')
            update = audit['persistent']
            if update['effective_from_index'] != index+1:
                raise AssertionError('ended update claimed retroactive effect')
            mode = protocol['mode']
            eligible = not (index < 30 and mode=='late_continual' or index > 30 and mode=='early_fixed')
            history_ids = [o['shot'] for o in order[max(0,index-32):index]]
            labels = {int(r['is_disrupt']) for r in records[-32:]}
            expected = protocol['bootstrap_steps'] if index==30 else protocol['online_steps']
            expected = expected if eligible and labels=={0,1} else 0
            if update['actual_steps'] != expected or bool(update['accepted']) != bool(expected):
                raise AssertionError('executed schedule differs from protocol')
            if expected:
                if update['history_ids'] != history_ids or len(update['records']) != expected:
                    raise AssertionError('task history or real step count differs')
                if previous_head is not None and update['head_before_sha256'] != previous_head:
                    raise AssertionError('head changed outside declared ended callback')
                previous_head = update['head_after_sha256']
                if update['coordinate_transition_reset'] != (index==30):
                    raise AssertionError('undeclared coordinate or optimizer reset')
                logical = expected if index==30 else logical+expected
                if (update['optimizer_steps_after'] != logical
                        or set(update['actual_adam_steps'].values()) != {float(logical)}):
                    raise AssertionError('Adam moments/step were lost between ordinary updates')
                for step in update['records']:
                    trained = step['records']
                    if (sorted(r['label'] for r in trained) != [0,1]
                            or any(r['shot'] not in history_ids for r in trained)
                            or index != 30 and sid not in [r['shot'] for r in trained]):
                        raise AssertionError('non-ended or wrong-class task entered meta update')
                    if protocol.get('peak_policy')=='bag':
                        references={r['shot']:r['query'] for r in update['fixed_peaks']}
                        for task in trained:
                            expected_queries=sorted({references[task['shot']],emitted_peaks[task['shot']]})
                            if (task['queries']!=expected_queries or
                                    task['reference_query']!=references[task['shot']]
                                    or task['selected_query'] not in expected_queries
                                    or [r['query'] for r in task['causal_replays']]!=expected_queries):
                                raise AssertionError('training bag omitted or changed sealed peak')
                            times,is_disrupt,down=ended_grids[task['shot']]
                            for query in expected_queries:
                                if (not 0<=query<len(times) or is_disrupt and
                                        round(float((down-times[query])*1000.),9)<=30.):
                                    raise AssertionError('ineligible endpoint entered supervised bag')
                receipts.append(dict(index=index, steps=expected, history_ids=history_ids))
                total += expected
            seals += 1
        rows[arm] = records
        results[arm] = {}
        for period, start, stop in (('all',0,len(order)),('first30',0,30),('after30',30,len(order))):
            selected = records[start:stop]
            counts = dict(tp=sum(r['timely'] for r in selected),
                fp=sum(r['alarm'] and not r['is_disrupt'] for r in selected),
                n_disrupt=sum(r['is_disrupt'] for r in selected),
                n_safe=sum(not r['is_disrupt'] for r in selected),
                early_over_3000=sum(r['timely'] and r['lead_ms']>3000. for r in selected))
            expected = recorded[period]
            if any(counts[k] != expected[k] for k in counts):
                raise AssertionError('raw counts differ from summary')
            counts.update(tpr=counts['tp']/counts['n_disrupt'], fpr=counts['fp']/counts['n_safe'])
            counts['J'] = counts['tpr']-counts['fpr']
            if any(counts[k] != expected[k] for k in ('tpr','fpr','J')):
                raise AssertionError('raw rates differ from summary')
            results[arm][period] = counts
        saved = torch.load(folder/'final_head.pt', map_location='cpu', weights_only=True)
        if (saved['ended_ids'] != [r['shot'] for r in order]
                or saved['optimizer_steps'] != logical or completed['head_sha256'] != previous_head):
            raise AssertionError('saved head or ended ledger differs from executed callbacks')
        heads[arm] = saved['decoder']
        steps[arm] = dict(total_executed=total, final_optimizer_steps=logical, updates=len(receipts))
    controls = {}
    if 'single_early_fixed' in args.arms:
        previous=ROOT/'runs/continual_decoder120_20261010/early_fixed/stream'
        controls['single_peak_dispatch_hook_vs_archived'] = compare_curves(
            args.run/'single_early_fixed/stream',previous,order)
    if 'early_fixed' in args.arms:
        controls['early_fixed_after30_vs_previous_fixed'] = compare_curves(
            args.run/'early_fixed/stream', BASELINE/'meta_after30', order, start=30)
    if 'early_continual_nofast' in args.arms:
        controls['learned_auxiliary_no_fast_vs_static'] = compare_curves(
            args.run/'early_continual_nofast/stream', BASELINE/'static', order)
    if {'early_continual','early_fixed'} <= set(args.arms):
        controls['early_learning_first30_same'] = compare_curves(
            args.run/'early_continual/stream', args.run/'early_fixed/stream', order[:30])
    if any(controls.values()):
        raise AssertionError('independent unchanged-score control failed')
    same_head = None
    if {'early_continual','early_continual_nofast'} <= set(args.arms):
        same_head = all(torch.equal(heads['early_continual'][k], heads['early_continual_nofast'][k])
                        for k in heads['early_continual'])
        if not same_head:
            raise AssertionError('no-fast comparison learned a different auxiliary head')
    raw_records = []
    for index, item in enumerate(order,1):
        sid = item['shot']
        disruptive, down = read_outcome('EAST',sid)
        folder = REFERENCE/'mask_static'
        seal = json.loads((folder/f'{index:04d}_{sid}_seal.json').read_text())
        if _hash(folder/seal['npz']) != seal['sha256']:
            raise AssertionError('original raw-source seal differs')
        with np.load(folder/seal['npz'],allow_pickle=False) as trace:
            raw_records.append(dict(shot=sid,**classify(trace['times'],trace['scores'],
                np.ones(len(trace['times']),bool),threshold=threshold,disruptive=disruptive,down=down)))
    raw_counts = dict(tp=sum(r['timely'] for r in raw_records),
        fp=sum(r['alarm'] and not r['is_disrupt'] for r in raw_records),
        n_disrupt=sum(r['is_disrupt'] for r in raw_records),
        n_safe=sum(not r['is_disrupt'] for r in raw_records))
    raw_counts.update(tpr=raw_counts['tp']/raw_counts['n_disrupt'],
                      fpr=raw_counts['fp']/raw_counts['n_safe'])
    raw_counts['J']=raw_counts['tpr']-raw_counts['fpr']
    rows['raw_source']=raw_records
    pairs = {f'{base}_to_{arm}':{period:changes(rows[base][start:],rows[arm][start:])
             for period,start in (('all',0),('after30',30))}
             for base in ('raw_source','static','original_intra','meta_after30') for arm in args.arms}
    _durable_json(output, dict(verified_seals=seals, metrics=results, paired_changes=pairs,
        actual_steps=steps, control_max_curve_errors=controls, identical_early_aux_heads=same_head,
        source_unchanged=True, causal_history_verified=True, archived_core_matches=True,
        raw_source_same_prefix_metrics=raw_counts,
        compute_matched=False, scope='DEVELOPMENT pilot, no independent or physical-law claim'))
    print(json.dumps(dict(receipt=str(output), metrics=results, actual_steps=steps,
                         controls=controls, verified_seals=seals)), flush=True)


if __name__ == '__main__':
    main()
