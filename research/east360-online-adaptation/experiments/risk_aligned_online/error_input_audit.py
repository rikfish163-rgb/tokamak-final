"""Static whole-shot input interventions and quality audit; not a repair rollout.

Only an archived qualified parent is read. Original predictions, source weights,
alarm threshold, order, and production input masks are never changed. CIII=-10
uses the EXISTING atlas23 documented sentinel tolerance, not a fitted cutoff.
Masking all Bt is a dependence test, not a valid field calibration.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from . import io23, simple_data, simple_train
from .diagnostic_training import load_parent
from .error_forensics import PARENT, RUNS, record, write_csv, write_json
from .simple_online import predict_shot, seal_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('fresh diagnostic output required')
    torch.set_num_threads(4)
    parent, ckpt = load_parent(PARENT)
    immutable = {k: v.detach().cpu().clone() for k, v in parent.state_dict().items()}
    parent_sha = simple_train.sha(PARENT)
    original = RUNS/'diagnostic_aligned_east360_20260929'
    protocol = json.loads((original/'protocol.json').read_text())
    order = protocol['order']
    threshold = float(ckpt['threshold'])
    args.output.mkdir(parents=True)
    arms = ('mask_CIII_sentinel', 'mask_Bt')
    for arm in arms:
        (args.output/arm).mkdir()
    saved = {a: [] for a in arms}
    quality, ledger = [], []
    for position, entry in enumerate(order, 1):
        sid = entry['shot']
        shot = simple_data.read_shot('EAST', sid)
        # Pointwise causal indicator; no full-shot-constant test changes inputs.
        sentinel = np.isclose(shot['values'][:, 1], -10., rtol=0., atol=1e-2)
        quality.append(dict(shot=sid, position=position,
                            first100_Bt_median=float(np.median(shot['values'][:100, 0])),
                            first100_CIII_sentinel_fraction=float(sentinel[:100].mean()),
                            CIII_sentinel_fraction=float(sentinel.mean()),
                            CIII_sentinel_marked_observed=int(np.sum(sentinel & shot['available'][:, 1])),
                            mode_n_all_zero=bool(np.all(shot['values'][:, 10] == 0.))))
        predictions = {}
        for arm in arms:
            changed = dict(shot, values=shot['values'].copy(), available=shot['available'].copy())
            if arm == 'mask_CIII_sentinel':
                changed['values'][sentinel, 1] = 0.
                changed['available'][sentinel, 1] = False
            else:
                changed['values'][:, 0] = 0.
                changed['available'][:, 0] = False
            probabilities, _ = predict_shot(parent, changed, intra=False, device='cuda')
            seal_predictions(args.output/arm, position, sid, shot['times'], probabilities,
                             arm=arm, threshold=threshold)
            predictions[arm] = probabilities
        # Outcomes are evaluator-only, after BOTH prediction traces are sealed.
        revealed = simple_data.attach_outcome(shot, 'EAST', {})
        valid = revealed['valid']
        with np.load(original/'static'/f'{position:04d}_{sid}_sealed.npz', allow_pickle=False) as old:
            baseline = record(old['times'], old['scores'], revealed['is_disrupt'], revealed['down'], threshold)
        row = dict(shot=sid, position=position, baseline=baseline['category'])
        quality[-1]['baseline'] = baseline['category']
        for arm, probabilities in predictions.items():
            saved[arm].append(dict(shot=sid, is_disrupt=revealed['is_disrupt'], down=revealed['down'],
                                  times=shot['times'][valid], scores=probabilities[valid]))
            measured = record(shot['times'], probabilities, revealed['is_disrupt'], revealed['down'], threshold)
            row.update({arm+'_'+k: v for k, v in measured.items()})
        ledger.append(row)
        if position % 60 == 0:
            print(f'static input counterfactual {position}/360', flush=True)
    summary = {}
    for arm in arms:
        summary[arm] = simple_train.metrics(saved[arm], threshold, lead=30.)
        summary[arm]['transitions'] = dict(Counter(r['baseline']+'->'+r[arm+'_category'] for r in ledger))
    summary['quality'] = dict(
        CIII_sentinel_any_shots=sum(r['CIII_sentinel_fraction'] > 0 for r in quality),
        CIII_sentinel_all_shots=sum(r['CIII_sentinel_fraction'] == 1 for r in quality),
        CIII_sentinel_all_by_class=dict(Counter(r['baseline'] for r in quality if r['CIII_sentinel_fraction'] == 1)),
        incorrectly_observed_CIII_points=sum(r['CIII_sentinel_marked_observed'] for r in quality),
        Bt_small_group_count=sum(abs(r['first100_Bt_median']) < .1 for r in quality),
        Bt_small_group_by_class=dict(Counter(r['baseline'] for r in quality if abs(r['first100_Bt_median']) < .1)),
        mode_n_all_zero_shots=sum(r['mode_n_all_zero'] for r in quality))
    if any(not torch.equal(v.detach().cpu(), immutable[k]) for k, v in parent.state_dict().items()) or simple_train.sha(PARENT) != parent_sha:
        raise AssertionError('read-only diagnostic changed source')
    write_json(args.output/'protocol.json', dict(
        evidence='DEVELOPMENT_ONLY_STATIC_INPUT_COUNTERFACTUAL_NOT_ADAPTATION',
        parent_sha256=parent_sha, parent=str(PARENT.resolve()), order=order, threshold=threshold,
        source_code_sha256=simple_train.sha(__file__),
        sentinel_source='src/atlas23/contract.py:CIII_MISSING_VALUE=-10; CIII_MISSING_ATOL=1e-2',
        sentinel_contract_sha256=simple_train.sha(io23.ROOT/'src/atlas23/contract.py'),
        production_masks_unchanged=True, parent_bitwise_unchanged=True,
        outcome_access='after sealing both counterfactual traces; evaluator only'))
    write_csv(args.output/'input_quality_all360.csv', quality)
    write_csv(args.output/'counterfactual_all360.csv', ledger)
    write_json(args.output/'summary.json', summary)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'predictions'}
                      for k, v in summary.items()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
