"""Independent metrics and invariants for the single-factor crossfit experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import audit_v2 as audit


def main(original, candidate, output):
    if output.exists():
        raise FileExistsError(output)
    original, candidate = Path(original), Path(candidate)
    prior = audit.read_json(original/'protocol.json')
    spec = audit.read_json(candidate/'protocol.json')
    outcomes = audit.read_json(candidate/'outcomes.json')
    assert outcomes == audit.read_json(original/'outcomes.json')
    assert spec['density_scale'] == 'leave_shot_out'
    assert spec['folds'] == prior['folds']
    audit.validate_folds(spec['folds'], [int(s) for s in outcomes])
    _, observations = audit.compare_npz(original/'observations.npz', candidate/'observations.npz')
    rows, restricted, updates = [], [], {a: 0 for a in audit.ARMS}
    checked_means, checked_inner_folds = 0, 0
    for fold in spec['folds']:
        old, new = original/f"fold{fold['fold']}", candidate/f"fold{fold['fold']}"
        sealed = audit.load_npz(new/'sealed.npz')
        data = audit.load_npz(new/'neural.npz')
        old_data = audit.load_npz(old/'neural.npz')
        for filename in ('sealed', 'neural'):
            seal_name = 'sealed.json' if filename == 'sealed' else 'neural_seal.json'
            assert audit.sha256(new/(filename+'.npz')) == audit.read_json(new/seal_name)['sha256']
        for key in ('baseline_logits', 'origin_logit', 'ids', 'ends', 'future_times'):
            np.testing.assert_array_equal(old_data[key], data[key])
        audit.validate_identical_control(sealed, data)
        counts, _ = audit.validate_norms(data)
        for arm, count in counts.items():
            updates[arm] += count
        joined = audit.validate_episode_alignment(sealed, data)
        for arm in ('real', 'misplaced'):
            a = torch.load(old/(arm+'_density.pt'), weights_only=True, map_location='cpu')
            b = torch.load(new/(arm+'_density.pt'), weights_only=True, map_location='cpu')
            assert b['crossfit']['method'] == 'leave_one_shot_out'
            for ca, cb in zip(a['channels'], b['channels'], strict=True):
                assert ca['supported'] == cb['supported']
                for pa, pb in zip(ca['classes'], cb['classes'], strict=True):
                    assert pa['nrows'] == pb['nrows'] and pa['shot_ids'] == pb['shot_ids']
                    if not ca['supported']:
                        assert pb['ridge'] is None and pb['crossfit']['residual_rows'] == 0
                        continue
                    for key, val in pa['ridge'].items():
                        if isinstance(val, torch.Tensor):
                            assert torch.equal(val, pb['ridge'][key]), key
                        else:
                            assert val == pb['ridge'][key], key
                    checked_means += 1
                    meta = pb['crossfit']
                    assert meta['residual_rows'] == pb['nrows']
                    assert len(meta['folds']) == pb['nshots']
                    assert set(pb['shot_ids']) <= set(fold['fit'])
                    for inner in meta['folds']:
                        assert set(inner['fit_shot_ids']) == set(pb['shot_ids'])-{inner['held_shot_id']}
                        assert not (set(inner['fit_shot_ids']) & set(fold['held']))
                        checked_inner_folds += 1
        for sid in fold['held']:
            selection = data['ids'] == sid
            r, masks = audit._score_shot(data, selection, sid, fold['fold'], outcomes[str(sid)])
            rows.extend(r)
            if not outcomes[str(sid)]['disruptive']:
                restricted.extend(r)
                continue
            selected_indices = joined[selection]
            eligible = audit.origin_timely_eligible(sealed['origin_time'][selected_indices],
                sealed['arrival_time'][selected_indices], outcomes[str(sid)])
            mask = masks['danger_timely'] & eligible[:, None]
            if not mask.any():
                continue
            y, _ = audit.future_groups(data['future_times'][selection], outcomes[str(sid)])
            before = audit.binary_cross_entropy(data['baseline_logits'][selection], y)
            for arm in audit.ARMS:
                delta = audit.binary_cross_entropy(data[arm+'_logits'][selection], y)-before
                restricted.append(dict(shot=sid, fold=fold['fold'], arm=arm, group='danger_timely',
                    windows=int(mask.any(1).sum()), timepoints=int(mask.sum()),
                    delta_bce=float(delta[mask].mean())))
    error = audit._compare_published_rows(rows, audit.read_json(candidate/'neural_per_shot.json'))
    groups, origin_groups = audit.aggregate_rows(rows), audit.aggregate_rows(restricted)
    primary_gate, primary_fail = audit.direction_gate(groups)
    origin_gate, origin_fail = audit.direction_gate(origin_groups)
    published = audit.read_json(candidate/'completed.json')['neural']
    assert updates == published['actual_nonzero_episode_updates']
    assert primary_gate == published['eligible_for_full_stream']
    for arm in audit.ARMS:
        for group, vals in published['groups'][arm].items():
            for key, val in vals.items():
                assert abs(groups[arm][group][key]-val) <= 1e-12
    result = dict(observations_match_original=observations, baseline_predictions_exact=True,
        final_ridge_models_exact=checked_means, checked_inner_exclusion_records=checked_inner_folds,
        maximum_metric_difference=error, updates=updates, groups=groups,
        origin_restricted_groups=origin_groups,
        primary_gate=primary_gate, primary_failed_groups=primary_fail,
        origin_restricted_gate=origin_gate, origin_restricted_failed_groups=origin_fail,
        eligible_for_full_stream=primary_gate and origin_gate,
        code_sha256=audit.sha256(Path(__file__)),
        candidate_completed_sha256=audit.sha256(candidate/'completed.json'),
        original_completed_sha256=audit.sha256(original/'completed.json'),
        scope='Saved predictions independently recomputed; not a new neural replay or calibration proof.')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('final_ridge_models_exact', 'checked_inner_exclusion_records',
        'maximum_metric_difference', 'eligible_for_full_stream', 'updates')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.original, args.candidate, args.output)
