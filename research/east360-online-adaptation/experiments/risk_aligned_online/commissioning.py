"""Causal EAST commissioning replay with a bounded startup phase.

The source checkpoint is reset for each arm. Predictions and their receipt are
durably sealed before outcome reveal. This is offline development diagnostics,
not a realtime or physical-safety guarantee.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.fast_online_idea.online import cohort_ids
from . import data, online as O, stage_metrics
from .model import ModelConfig, RiskAlignedModel
from .source_gate import qualify_source


ARMS = ('static', 'startup_freeze', 'intra', 'inter', 'both')
LEAD_MS = 30.
LEAD_ROUND_DIGITS_MS = 9  # numerical precision only; no extra physical margin


def summarize_strict(rows, threshold):
    """First eligible EAST alarm; round lead to 1e-9 ms, then require >30."""
    if not np.isfinite(threshold):
        raise ValueError('finite fixed threshold required')
    tp = fp = nd = ns = 0
    leads, predictions = [], []
    for row in rows:
        times = np.asarray(row['times'], np.float64)
        scores = np.asarray(row['scores'], np.float64)
        if times.shape != scores.shape or times.ndim != 1:
            raise ValueError('shot times/scores mismatch')
        disruptive = bool(row['is_disrupt'])
        down = row['down']
        if disruptive and (down is None or not np.isfinite(down)):
            raise ValueError('disruptive shot lacks DownTime')
        eligible = times <= down if disruptive else np.ones(len(times), bool)
        fired = np.flatnonzero(eligible & (scores >= threshold))
        alarm = bool(len(fired))
        lead = (round(float((down-times[fired[0]])*1000.), LEAD_ROUND_DIGITS_MS)
                if disruptive and alarm else None)
        timely = bool(lead is not None and lead > LEAD_MS)
        if disruptive:
            nd += 1
            tp += timely
            if timely:
                leads.append(lead)
        else:
            ns += 1
            fp += alarm
        predictions.append(dict(shot=int(row['shot']), is_disrupt=disruptive,
                                alarm=alarm, timely=timely, lead_ms=lead))
    tpr = tp/nd if nd else None
    fpr = fp/ns if ns else None
    return dict(tpr=tpr, fpr=fpr, J=tpr-fpr if nd and ns else None,
                tp=tp, fp=fp, n_disrupt=nd, n_safe=ns,
                median_lead_ms=float(np.median(leads)) if leads else None,
                early_over_3000=sum(lead > 3000 for lead in leads),
                threshold=float(threshold), predictions=predictions)


def _history_proxy(model, replay, threshold, device):
    """Risk-window proxy on PRIOR revealed raw replay, not held-out shots."""
    records = [r for r in replay if len(r['risk_windows'])]
    if not records:
        return None
    values = torch.as_tensor(np.concatenate([r['risk_windows'] for r in records]), device=device)
    available = torch.as_tensor(np.concatenate([r['risk_available'] for r in records]), device=device)
    labels = torch.as_tensor(np.concatenate([r['y125'] for r in records]), device=device)
    if len(values) != len(labels):
        raise ValueError('historical guard window/label count mismatch')
    model.eval()
    with torch.no_grad():
        logits = model(values, available)['logits'][:, 1]
        losses = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        correct = (logits.sigmoid() >= threshold) == (labels >= .5)
    return dict(correct=correct.cpu().numpy(), mean_bce=float(losses.mean()),
                windows=len(values))


def apply_persistent_candidate(model, shot, replay, threshold, sealed_scores, *,
                               device, lr, steps, guarded=True, data_api=data,
                               control='learned'):
    """Train from revealed data; reject damaging candidate and restore FULL state."""
    if not np.isfinite(lr) or lr <= 0 or steps < 1:
        raise ValueError('positive learning rate and steps required')
    # Full state always precedes the candidate, including the first/no-history
    # update and the unguarded ablation. Any exception rolls back and re-raises.
    snapshot = {key: value.detach().clone() for key, value in model.state_dict().items()}
    try:
        history = _history_proxy(model, replay, threshold, device) if guarded else None
        learning, record = O.persistent_update(model, shot, replay, device, lr, steps,
                                               sealed_scores=sealed_scores,
                                               data_api=data_api, control=control)
        candidate_state = model.state_dict()
        if any(not torch.isfinite(value).all() for value in candidate_state.values()):
            raise FloatingPointError('persistent candidate contains nonfinite state')
        candidate_delta = sum(float((value.detach().double()-snapshot[key].double()).square().sum())
                              for key, value in candidate_state.items()) ** .5
        if not guarded:
            guard = dict(accepted=True, reason='unguarded_ablation', prior_windows=0,
                         new_wrong_previously_correct=0, mean_bce_increase=None)
        elif history is None:
            guard = dict(accepted=True, reason='no_history', prior_windows=0,
                         new_wrong_previously_correct=0, mean_bce_increase=None)
        else:
            candidate = _history_proxy(model, replay, threshold, device)
            new_wrong = int(np.count_nonzero(history['correct'] & ~candidate['correct']))
            increase = candidate['mean_bce'] - history['mean_bce']
            if not np.isfinite(increase):
                raise FloatingPointError('nonfinite historical guard loss change')
            accepted = bool(new_wrong == 0 and increase <= 1e-4)
            guard = dict(accepted=accepted,
                         reason=('accepted_window_proxy' if accepted else
                                 'historical_window_regression'),
                         prior_windows=history['windows'],
                         new_wrong_previously_correct=new_wrong,
                         mean_bce_before=history['mean_bce'],
                         mean_bce_after=candidate['mean_bce'],
                         mean_bce_increase=float(increase))
        if not guard['accepted']:
            model.load_state_dict(snapshot)
    except Exception:
        model.load_state_dict(snapshot)
        raise
    guard.update(candidate_model_delta_norm=candidate_delta,
                 accepted_model_delta_norm=candidate_delta if guard['accepted'] else 0.,
                 candidate_parameter_delta_norms=learning.get('parameter_delta_norms'),
                 accepted_parameter_delta_norms=(learning.get('parameter_delta_norms')
                     if guard['accepted'] else
                     {key: 0. for key in learning.get('parameter_delta_norms', {})}),
                 shot_safety_guarantee=False,
                 proxy='prior revealed risk windows; training replay overlaps guard')
    return learning, record, guard


def _compact_cumulative(tp, fp, nd, ns):
    tpr = tp/nd if nd else None
    fpr = fp/ns if ns else None
    return dict(tp=tp, fp=fp, n_disrupt=nd, n_safe=ns,
                tpr=tpr, fpr=fpr, J=tpr-fpr if nd and ns else None)


def stage_summaries(results, full_order, startup_budget):
    """Posthoc sidecar for both tracks, with quartiles from FULL roster."""
    return {arm: {
        key: stage_metrics.summarize_commissioning(result[key]['predictions'],
              full_order, startup_budget, block_size=20)
        for key in ('deployed', 'same_parent_no_fast_shadow')}
        for arm, result in results.items()}


def run_arm(model, *, order, arm, threshold, output_dir, startup_budget=20,
            device='cpu', data_api=data, guarded=True, startup_lr=.001,
            later_lr=.00025, inter_steps=1, control='learned'):
    if arm not in ARMS or startup_budget < 1:
        raise ValueError('unknown arm or invalid startup budget')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    parent = O.state_hash(model)
    replay = deque(maxlen=8)
    rows, shadows = [], []
    tp = fp = nd = ns = accepted_updates = rejected_updates = 0
    started = time.perf_counter()
    for number, item in enumerate(order, 1):
        tick = time.perf_counter()
        sid = int(item['shot'])
        phase = 'startup' if number <= startup_budget else 'post_startup'
        shot = data_api.read_shot('EAST', sid)
        if int(shot['shot']) != sid or shot.get('device', 'EAST') != 'EAST':
            raise ValueError('signal shot identity differs from roster')
        shot_parent = O.state_hash(model)
        prediction = O.predict_shot(model, shot, intra=arm in ('intra', 'both'),
                                    target_domain='EAST', device=device,
                                    data_api=data_api, control=control)
        sealed = output_dir/f'{sid}_sealed.npz'
        with sealed.open('xb') as handle:
            np.savez_compressed(handle, times=shot['times'], scores=prediction['scores'],
                                shadow_scores=prediction['shadow_scores'])
            handle.flush()
            os.fsync(handle.fileno())
        seal_hash = O.file_hash(sealed)
        O.write_json(output_dir/f'{sid}_seal.json',
                     dict(sealed_sha256=seal_hash, parent_hash=shot_parent, threshold=threshold))
        # The only metadata access is after both the score file and receipt
        # have been durably written. No future shot is loaded before this point.
        data_api.attach_outcome(shot, 'EAST', {})
        shot['device'] = 'EAST'
        row = dict(shot=sid, times=shot['times'], scores=prediction['scores'],
                   is_disrupt=shot['is_disrupt'], down=shot['down'])
        rows.append(row)
        shadows.append(dict(row, scores=prediction['shadow_scores']))
        one = summarize_strict([row], threshold)['predictions'][0]
        if one['is_disrupt']:
            nd += 1
            tp += one['timely']
        else:
            ns += 1
            fp += one['alarm']
        update_allowed = arm in ('startup_freeze', 'inter', 'both') and (
            arm != 'startup_freeze' or number <= startup_budget)
        learning = None
        guard = dict(accepted=None, reason='no_persistent_update',
                     shot_safety_guarantee=False)
        lr = (startup_lr if phase == 'startup' else later_lr) if update_allowed else None
        if update_allowed:
            learning, record, guard = apply_persistent_candidate(
                model, shot, replay, threshold, prediction['scores'], device=device,
                lr=lr, steps=inter_steps, guarded=guarded,
                data_api=data_api, control=control)
            replay.append(record)  # even a rejected candidate remains revealed history
            accepted_updates += int(guard['accepted'])
            rejected_updates += int(not guard['accepted'])
        elif arm == 'startup_freeze':
            replay.append(O.select_experience(model, shot, sealed_scores=prediction['scores'],
                                              data_api=data_api))
        audit = dict(index=number, shot=sid, phase=phase, startup_budget=startup_budget,
                     parent_hash=shot_parent, next_hash=O.state_hash(model),
                     sealed_sha256=seal_hash, threshold=threshold,
                     is_disrupt=bool(shot['is_disrupt']), down=shot['down'],
                     prediction=one, fast_reset=prediction['fast_reset'],
                     fast_updates=prediction['fast_updates'],
                     inter_update=learning, guard=guard, persistent_lr=lr,
                     candidate_model_delta_norm=guard.get('candidate_model_delta_norm'),
                     accepted_model_delta_norm=guard.get('accepted_model_delta_norm'),
                     cumulative=_compact_cumulative(tp, fp, nd, ns),
                     accepted_updates=accepted_updates, rejected_updates=rejected_updates,
                     prediction_wall_s=prediction['wall_s'],
                     fast_update_wall_s=prediction['fast_update_wall_s'],
                     shot_wall_s=time.perf_counter()-tick)
        O.write_json(output_dir/f'{sid}_audit.json', audit)
        if number == startup_budget:
            torch.save(dict(model=model.state_dict(), model_config=model.cfg.to_dict(),
                            source_center=model.source_center.cpu(),
                            source_scale=model.source_scale.cpu(),
                            revealed_shots=[r['shot'] for r in rows],
                            threshold=threshold, parent_hash=parent,
                            next_hash=O.state_hash(model)),
                       output_dir/f'snapshot_{startup_budget}.pt')
        print(json.dumps(dict(arm=arm, index=number, shot=sid, phase=phase,
                              tpr=audit['cumulative']['tpr'],
                              fpr=audit['cumulative']['fpr'],
                              guard=guard['reason'])), flush=True)
    result = dict(deployed=summarize_strict(rows, threshold),
                  same_parent_no_fast_shadow=summarize_strict(shadows, threshold),
                  posthoc_diagnostic_operating_points={},
                  parent_hash=parent, final_hash=O.state_hash(model),
                  accepted_updates=accepted_updates, rejected_updates=rejected_updates,
                  startup_budget=startup_budget, strict_east_lead='>30ms',
                  guard_proxy='prior revealed risk windows; not a shot guarantee',
                  wall_s=time.perf_counter()-started, hard_realtime_guarantee=False)
    O.write_json(output_dir/'metrics.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--startup-budget', type=int, default=20)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--startup-lr', type=float, default=.001)
    parser.add_argument('--later-lr', type=float, default=.00025)
    parser.add_argument('--inter-steps', type=int, default=1)
    parser.add_argument('--unguarded', action='store_true')
    parser.add_argument('--control', choices=('learned', 'fixed'))
    parser.add_argument('--allow-unqualified-source', action='store_true',
                        help='explicit diagnostic override of performance gate, never artifact integrity')
    args = parser.parse_args(argv)
    if args.startup_budget < 1 or args.limit < 0 or args.inter_steps < 1 or any(
            not np.isfinite(v) or v <= 0 for v in (args.startup_lr, args.later_lr)):
        parser.error('positive startup budget, learning rates, steps and nonnegative limit required')
    if len(set(args.arms)) != len(args.arms):
        parser.error('duplicate arms')
    if args.output.exists():
        raise FileExistsError('commissioning output already exists')
    qualification = qualify_source(args.checkpoint)
    if not qualification['qualified'] and not args.allow_unqualified_source:
        raise ValueError(f'J-TEXT source gate failed: {qualification["reasons"]}; no EAST signals loaded')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for real commissioning replay')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    mode = checkpoint['config'].get('mode')
    control = args.control or {'joint': 'fixed', 'meta': 'learned'}.get(mode)
    if control is None:
        parser.error('unknown source mode; specify --control')
    threshold = float(checkpoint['thresholds']['east'])
    if not np.isfinite(threshold):
        raise ValueError('nonfinite fixed source threshold')
    model = RiskAlignedModel(ModelConfig(**checkpoint['model_config']),
                             checkpoint['source_center'], checkpoint['source_scale']).cuda()
    model.load_state_dict(checkpoint['model'])
    torch.set_num_threads(4)
    source_parent = O.state_hash(model)
    ids = cohort_ids('EAST')
    full_order = data.build_order('EAST', ids, order='low_to_high')
    roster = [int(r['shot']) for r in full_order]
    if len(roster) != len(ids) or len(roster) != len(set(roster)) or set(roster) != set(ids):
        raise ValueError('commissioning order does not preserve the full EAST roster')
    order = full_order[:args.limit] if args.limit else full_order
    args.output.mkdir(parents=True, exist_ok=False)
    # Raw Bt proxy is not assigned a verified Tesla unit.
    O.write_json(args.output/'order.json', dict(full_order=full_order,
                                               selected_prefix=order,
                                               method='data.build_order:low_to_high'))
    evidence = O.run_provenance(checkpoint, args.limit)
    evidence['code_sha256'].update({name: O.file_hash(Path(__file__).with_name(name))
                                    for name in ('commissioning.py', 'stage_metrics.py', 'source_gate.py')})
    O.write_json(args.output/'config.json', dict(checkpoint=str(args.checkpoint),
                checkpoint_sha256=O.file_hash(args.checkpoint), arms=args.arms,
                domain='EAST', threshold=threshold, source_mode=mode,
                control=control, limit=args.limit, startup_budget=args.startup_budget,
                startup_lr=args.startup_lr, later_lr=args.later_lr,
                inter_steps=args.inter_steps, unguarded=args.unguarded,
                source_qualification=qualification,
                allow_unqualified_source=args.allow_unqualified_source,
                historical_guard='prior revealed risk windows; no shot-level guarantee',
                strict_east_lead='round lead_ms to 9 decimals, then >30ms',
                outcomes='automatic reveal after fsynced score seal and receipt; no query budget',
                source_parent_hash=source_parent,
                hard_realtime_guarantee=False, **evidence))
    results = {}
    for arm in args.arms:
        model.load_state_dict(checkpoint['model'])
        if O.state_hash(model) != source_parent:
            raise RuntimeError('source parent changed between arms')
        results[arm] = run_arm(model, order=order, arm=arm, threshold=threshold,
                               output_dir=args.output/arm,
                               startup_budget=args.startup_budget, device='cuda',
                               guarded=not args.unguarded,
                               startup_lr=args.startup_lr, later_lr=args.later_lr,
                               inter_steps=args.inter_steps, control=control)
    stages = stage_summaries(results, full_order, args.startup_budget)
    O.write_json(args.output/'summary.json', results)
    O.write_json(args.output/'stage_metrics.json', stages)
    return results


if __name__ == '__main__':
    main()
