"""Posthoc matched-step comparison of Adam and its momentum direction.

The real-gradient arms never use query labels in their updates. The oracle
does, and is only a mechanism diagnostic, not an online policy or an upper
bound. Nothing in this file changes the deployed/default Adam update.
"""

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import analyze_risk_oracle_direction_20261108 as d
from risk_feature_intra_20261107 import FeatureIntra
from risk_meta_adam_20261108 import adam_step
from risk_meta_step_20261107 import MetaOrderedTrajectory


OUT = d.FIT / 'east_momentum_direction_cpu.json'
RULES = ('adam', 'momentum_direction')


def local_updates(model, episodes, query, available, label, *, feedback, rule):
    """Run four fresh-shot updates, recording actual steps and moment history."""
    from risk_meta_adam_20261108 import direction_step

    if rule not in RULES or feedback not in ('signal', 'oracle') or len(episodes) != 4:
        raise ValueError('two declared rules, signal/oracle feedback and exactly four steps required')
    before = d.output_at_query(model, query, available, label)
    history = ()
    records = []
    for episode in episodes:
        if feedback == 'signal':
            task = model.signal_loss(episode)
        else:
            logit = model.risk(query, available)['logit']
            task = F.binary_cross_entropy_with_logits(logit, label)
        prior = (model.weight-model.initial_weight).square().sum()
        objective = task + d.j.PRIOR_WEIGHT * prior
        if not torch.isfinite(objective):
            raise FloatingPointError('nonfinite local objective')
        gradient, = torch.autograd.grad(objective, model.weight)
        step = (adam_step if rule == 'adam' else direction_step)(model.weight, gradient, history,
                                                                 lr=d.j.INTRA_LR)
        change = step['weight']-model.weight
        actual_length = float(change.norm().detach())
        if rule == 'momentum_direction':
            adam_change = step['adam_weight']-model.weight
            target_length = float(adam_change.norm().detach())
            angle = d.cosine(change, adam_change)
        else:
            target_length, angle = actual_length, 1.0
        d.need(abs(actual_length-target_length) <= 2e-7, 'step norm matching failed')
        records.append(dict(objective=float(objective.detach()), task_loss=float(task.detach()),
                            prior=float(prior.detach()), gradient_norm=float(gradient.norm()),
                            step_norm=actual_length, adam_candidate_step_norm=target_length,
                            step_length_error=abs(actual_length-target_length),
                            actual_vs_adam_direction_cosine=angle))
        with torch.no_grad():
            model.weight.copy_(step['weight'])
        history = tuple(g.detach() for g in step['history'])
    after = d.output_at_query(model, query, available, label)
    positive = bool(label.item())
    gap = d.THRESHOLD-before['probability'] if positive else before['probability']-d.THRESHOLD
    progress = (after['probability']-before['probability'] if positive
                else before['probability']-after['probability'])
    return dict(before=before, after=after, steps=records,
                local_cross=(after['probability'] >= d.THRESHOLD if positive
                             else after['probability'] < d.THRESHOLD),
                signed_gap_closed_fraction=progress/gap if gap > 0 else None,
                weight_displacement_norm=float((model.weight-model.initial_weight).norm().detach()))


def summarize(cases):
    result = {}
    names = [f'{arm}__{rule}' for arm in d.ARMS for rule in RULES]
    for group in ('all', 'tardy', 'miss', 'fp'):
        rows = [row for row in cases if group == 'all' or row['static_status'] == group]
        result[group] = {}
        for name in names:
            items = [row['arms'][name] for row in rows]
            result[group][name] = dict(n=len(items), local_cross=sum(v['local_cross'] for v in items),
                bce_improved=sum(v['after']['bce'] < v['before']['bce'] for v in items),
                bce_change=d.distribution([v['after']['bce']-v['before']['bce'] for v in items]),
                signed_gap_closed_fraction=d.distribution([v['signed_gap_closed_fraction'] for v in items]),
                weight_displacement_norm=d.distribution([v['weight_displacement_norm'] for v in items]))
    return result


def main():
    started = time.monotonic()
    d.configure_cpu()
    d.need(not OUT.exists(), f'refusing to overwrite {OUT}')
    old = d.j.r.read(d.OUTPUT)
    d.need(old['status'] == 'COMPLETE_CPU_POSTHOC_LOCAL_DIRECTION', 'completed same-case reference required')
    cfg, mapping, errors = d.load_inputs()
    previous = {row['shot']: row for row in old['cases']}
    excluded = {row['shot']: row for row in old['exclusions']}
    base, _ = d.j.r.load_parent(d.j.r.DEFAULT_PARENT)
    source_state = d.snapshot(base)
    decoders = {}
    for name in ('control', 'short'):
        decoder = MetaOrderedTrajectory(base.center, base.scale, dim=base.cfg.dim, scenarios=4).eval()
        decoder.load_state_dict(torch.load(d.checkpoint_files()[name], map_location='cpu', weights_only=True))
        decoders[name] = decoder.requires_grad_(False)
    decoder_states = {key: d.snapshot(value) for key, value in decoders.items()}
    records = []
    parity_error = 0.0
    with sdpa_kernel(SDPBackend.MATH):
        for row in errors:
            case, raw = d.choose_and_verify(row, cfg)
            shot, q = case['shot'], case['query_index']
            if shot in excluded:
                d.need(case == excluded[shot], 'prescribed exclusion changed')
                continue
            d.need(shot in previous and not case['exclusion_reasons'], 'case eligibility changed')
            prior = previous[shot]
            d.need(all(case[k] == prior[k] for k in case), 'prescribed case or peak changed')
            prefix = dict(raw, values=raw['values'][:q+1], available=raw['available'][:q+1], times=raw['times'][:q+1])
            signal = d.j.r.prepare_actual(prefix, case['rank'], mapping)
            d.need(not signal['available'][:, 2].any(), 'EAST MNM mask changed')
            episodes = [d.make_episode(signal, origin, q) for origin in prior['origins']]
            qx, qa = d.j.simple_data.window(signal, q, 128)
            qx, qa = torch.as_tensor(qx[None]), torch.as_tensor(qa[None])
            label = torch.tensor([float(case['native_y_at_query'])])
            arms = {}
            for arm in d.ARMS:
                key = 'short' if arm == 'SSL_short' else 'control'
                for rule in RULES:
                    model = FeatureIntra(base, decoders[key])
                    item = local_updates(model, episodes, qx, qa, label,
                                         feedback='oracle' if arm == 'oracle_4steps' else 'signal', rule=rule)
                    if rule == 'adam':
                        original = prior['arms'][arm]
                        for phase in ('before', 'after'):
                            for stat in ('probability', 'logit', 'bce'):
                                error = abs(item[phase][stat]-original[phase][stat])
                                parity_error = max(parity_error, error)
                                d.need(error <= 1e-5, 'native Adam reference did not reproduce')
                        d.need(item['local_cross'] == original['local_cross'], 'reference local crossing differs')
                    d.check_state(model.risk, source_state, excluded=(d.WEIGHT,))
                    d.check_state(model.decoder, decoder_states[key])
                    arms[f'{arm}__{rule}'] = item
            records.append(dict(**case, origins=prior['origins'], arms=arms))
    d.need(len(records) == len(previous), 'fixed same-case population incomplete')
    d.check_state(base, source_state)
    for key, model in decoders.items():
        d.check_state(model, decoder_states[key])
    result = dict(status='COMPLETE_CPU_POSTHOC_DIRECTION_GEOMETRY', seconds=time.monotonic()-started,
        reference=str(d.OUTPUT), command="PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rl_env/python -u analyze_risk_direction_geometry_20261108.py",
        contract=dict(rules=RULES, parameter=d.WEIGHT, steps=4, lr=d.j.INTRA_LR,
            prior_weight=d.j.PRIOR_WEIGHT, clip=1., threshold=d.THRESHOLD, no_hyperparameter_sweep=True,
            matched_length='own-path Adam candidate length, same clipped-gradient history; no cross-arm future reference',
            direction='unit accumulated first moment instead of per-coordinate RMS scaling',
            interpretation='same posthoc selected error peaks; oracle labelled; neither online result nor mathematical upper bound',
            signal_updates_no_query_labels=True, source_and_decoder_templates_unchanged=True,
            full_stream_metrics=False, independent_validation=False),
        verification=dict(eligible=len(records), excluded=len(excluded),
            maximum_native_adam_stat_error=parity_error,
            maximum_step_length_error=max(s['step_length_error'] for r in records for a in r['arms'].values() for s in a['steps'])),
        summary=summarize(records), cases=records, exclusions=old['exclusions'],
        source_files={str(p):p.read_text() for p in (Path(__file__), Path(__file__).with_name('risk_meta_adam_20261108.py'))})
    with OUT.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k:result[k] for k in ('status', 'seconds', 'verification', 'summary')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
