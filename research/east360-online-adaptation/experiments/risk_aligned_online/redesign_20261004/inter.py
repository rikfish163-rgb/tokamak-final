"""Sealed EAST feedback at the source-trained internal token adaptation site.

Only conditional persistent parameters are stepped. Historical endpoint checks
are bounded samples, not a replay of the operational intra-shot alarm process.
"""
from collections import OrderedDict
import math

import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data


MAX_REPLAY_SHOTS = 32
TRAIN_WINDOWS = 64
GUARD_PER_SHOT = 4
ANCHOR_REGULARIZATION = 1e-4
MAX_RISK_MARGIN_DEGRADATION = .02
MAX_HISTORY_BCE_DEGRADATION = .01
NUMERIC_TOLERANCE = 1e-6
COUNT_KEYS = ('positive', 'negative', 'excluded_late_positive',
              'excluded_early_negative')


def _pools(shot):
    """Check revealed metadata without ever loading an outcome ourselves."""
    required = {'shot', 'device', 'values', 'available', 'times', 'y', 'valid',
                'is_disrupt', 'down', 'horizon', '_sealed_scores'}
    if not required <= shot.keys():
        raise ValueError('postshot_update requires sealed scores and a revealed outcome')
    if shot['device'] != 'EAST' or shot['horizon'] != 125.:
        raise ValueError('only revealed EAST shots with fixed 125 ms labels are allowed')
    values, available = np.asarray(shot['values']), np.asarray(shot['available'])
    times, labels = np.asarray(shot['times'], float), np.asarray(shot['y'], float)
    valid = np.asarray(shot['valid'])
    n = len(times)
    if (values.shape != (n, 23) or available.shape != values.shape
            or available.dtype != bool or labels.shape != (n,)
            or valid.shape != (n,) or valid.dtype != bool):
        raise ValueError('revealed shot must have flat23 values/masks and aligned labels')
    if (not np.isfinite(times).all() or (np.diff(times) < 0).any()
            or not np.isfinite(labels).all() or not np.isin(labels, [0., 1.]).all()):
        raise ValueError('finite chronological times and binary revealed labels required')
    scores = np.asarray(shot['_sealed_scores'])
    if scores.shape != (n,) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('finite allpoint sealed probabilities required')
    if shot['is_disrupt']:
        if shot['down'] is None or not np.isfinite(shot['down']):
            raise ValueError('revealed disruption requires a finite endpoint outcome')
        raw_lead = (float(shot['down']) - times) * 1000.
        expected_valid = raw_lead >= -NUMERIC_TOLERANCE
        expected_y = ((raw_lead <= 125. + NUMERIC_TOLERANCE) & expected_valid)
        if (not np.array_equal(valid, expected_valid)
                or not np.array_equal(labels, expected_y.astype(float))):
            raise ValueError('revealed labels differ from the fixed 125 ms contract')
        # Remove floating-point subtraction dust at exact 30/125 ms boundaries;
        # this is not a relaxed >30 ms operational definition.
        lead = np.round(raw_lead, 9)
        positive = np.flatnonzero(valid & (labels == 1) & (lead > 30.) & (lead <= 125.))
        negative = np.empty(0, dtype=int)
        late = int(np.count_nonzero(valid & (labels == 1) & (lead <= 30.)))
        early = int(np.count_nonzero(valid & (labels == 0)))
    else:
        if np.any(labels != 0) or not valid.all() or shot['down'] is not None:
            raise ValueError('revealed safe outcome requires valid zero labels and no DownTime')
        positive, negative = np.empty(0, dtype=int), np.flatnonzero(valid)
        late = early = 0
    counts = dict(positive=len(positive), negative=len(negative),
                  excluded_late_positive=late, excluded_early_negative=early)
    return positive, negative, counts


def _sealed_category(shot, threshold):
    """Classify an already revealed/sealed shot by the actual first alarm.

    The 125 ms *training* window is not a maximum accepted alarm lead. An
    alarm after the revealed terminal time cannot rescue a disruptive shot.
    Validation of this receipt is performed by _pools before this function.
    """
    hits = np.flatnonzero(np.asarray(shot['valid']) &
                          (np.asarray(shot['_sealed_scores']) >= threshold))
    if not shot['is_disrupt']:
        return 'FP' if len(hits) else 'TN'
    if not len(hits):
        return 'FN'
    lead = round((float(shot['down']) - float(shot['times'][hits[0]])) * 1000., 9)
    return 'TP' if lead > 30. else 'TARDY'


def _select_guard(shot, positive, negative, threshold, policy='margin'):
    """Select at most four identities once from an already sealed trajectory."""
    eligible = positive if shot['is_disrupt'] else negative
    if policy == 'crossings' and shot['is_disrupt']:
        lead = np.round((float(shot['down']) - np.asarray(shot['times'])) * 1000., 9)
        eligible = np.flatnonzero(np.asarray(shot['valid']) & (lead > 30.))
    if not len(eligible):
        return dict(shot=int(shot['shot']), disruptive=bool(shot['is_disrupt']),
                    endpoints=[], error=None)
    scores = np.asarray(shot['_sealed_scores'])
    peak = int(eligible[np.argmax(scores[eligible])])
    nearest = int(eligible[np.argmin(np.abs(scores[eligible] - threshold))])
    ends = list(dict.fromkeys((peak, nearest, int(eligible[0]), int(eligible[-1]))))
    sealed_hit = bool(scores[peak] >= threshold)
    error = ('timely_miss' if not sealed_hit else None) if shot['is_disrupt'] else (
        'safe_false_alarm' if sealed_hit else None)
    return dict(shot=int(shot['shot']), disruptive=bool(shot['is_disrupt']),
                endpoints=[dict(end=end, sealed_score=float(scores[end]),
                                is_sealed_peak=end == peak) for end in ends],
                error=dict(shot=int(shot['shot']), end=peak, kind=error) if error else None)


def _prepare_records(model, shot, replay, threshold):
    history_all = list(replay)
    ids = [int(s['shot']) for s in history_all]
    if int(shot['shot']) in ids or len(set(ids)) != len(ids):
        raise ValueError('replay contains current shot or duplicate revealed identities')
    history = history_all[-MAX_REPLAY_SHOTS:]
    previous = getattr(model, '_inter_threshold', threshold)
    if threshold != previous:
        raise ValueError('the fixed threshold cannot change across postshot updates')
    guard_policy = getattr(model, 'guard_policy', 'margin')
    if guard_policy not in ('margin', 'crossings'):
        raise ValueError('unknown history guard policy')
    if guard_policy != getattr(model, '_inter_guard_policy', guard_policy):
        raise ValueError('history guard policy cannot change within a stream')
    # Validate *all* used records before attaching any adaptation metadata.
    pools = [_pools(s) for s in [*history, shot]]
    cache = OrderedDict(getattr(model, '_inter_guard_cache', {}))
    cache = OrderedDict((sid, entry) for sid, entry in cache.items() if sid in ids)
    records = []
    for s, (positive, negative, counts) in zip([*history, shot], pools):
        sid = int(s['shot'])
        if sid not in cache:
            cache[sid] = _select_guard(s, positive, negative, threshold, guard_policy)
        records.append(dict(shot=s, positive=positive, negative=negative,
                            counts=counts, guard=cache[sid], current=s is shot))
    # Keep only identities (not another full-shot replay) for the last 31 past
    # shots plus the just-revealed current shot. The caller appends after return.
    retained = [int(r['shot']['shot']) for r in records][-MAX_REPLAY_SHOTS:]
    model._inter_guard_cache = OrderedDict((sid, cache[sid]) for sid in retained)
    model._inter_threshold = threshold
    model._inter_guard_policy = guard_policy
    return records, dict(provided=len(history_all), used=len(history),
                         discarded=max(0, len(history_all) - MAX_REPLAY_SHOTS),
                         shot_ids=[int(s['shot']) for s in history])


def _sample(records, rng):
    pools = {label: {'current': [], 'history': []} for label in (0, 1)}
    counts = {origin: {key: 0 for key in COUNT_KEYS} for origin in ('current', 'history')}
    for record in records:
        origin = 'current' if record['current'] else 'history'
        for key in COUNT_KEYS:
            counts[origin][key] += record['counts'][key]
        for label, key in ((1, 'positive'), (0, 'negative')):
            pools[label][origin].extend((record['shot'], int(end), label, origin)
                                        for end in record[key])
    counts['total'] = {key: sum(counts[o][key] for o in ('current', 'history')) for key in COUNT_KEYS}
    present = [label for label in (1, 0) if any(pools[label].values())]
    strategy = 'stratified_50_50' if len(present) == 2 else 'single_class' if present else 'empty'
    audit = dict(strategy=strategy, positive=0, negative=0, current_positive=0,
                 history_positive=0, current_negative=0, history_negative=0,
                 unique_windows=0, replacement_draws=0)
    selected = []
    for label in present:
        slots = TRAIN_WINDOWS // len(present)
        origins = [origin for origin in ('current', 'history') if pools[label][origin]]
        for origin in origins:
            n = slots // len(origins)
            pool = pools[label][origin]
            indices = rng.choice(len(pool), size=n, replace=len(pool) < n)
            selected.extend(pool[int(i)] for i in indices)
            key = 'positive' if label else 'negative'
            audit[key] += n
            audit[f'{origin}_{key}'] += n
    audit['unique_windows'] = len({(int(s['shot']), end) for s, end, _, _ in selected})
    audit['replacement_draws'] = len(selected) - audit['unique_windows']
    return selected, counts, audit


def _windows(rows, model, device, dtype, data_api):
    if not rows:
        return None
    windows = [data_api.window(s, end, model.cfg.window) for s, end, *_ in rows]
    x = torch.as_tensor(np.stack([w[0] for w in windows]), dtype=dtype, device=device)
    available = torch.as_tensor(np.stack([w[1] for w in windows]), dtype=torch.bool, device=device)
    y = torch.tensor([row[2] for row in rows], dtype=dtype, device=device)
    return x, available, y


def _logits(model, batch):
    x, available, _ = batch
    fast = torch.zeros_like(model.initial_fast(len(x)), device=x.device)
    result = model(x, available, fast=fast)
    logits = result['logit'] if isinstance(result, dict) else result[0]
    if logits.shape != (len(x),):
        raise ValueError('ContextModel risk logit must be one scalar per window')
    if not torch.isfinite(logits).all():
        raise FloatingPointError('nonfinite_logits')
    return logits


def _balanced_bce(logits, labels):
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    terms = [losses[labels == label].mean() for label in (0., 1.) if (labels == label).any()]
    return torch.stack(terms).mean()


def _norm(tensors):
    return math.sqrt(sum(float(t.detach().double().square().sum()) for t in tensors))


def _audit_norm(tensors):
    norm = _norm(tensors)
    return norm if math.isfinite(norm) else None


def _objective(model, batch, params, source_indices=None, source_audit=None):
    logits = _logits(model, batch)
    bce = _balanced_bce(logits, batch[2])
    penalty = sum(p.square().sum() for p in params) / sum(p.numel() for p in params)
    objective = bce + ANCHOR_REGULARIZATION * penalty
    bank = getattr(model, 'source_bank', None)
    if bank is not None:
        from .source_replay import source_loss
        source, detail = source_loss(model, bank, batch[0].device, None,
                                     indices=source_indices)
        if source_audit is not None:
            source_audit.update(detail)
        objective = objective + getattr(model, 'source_replay_weight', 1.) * source
    return objective, bce


def _guard_rows(records):
    rows, details, empty = [], [], []
    for record in records[:-1]:  # Never protect a hypothetical future/current prediction.
        s, entry = record['shot'], record['guard']
        if s['is_disrupt'] and not entry['endpoints']:
            empty.append(int(s['shot']))
        for endpoint in entry['endpoints']:
            label = int(s['is_disrupt'])
            rows.append((s, endpoint['end'], label, 'history'))
            details.append(dict(shot=int(s['shot']), label=label, **endpoint,
                                before=None, candidate=None))
    audit = dict(scope='sampled_endpoints_fast0', full_shot_operational=False,
                 selection='cached_sealed_scores', windows=len(rows),
                 shot_ids=list(dict.fromkeys(d['shot'] for d in details)),
                 no_timely_endpoint_shot_ids=empty, endpoints=details,
                 before_bce=None, candidate_bce=None,
                 max_margin_degradation=None,
                 risk_margin_limit=MAX_RISK_MARGIN_DEGRADATION,
                 bce_degradation_limit=MAX_HISTORY_BCE_DEGRADATION)
    return rows, audit


def _assess_guard(model, batch, guard, baseline=None):
    if batch is None:
        return None, []
    with torch.no_grad():
        logits = _logits(model, batch)
        risk = logits.sigmoid().cpu().numpy()
        bce = float(_balanced_bce(logits, batch[2]))
    key = 'before' if baseline is None else 'candidate'
    guard[f'{key}_bce'] = bce
    for row, score in zip(guard['endpoints'], risk):
        row[key] = float(score)
    if baseline is None:
        return risk, []
    labels = batch[2].cpu().numpy()
    harmful = np.where(labels == 1, baseline - risk, risk - baseline)
    guard['max_margin_degradation'] = max(0., float(harmful.max()))
    reasons = []
    if (guard.get('policy', 'margin') == 'margin'
            and (harmful > MAX_RISK_MARGIN_DEGRADATION + NUMERIC_TOLERANCE).any()):
        reasons.append('historical_margin_degradation')
    if bce > guard['before_bce'] + MAX_HISTORY_BCE_DEGRADATION:
        reasons.append('historical_bce_degraded')
    for sid in guard['shot_ids']:
        indices = [i for i, row in enumerate(guard['endpoints']) if row['shot'] == sid]
        before_peak, after_peak = float(baseline[indices].max()), float(risk[indices].max())
        threshold = guard['threshold']
        if labels[indices[0]] == 1 and before_peak >= threshold > after_peak:
            reasons.append('timely_tp_lost')
        if labels[indices[0]] == 0 and before_peak < threshold <= after_peak:
            reasons.append('new_safe_false_alarm')
    return risk, list(dict.fromkeys(reasons))


def postshot_update(model, shot, replay, device, rng, threshold, lr=.03,
                    radius=1., steps=4, data_api=simple_data):
    """Propose a persistent-only neural update after a shot is sealed/revealed.

    ``replay`` is caller-owned previous revealed EAST shots (normally deque32).
    ``_sealed_scores`` is a trusted post-reveal receipt supplied by the runner,
    not proof of a durable seal in this helper. No outcome-reader is called.
    Training excludes early disruptive negatives and <=30 ms positives. Both
    available classes receive exactly 32/64 draws; an absent class is not made
    up. Historical checks use <=4 cached endpoint identities/shot at fast=0.
    ``radius`` bounds the global L2 norm of all persistent parameters from ZERO,
    not change from the previous shot. Any failed proposal is rolled back.
    """
    if (not all(math.isfinite(float(v)) for v in (threshold, lr, radius))
            or not 0 < threshold < 1 or lr < 0 or radius < 0
            or not isinstance(steps, int) or isinstance(steps, bool) or steps < 0):
        raise ValueError('finite fixed threshold, nonnegative lr/radius and integer steps required')
    update_policy = getattr(model, 'update_policy', 'all')
    if update_policy not in ('all', 'error_only'):
        raise ValueError('unknown postshot update policy')
    source_bank = getattr(model, 'source_bank', None)
    source_guard_policy = getattr(model, 'source_guard_policy', 'crossings')
    if source_guard_policy not in ('crossings', 'diagnostic'):
        raise ValueError('unknown source guard policy')
    backtracking = getattr(model, 'source_backtracking', 0)
    if (type(backtracking) is not int or not 0 <= backtracking <= 12
            or backtracking and source_bank is None):
        raise ValueError('source backtracking requires a bank and 0..12 integer halvings')
    if source_bank is not None:
        source_weight = getattr(model, 'source_replay_weight', 1.)
        if not math.isfinite(float(source_weight)) or source_weight <= 0:
            raise ValueError('a finite positive source replay weight is required')
    params = list(model.persistent_parameters())
    named = {id(p): name for name, p in model.named_parameters()}
    base_ids = {id(p) for p in model.base.parameters()}
    if (not params or len({id(p) for p in params}) != len(params)
            or any(id(p) not in named or id(p) in base_ids or not p.requires_grad for p in params)):
        raise ValueError('persistent_parameters must select only trainable internal parameters outside base')
    records, history = _prepare_records(model, shot, replay, float(threshold))
    policy = getattr(model, 'replay_policy', 'uniform')
    if policy == 'uniform':
        selected, counts, sampling = _sample(records, rng)
    elif policy == 'error_aware':
        from .error_replay import sample
        selected, counts, sampling = sample(records, rng)
    else:
        raise ValueError('unknown replay sampling policy')
    sampling['policy'] = policy
    guard_rows, guard = _guard_rows(records)
    guard['threshold'] = float(threshold)
    guard['policy'] = getattr(model, 'guard_policy', 'margin')
    category = _sealed_category(shot, float(threshold))
    before = [p.detach().clone() for p in params]
    audit = dict(accepted=False, reason=None, reasons=[], learned_neural_update=False,
                 update_policy=update_policy, sealed_category=category,
                 source_replay=(dict(weight=float(source_weight),
                    bank_sha256=source_bank['bank_sha256'], indices=None,
                    guard_policy=source_guard_policy,
                    before=None, candidate=None, guard=None)
                    if source_bank is not None else None),
                 pool_counts=counts, sampling=sampling, history=history, guard=guard,
                 scope=dict(parameter_names=[named[id(p)] for p in params],
                            parameter_count=sum(p.numel() for p in params),
                            fast='zero_temporary_delta', base_updated=False,
                            risk_score_correction_head=False),
                 drift=dict(anchor='zero', radius=float(radius), before=None,
                            candidate=None, after=None, step_l2=0., candidate_step_l2=None),
                 loss=dict(before_bce=None, candidate_bce=None, before_objective=None,
                           candidate_objective=None, regularization=ANCHOR_REGULARIZATION),
                 steps_completed=0, gradient_l2=None,
                 proposal_scale=1., backtracking=[],
                 retained_error_identities=[e['error'] for e in model._inter_guard_cache.values()
                                            if e['error'] is not None])
    modes = [(module, module.training) for module in model.modules()]
    reasons = []
    try:
        model.eval()
        if any(not torch.isfinite(p).all() for p in params):
            raise FloatingPointError('nonfinite_initial_parameters')
        audit['drift']['before'] = _audit_norm(before)
        if audit['drift']['before'] is None:
            raise FloatingPointError('nonfinite_initial_norm')
        if audit['drift']['before'] > radius + NUMERIC_TOLERANCE:
            reasons.append('memory_outside_zero_radius')
        elif update_policy == 'error_only' and category in ('TP', 'TN'):
            # Correct examples are retained in the caller-owned replay and
            # cached guard, but do not force one-class confidence training.
            # This is a deliberate response to measured safe-prefix collapse,
            # not a bypass for an unavailable label or failed computation.
            reasons.append('correct_shot_experience_only')
        elif not selected:
            reasons.append('no_primary_examples')
        elif steps == 0 or lr == 0:
            reasons.append('no_update_steps')
        else:
            batch = _windows(selected, model, device, params[0].dtype, data_api)
            guard_batch = _windows(guard_rows, model, device, params[0].dtype, data_api)
            source_indices = None
            source_before, source_candidate = {}, {}
            if source_bank is not None:
                from .source_replay import sample_source, evaluate_source_guard
                # Keep target replay draws unchanged in the source-memory
                # ablation; an extra bank must not consume the target RNG.
                if not hasattr(model, 'source_rng'):
                    model.source_rng = np.random.default_rng(131)
                source_indices = sample_source(source_bank, model.source_rng, batch_size=64)
                audit['source_replay']['indices'] = source_indices.tolist()
            with torch.no_grad():
                objective, bce = _objective(model, batch, params, source_indices, source_before)
            if source_bank is not None:
                audit['source_replay']['before'] = source_before
            if not torch.isfinite(objective):
                raise FloatingPointError('nonfinite_initial_objective')
            audit['loss'].update(before_bce=float(bce), before_objective=float(objective))
            baseline, _ = _assess_guard(model, guard_batch, guard)
            for _ in range(steps):
                objective, _ = _objective(model, batch, params, source_indices)
                if not torch.isfinite(objective):
                    raise FloatingPointError('nonfinite_objective')
                if not objective.requires_grad:
                    reasons.append('no_neural_gradient')
                    break
                gradients = torch.autograd.grad(objective, params, allow_unused=True)
                gradients = [torch.zeros_like(p) if g is None else g for p, g in zip(params, gradients)]
                if any(not torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError('nonfinite_gradient')
                gradient_norm = _norm(gradients)
                if not math.isfinite(gradient_norm):
                    raise FloatingPointError('nonfinite_gradient_norm')
                audit['gradient_l2'] = gradient_norm
                with torch.no_grad():
                    for p, g in zip(params, gradients):
                        p.add_(g, alpha=-lr * min(1., 1. / max(gradient_norm, 1e-12)))
                    if any(not torch.isfinite(p).all() for p in params):
                        raise FloatingPointError('nonfinite_candidate_parameters')
                    norm = _norm(params)
                    if not math.isfinite(norm):
                        raise FloatingPointError('nonfinite_candidate_norm')
                    if norm > radius:
                        for p in params:
                            p.mul_(radius / norm)
                audit['steps_completed'] += 1
            proposal = [p.detach().clone() for p in params]
            gradient_reasons = list(reasons)
            # Feasibility search on the SAME proposed direction, using only
            # source knowledge and already-revealed history. No future labels,
            # alarm tuning or hyperparameter selection on later performance.
            # Convex interpolation retains the existing global-radius bound.
            for halving in range(backtracking + 1):
                scale = 2. ** -halving
                if halving:
                    with torch.no_grad():
                        for p, original, proposed in zip(params, before, proposal):
                            p.copy_(original + scale * (proposed - original))
                reasons = list(gradient_reasons)
                source_candidate = {}
                with torch.no_grad():
                    objective, bce = _objective(model, batch, params, source_indices, source_candidate)
                if source_bank is not None:
                    audit['source_replay']['candidate'] = source_candidate
                    source_guard = evaluate_source_guard(model, source_bank, device,
                        threshold=float(getattr(model, 'source_threshold', threshold)))
                    audit['source_replay']['guard'] = source_guard
                    # This explicit mechanism ablation retains source BCE and
                    # records source crossings but removes their hard veto.
                    # Target history/loss/radius checks still apply. It tests
                    # whether source-point protection prevents EAST adaptation;
                    # it is not a production safety policy or error workaround.
                    if not source_guard['passed'] and source_guard_policy == 'crossings':
                        reasons.append(source_guard['reason'])
                if not torch.isfinite(objective):
                    raise FloatingPointError('nonfinite_candidate_objective')
                audit['loss'].update(candidate_bce=float(bce), candidate_objective=float(objective))
                _, guard_reasons = _assess_guard(model, guard_batch, guard, baseline)
                reasons.extend(guard_reasons)
                if float(bce) > audit['loss']['before_bce'] + NUMERIC_TOLERANCE:
                    reasons.append('candidate_bce_degraded')
                if float(objective) > audit['loss']['before_objective'] + NUMERIC_TOLERANCE:
                    reasons.append('candidate_objective_degraded')
                if _norm([p.detach() - original for p, original in zip(params, before)]) == 0:
                    reasons.append('no_parameter_change')
                audit['proposal_scale'] = scale
                if backtracking:
                    audit['backtracking'].append(dict(scale=scale,
                        target_bce=float(bce), objective=float(objective),
                        reasons=list(dict.fromkeys(reasons)),
                        source_guard=audit['source_replay']['guard']))
                if not reasons or gradient_reasons:
                    break
    except FloatingPointError as error:
        reasons.append(str(error))
    except Exception:
        # Unexpected contract/runtime errors also cannot leave a partial update.
        with torch.no_grad():
            for p, original in zip(params, before):
                p.copy_(original)
        raise
    finally:
        for module, training in modes:
            module.training = training
    # Capture attempted finite drift BEFORE rollback, including failures that
    # occur after a partial step. Unavailable nonfinite metrics serialize as null.
    audit['drift']['candidate'] = _audit_norm(params)
    audit['drift']['candidate_step_l2'] = _audit_norm(
        [p.detach() - original for p, original in zip(params, before)])
    if reasons:
        with torch.no_grad():
            for p, original in zip(params, before):
                p.copy_(original)
    audit['reasons'] = list(dict.fromkeys(reasons))
    audit['accepted'] = not reasons
    audit['reason'] = reasons[0] if reasons else 'accepted'
    audit['learned_neural_update'] = audit['accepted']
    audit['drift']['after'] = _audit_norm(params)
    audit['drift']['step_l2'] = _audit_norm([p.detach() - original for p, original in zip(params, before)])
    return audit
