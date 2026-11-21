"""Two early decisions with full historical costs, not random event credit.

The source FFN direction is fixed. Each choice controls 20 valid .02/.08
events, then holds state. Learned rules use already-ended EAST shots only;
this is inter-trained intra adaptation, not source-only unsupervised TTT.
"""
import numpy as np
import torch

from .context import ACTIONS, advance_q, segment_features


def macro_plan(signals, center, scale, *, rule=None, pair=None, arm='H'):
    if signals['device'] != 'EAST' or arm not in ('H', 'P') or (rule is None) == (pair is None):
        raise ValueError('EAST H/P with exactly one rule or fixed pair required')
    if rule is not None and (rule.shape != (3,117) or not torch.isfinite(rule).all()):
        raise ValueError('finite three-cost 117-dimensional rule required')
    if pair is not None and (len(pair) != 2 or any(type(a) is not int or a not in ACTIONS for a in pair)):
        raise ValueError('two fixed actions required')
    x, mask = np.asarray(signals['values']), np.asarray(signals['available'])
    if x.shape != mask.shape or x.ndim != 2 or x.shape[1] != 23 or mask.dtype != bool:
        raise ValueError('flat23 values and boolean mask required')
    current = 0.; curve = np.empty(len(x), np.float64)
    decisions, events, first_signal, start = [], [], None, 0
    center, scale = center.cpu().double(), scale.cpu().double()
    frozen_rule = rule.detach().cpu().double() if rule is not None else None
    for origin in range(0, len(x), 26):
        for offset, segment in ((5, 'early'), (25, 'late')):
            end = origin + offset
            if end >= len(x) or len(events) >= 40:
                continue
            curve[start:end+1] = current; start = end+1
            first = origin+1 if segment == 'early' else origin+5
            truth = torch.as_tensor(x[first:end+1].copy(), dtype=torch.float64)
            observed = torch.as_tensor(mask[first:end+1].copy(), dtype=torch.bool)
            features = segment_features(truth, observed, center, scale, segment)
            if not features[69:92].any():
                continue  # Explicit empty-event contract: no macro count or state change.
            if first_signal is None:
                first_signal = features[:69].clone()
            if arm == 'P':
                features[:69] = first_signal
            if len(events) in (0, 20):
                context = torch.cat((features, features.new_tensor([current, 1.])))
                slot = len(events)//20
                action = pair[slot] if pair is not None else ACTIONS[int((frozen_rule@context).argmin())]
                decisions.append(dict(slot=slot, end=end, effect=end+1 if end+1 < len(x) else None,
                                      context=context.tolist(), action=action))
            previous = current
            current = advance_q(current, action, segment)
            events.append(dict(end=end, effect=end+1 if end+1 < len(x) else None,
                               segment=segment, action=action, q_before=previous, q_after=current))
        if len(events) == 40:
            break
    curve[start:] = current
    return curve, decisions, events


def _context(table, a=None, arm='H'):
    c = table['context_1'] if a is None else table['context_2'][a]
    if c is None:
        return None
    c = torch.tensor(c, dtype=torch.float64)
    if arm == 'P' and a is not None:
        c[:69] = torch.tensor(table['context_1'][:69], dtype=c.dtype)
    return c


def choose_pair(table, rule, arm='H'):
    c = _context(table)
    if c is None:
        return 0, 0
    a = int((rule@c).argmin())
    c2 = _context(table, a, arm)
    b = int((rule@c2).argmin()) if c2 is not None else 0
    return a, b


def credit_examples(table, rule):
    """Actual student continuation, not minimum-over-future oracle costs."""
    c = _context(table)
    if c is None:
        return []
    costs = np.asarray(table['costs'], float)
    a, _ = choose_pair(table, rule)
    q = []
    for action in range(3):
        c2 = _context(table, action)
        b = int((rule@c2).argmin()) if c2 is not None else 0
        q.append(costs[action,b])
    q = np.asarray(q)
    examples = [(c, q-q.mean())]
    c2 = _context(table,a)
    if c2 is not None:
        examples.append((c2, costs[a]-costs[a].mean()))
    return examples


def class_errors(tables, rule):
    """Actual joint decisions: [safe FP count, disrupted non-timely count]."""
    errors = [0,0]
    for t in tables:
        a,b = choose_pair(t, rule)
        errors[int(t['disruptive'])] += int(t['costs'][a][b])
    return errors


def fit_rule(tables, initial, *, conservative=False):
    """At most three rounds; optional *historical* joint-cost acceptance.

    The conservative experiment was introduced after observing the original
    three-round failure. It never promises future monotonicity: only already
    completed historical shots determine replacement. Equal-cost proposals
    retain the exact old rule, and the first rejection ends fitting rather
    than retrying the same deterministic proposal or choosing a best epoch.
    """
    counts = [sum(int(t['disruptive']) == k for t in tables) for k in (0,1)]
    if not all(counts):
        raise ValueError('both historical classes required, do not substitute synthetic labels')
    rule = initial.detach().cpu().double().clone(); logs = []
    for iteration in range(3):
        xs, targets, weights = [], [], []
        for table in tables:
            examples = credit_examples(table, rule)
            for c, target in examples:
                xs.append(c); targets.append(torch.tensor(target, dtype=rule.dtype))
                weights.append(1/(2*counts[int(table['disruptive'])]*len(examples)*3))
        if not xs:
            raise ValueError('no historical effective decisions')
        x, y, w = torch.stack(xs), torch.stack(targets), torch.tensor(weights, dtype=rule.dtype)
        lhs = x.T@(w[:,None]*x) + 1e-3*torch.eye(117, dtype=rule.dtype)
        proposal = torch.linalg.solve(lhs, x.T@(w[:,None]*y)).T
        if not torch.isfinite(proposal).all():
            raise FloatingPointError('nonfinite historical cost fit')
        old_errors, new_errors = class_errors(tables,rule), class_errors(tables,proposal)
        improves = all(a<=b for a,b in zip(new_errors,old_errors)) and new_errors != old_errors
        accepted = not conservative or improves
        logs.append(dict(iteration=iteration+1, examples=len(xs),
            non_tied_examples=int((y.max(1).values-y.min(1).values > 0).sum()),
            weighted_squared_error=float((w[:,None]*(x@proposal.T-y).square()).sum()),
            rule_norm=float(proposal.norm()), current_class_errors=old_errors,
            proposal_class_errors=new_errors, accepted=accepted))
        if not accepted:
            break
        rule = proposal
    return rule, logs
