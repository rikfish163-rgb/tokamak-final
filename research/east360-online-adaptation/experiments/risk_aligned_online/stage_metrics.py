"""Pure posthoc commissioning analysis; no files, models, or policy decisions.

The nonnegative ``proxy`` is data.build_order's relative raw Ip/Bt ordering
quantity. It is not measured damage, energy, a physical DBC, or verified units.
The highest-proxy quartile uses the COMPLETE supplied manifest, including
unobserved shots. It is an offline diagnostic group, never online selection.
"""
from math import ceil, isfinite
from numbers import Integral, Real


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _number(value, name, nonnegative=False):
    if (isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value)
            or (nonnegative and value < 0)):
        raise ValueError(f'{name} must be finite' + (' and nonnegative' if nonnegative else ''))
    return float(value)


def _empty():
    return dict(n=0, TP=0, FN=0, FP=0, TN=0, early_over_3000=0,
                missed_proxy_sum=0., total_disrupt_proxy=0.)


def _add(counts, row):
    counts['n'] += 1
    if row['is_disrupt']:
        counts['TP' if row['timely'] else 'FN'] += 1
        counts['total_disrupt_proxy'] += row['proxy']
        if not row['timely']:
            counts['missed_proxy_sum'] += row['proxy']
        if row['alarm'] and row['lead_ms'] is not None and row['lead_ms'] > 3000:
            counts['early_over_3000'] += 1
    else:
        counts['FP' if row['alarm'] else 'TN'] += 1


def _finish(counts):
    result = dict(counts)
    nd = counts['TP'] + counts['FN']
    nn = counts['FP'] + counts['TN']
    tpr = counts['TP']/nd if nd else None
    fpr = counts['FP']/nn if nn else None
    result.update(n_disrupt=nd, n_safe=nn, TPR=tpr, FPR=fpr,
                  balanced_accuracy=(tpr+1-fpr)/2 if nd and nn else None,
                  J=tpr-fpr if nd and nn else None,
                  timely_accuracy=(counts['TP']+counts['TN'])/counts['n'] if counts['n'] else None,
                  normalized_missed_proxy=(counts['missed_proxy_sum']/counts['total_disrupt_proxy']
                                           if counts['total_disrupt_proxy'] > 0 else None))
    return result


def _segment(rows):
    counts = _empty()
    for row in rows:
        _add(counts, row)
    return _finish(counts)


def summarize_commissioning(predictions, order, startup_budget, block_size=20):
    """Summarize sealed decisions in manifest order without changing them.

    ``predictions`` must be an exact leading prefix of ``order`` (possibly the
    entire manifest); both require unique integer shot IDs. Each prediction
    supplies boolean is_disrupt/alarm/timely and lead_ms. A disruptive alarm
    requires finite nonnegative lead_ms; all other records require None.
    Timely MUST agree with alarm AND disruption AND lead_ms > 30 exactly.
    Inconsistency raises rather than repairing results.

    Startup is the first B observed positions, clipped at N. Startup cost sums
    ALL disruptive proxy, including timely hits; post-startup cost sums only
    missed/tardy disruptive proxy. Segment missed costs always count FN only.
    Rates with absent denominators are independently None; BA/J require both
    classes. Timely accuracy is defined for every nonempty segment.

    Highest quartile = top ceil(manifest_size/4) entries ranked by (proxy, shot),
    then reported in manifest order; ties are broken by shot ID. Metrics use
    only observed members, with no invented outcomes for unobserved members.
    Blocks partition observed positions starting at 1; the final block may be
    short. Cumulative rows have constant-size counts/rates and additive costs.
    All returned values are JSON-compatible; caller inputs are never mutated.
    """
    budget = _integer(startup_budget, 'startup_budget')
    size = _integer(block_size, 'block_size', 1)
    manifest = []
    seen = set()
    for item in order:
        sid = _integer(item['shot'], 'shot')
        if sid in seen:
            raise ValueError('duplicate manifest shot')
        seen.add(sid)
        manifest.append(dict(shot=sid, proxy=_number(item['proxy'], 'proxy', True)))
    rows = []
    seen = set()
    for index, prediction in enumerate(predictions):
        sid = _integer(prediction['shot'], 'shot')
        if sid in seen:
            raise ValueError('duplicate prediction shot')
        seen.add(sid)
        if index >= len(manifest) or manifest[index]['shot'] != sid:
            raise ValueError('prediction identity/order must match manifest prefix')
        flags = {}
        for name in ('is_disrupt', 'alarm', 'timely'):
            if not isinstance(prediction[name], bool):
                raise ValueError(f'{name} must be boolean')
            flags[name] = prediction[name]
        lead = prediction['lead_ms']
        if flags['is_disrupt'] and flags['alarm'] and lead is None:
            raise ValueError('disruptive alarm requires lead_ms evidence')
        if lead is not None:
            lead = _number(lead, 'lead_ms', nonnegative=True)
            if not flags['is_disrupt'] or not flags['alarm']:
                raise ValueError('lead_ms requires a disruptive alarm')
            if flags['timely'] != (lead > 30.):
                raise ValueError('timely conflicts with strict lead_ms > 30')
        if flags['timely'] and not (flags['alarm'] and flags['is_disrupt']):
            raise ValueError('timely requires a disruptive alarm')
        rows.append(dict(manifest[index], **flags, lead_ms=lead))

    split = min(budget, len(rows))
    startup = _segment(rows[:split])
    post = _segment(rows[split:])
    quartile_count = ceil(len(manifest)/4)
    ranked = sorted(manifest, key=lambda r: (r['proxy'], r['shot']), reverse=True)
    high_ids = {r['shot'] for r in ranked[:quartile_count]}
    cumulative = []
    counts = _empty()
    startup_cost = post_cost = 0.
    for index, row in enumerate(rows, 1):
        _add(counts, row)
        if row['is_disrupt']:
            if index <= budget:
                startup_cost += row['proxy']
            elif not row['timely']:
                post_cost += row['proxy']
        cumulative.append(dict(index=index, shot=row['shot'], proxy=row['proxy'],
                               **_finish(counts), startup_cost_proxy=startup_cost,
                               post_startup_missed_cost_proxy=post_cost))
    return dict(
        manifest_count=len(manifest), observed_count=len(rows), startup_budget=budget,
        startup_observed_count=split, block_size=size,
        proxy_definition='relative raw Ip/Bt ordering proxy; not physical damage or DBC',
        full=_finish(counts), startup=startup, post_startup=post,
        startup_cost_proxy=startup['total_disrupt_proxy'],
        post_startup_missed_cost_proxy=post['missed_proxy_sum'],
        blocks=[dict(start_index=start+1, end_index=min(start+size, len(rows)),
                     **_segment(rows[start:start+size])) for start in range(0, len(rows), size)],
        high_proxy_quartile=dict(
            interpretation='OFFLINE_DIAGNOSTIC_COMPLETE_MANIFEST; not online selection',
            rule='top ceil(N/4) by (proxy,shot), ties by shot ID',
            manifest_shots=[r['shot'] for r in manifest if r['shot'] in high_ids],
            metrics=_segment([r for r in rows if r['shot'] in high_ids])),
        cumulative=cumulative)
