"""Read-only, causal signal-change audit. Never an alarm or adaptation policy."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha

REFERENCE = io23.ROOT / 'experiments/risk_aligned_online/runs/attention_scope_cv20_20261023'
REF_SHA = '4ee481fc894cc8e392b38620bfe971a743bb951859e528654259d07ba8a5ca81'
PLAN = io23.ROOT / 'docs/research/source_event_rebuild_20261022/信号变化与反馈时机_诊断约定_20261023.md'


def changes(signal, center, scale):
    """At t compare [t-24,t] and [t-49,t-25]; no partial missing means."""
    if set(signal) != {'values', 'available'}:
        raise ValueError('signal-only fields required')
    x, a = np.asarray(signal['values']), np.asarray(signal['available'])
    c, s = np.asarray(center), np.asarray(scale)
    if (x.ndim != 2 or x.shape[1] != 23 or a.shape != x.shape or a.dtype != bool
            or c.shape != (23,) or s.shape != (23,) or not np.isfinite(c).all()
            or not np.isfinite(s).all() or (s <= 0).any()):
        raise ValueError('finite positive flat23 source coordinate and mask required')
    a = a & np.isfinite(x)
    q = np.arcsinh((np.where(a, x, c).astype(np.float64)-c)/s)
    delta, valid = np.zeros_like(q), np.zeros_like(a)
    if len(x) >= 50:
        means = np.lib.stride_tricks.sliding_window_view(q, 25, axis=0).mean(-1)
        delta[49:] = means[25:]-means[:-25]
        valid[49:] = np.lib.stride_tricks.sliding_window_view(a, 50, axis=0).all(-1)
    valid[:, io23.RAW23.index('mode_number_n')] = False
    return delta, valid


def fit_reference(rows, ids):
    if not rows or len(rows) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError('distinct safe reference shots required')
    quantiles = np.full((len(rows), 23), np.nan)
    for i, (d, v) in enumerate(rows):
        for c in range(23):
            if v[:, c].any():
                quantiles[i, c] = np.quantile(abs(d[v[:, c], c]), .95)
    active = np.isfinite(quantiles).all(0)
    scales = np.zeros(23)
    scales[active] = np.median(quantiles[:, active], axis=0)
    # Numerical-resolution guard in dimensionless coordinates, NOT a physical
    # noise floor. A zero/unknown reference is excluded, not divided by epsilon.
    active &= scales > 64*np.finfo(np.float64).eps
    if active.sum() < 2:
        raise ValueError('at least two measured nonconstant reference channels required')
    result = dict(safe_ids=list(ids), active=active.tolist(), scale=scales.tolist())
    peaks = []
    for d, v in rows:
        score, _ = novelty(d, v, result)
        if not np.isfinite(score).any():
            raise ValueError('safe reference has no jointly observed change')
        peaks.append(float(np.nanmax(score)))
    result['safe_maxima'] = peaks
    result['boundary'] = max(peaks)
    result['excluded_channels'] = [io23.RAW23[c] for c in np.flatnonzero(~active)]
    return result


def novelty(delta, valid, reference):
    scale = np.asarray(reference['scale']); active = np.asarray(reference['active'], bool)
    if delta.shape != valid.shape or delta.ndim != 2 or delta.shape[1] != 23:
        raise ValueError('matching flat23 change arrays required')
    usable = valid & active[None]
    ratios = np.full_like(delta, np.nan)
    np.divide(abs(delta), scale, out=ratios, where=usable)
    # Second largest: requires two measured channels, rather than treating
    # absent channels as quiet physical observations.
    values = np.where(usable, ratios, -np.inf)
    score = np.partition(values, -2, axis=1)[:, -2].copy()
    score[usable.sum(1) < 2] = np.nan
    return score, ratios


def sustained(score, boundary):
    above = np.isfinite(score) & (score > boundary)
    answer = np.zeros(len(score), bool)
    if len(score) >= 3:
        answer[2:] = above[2:] & above[1:-1] & above[:-2]
    return answer


def first_sustained(score, boundary):
    indices = np.flatnonzero(sustained(score, boundary))
    return int(indices[0]) if len(indices) else None


def next_feedback(end, length, *, full_after=False):
    """First affected prediction of a scheduled update seeing endpoint end.

    full_after additionally requires the latest H25 forecast origin to be at
    or after end. This is a stricter diagnostic, not the existing update rule.
    """
    required = end + (26 if full_after else 1)
    count = max(256, ((required+127)//128)*128)
    return count if count < length else None


def read_signal(sid):
    raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
    return raw, {k: raw[k] for k in ('values', 'available')}


def summarize_shot(sid, fold, raw, delta, valid, score, ratios, risk, reference, outcome):
    disruptive, down = outcome
    lead = np.round((down-raw['times'])*1000, 9) if disruptive else None
    keep = lead >= 0 if disruptive else np.ones(len(score), bool)
    finite = keep & np.isfinite(score)
    critical = (lead > 30) & (lead <= 125) if disruptive else keep
    eligible = finite & critical
    trigger = sustained(score, reference['boundary']) & keep

    def event(indices):
        if not len(indices): return None
        e = int(indices[0]); count = next_feedback(e, len(score))
        fresh = next_feedback(e, len(score), full_after=True)
        return dict(end=e, time=float(raw['times'][e]),
                    lead_ms=float(lead[e]) if disruptive else None,
                    static_risk=float(risk[e]),
                    next_first_prediction=count,
                    next_lead_ms=float(lead[count]) if disruptive and count is not None else None,
                    complete_after_first_prediction=fresh,
                    complete_after_lead_ms=float(lead[fresh]) if disruptive and fresh is not None else None)

    indices = np.flatnonzero(eligible)
    peak = int(indices[np.argmax(score[indices])]) if len(indices) else None
    top = []
    if peak is not None:
        cs = np.flatnonzero(np.isfinite(ratios[peak]))
        for c in sorted(cs, key=lambda k: ratios[peak, k], reverse=True)[:5]:
            top.append(dict(channel=io23.RAW23[c], normalized_change=float(ratios[peak, c]),
                signed_source_coordinate_change=float(delta[peak, c]),
                earlier_raw_mean=float(raw['values'][peak-49:peak-24, c].astype('float64').mean()),
                recent_raw_mean=float(raw['values'][peak-24:peak+1, c].astype('float64').mean())))
    return dict(shot=sid, fold=fold, is_disrupt=disruptive, boundary=reference['boundary'],
        whole_peak=float(score[finite].max()) if finite.any() else None,
        critical_peak=float(score[peak]) if peak is not None else None,
        critical_peak_time=float(raw['times'][peak]) if peak is not None else None,
        critical_peak_lead_ms=float(lead[peak]) if disruptive and peak is not None else None,
        critical_peak_static_risk=float(risk[peak]) if peak is not None else None,
        first_exceedance=event(np.flatnonzero(trigger)),
        critical_first_exceedance=event(np.flatnonzero(trigger & critical)),
        any_timely_exceedance=bool((trigger & (lead > 30)).any()) if disruptive else None,
        reference_tail_rank=(1+sum(p >= score[peak] for p in reference['safe_maxima']))/(1+len(reference['safe_maxima'])) if peak is not None else None,
        peak_channels=top)


def main(root):
    if root.exists(): raise FileExistsError(root)
    start = time.monotonic()
    assert sha(REFERENCE/'completed.json') == REF_SHA
    previous = json.loads((REFERENCE/'completed.json').read_text())
    for name, digest in previous['artifact_sha256'].items(): assert sha(REFERENCE/name) == digest
    protocol = json.loads((REFERENCE/'protocol.json').read_text())
    base, _ = load_parent(DEFAULT_PARENT)
    center = base.center.detach().cpu().numpy().astype('float64')
    scale = base.scale.detach().cpu().numpy().astype('float64')
    root.mkdir(); all_ids = sorted({sid for p in protocol['folds'] for sid in p['held']})
    assert len(all_ids) == 20
    code = Path(__file__)
    _durable_json(root/'protocol.json', dict(reference=str(REFERENCE), reference_sha256=REF_SHA,
        folds=protocol['folds'], source_sha256=PARENT_SHA256, plan_sha256=sha(PLAN),
        code_sha256=sha(code), read_only=True, test360_accessed=False,
        evidence='DEVELOPMENT_DESCRIPTIVE_SIGNAL_CHANGE_NOT_PHYSICAL_ONSET_OR_ONLINE_POLICY'))
    reference_rows, seals, label_access = {}, [], []
    read_outcome = io23.read_outcome
    for plan in protocol['folds']:
        f = plan['fold']; directory = root/f'fold{f}'; directory.mkdir()
        # Historical fit labels are authorized; held labels are not needed here.
        safe = []
        for sid in plan['fit']:
            disruptive, _ = read_outcome('EAST', sid)
            label_access.append(dict(fold=f, shot=sid, role='historical_fit'))
            if not disruptive: safe.append(sid)
        assert len(safe) == 12 and not set(safe) & set(plan['held'])
        mapping = json.loads((REFERENCE/f'fold{f}'/'input_map.json').read_text())
        assert np.array_equal(mapping['gain'], np.ones(23)) and not np.any(mapping['offset'])
        with patch.object(io23, 'read_outcome', side_effect=AssertionError('no outcomes during signal scoring')):
            rows = [changes(read_signal(sid)[1], center, scale) for sid in safe]
            reference = fit_reference(rows, safe); reference_rows[f] = reference
            _durable_json(directory/'reference.json', reference)
            for sid in plan['held']:
                raw, signal = read_signal(sid)
                delta, valid = changes(signal, center, scale)
                score, ratios = novelty(delta, valid, reference)
                with np.load(REFERENCE/f'fold{f}'/'shots'/f'{sid}.npz') as old:
                    np.testing.assert_array_equal(raw['times'], old['times'])
                    risk = old['static'].copy()
                path = directory/f'{sid}.npz'
                np.savez_compressed(path, times=raw['times'], delta=delta, valid=valid,
                                    score=score, ratios=ratios, static=risk)
                seals.append(dict(fold=f, shot=sid, sha256=sha(path)))
        print(dict(stage='scored_fold', fold=f, seconds=time.monotonic()-start), flush=True)
    _durable_json(root/'seals.json', seals)
    summaries = []
    for seal in seals:
        sid, f = seal['shot'], seal['fold']
        raw, _ = read_signal(sid); outcome = read_outcome('EAST', sid)
        label_access.append(dict(fold=f, shot=sid, role='evaluation_after_all_seals'))
        with np.load(root/f'fold{f}'/f'{sid}.npz') as trace:
            summaries.append(summarize_shot(sid, f, raw, trace['delta'], trace['valid'],
                trace['score'], trace['ratios'], trace['static'], reference_rows[f], outcome))
    _durable_json(root/'label_access.json', label_access)
    _durable_json(root/'per_shot.json', summaries)
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    result = dict(status='COMPLETED', seconds=time.monotonic()-start, source_unchanged=True,
        test360_accessed=False, neural_updates=0,
        timely_change_exceedances=sum(r['any_timely_exceedance'] is True for r in summaries),
        safe_change_exceedances=sum(not r['is_disrupt'] and r['first_exceedance'] is not None for r in summaries),
        disruptive_details=[r for r in summaries if r['is_disrupt']],
        artifacts={str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()})
    _durable_json(root/'completed.json', result)
    print({k: v for k, v in result.items() if k != 'artifacts'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
