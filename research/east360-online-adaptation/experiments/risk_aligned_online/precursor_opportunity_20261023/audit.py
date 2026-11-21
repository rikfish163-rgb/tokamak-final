"""Independent prefix-sum arithmetic and timing audit; no model training."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import io23
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from .diagnostic import read_signal


def independent_change(signal, center, scale):
    x = signal['values'].astype('float64')
    available = signal['available'] & np.isfinite(x)
    q = np.arcsinh((np.where(available, x, center)-center)/scale)
    cs = np.vstack((np.zeros((1, 23)), q.cumsum(0)))
    cm = np.vstack((np.zeros((1, 23), np.int64), available.cumsum(0)))
    difference = np.zeros_like(q); valid = np.zeros_like(available)
    ends = np.arange(49, len(x))
    difference[ends] = (cs[ends+1]-2*cs[ends-24]+cs[ends-49])/25
    valid[ends] = cm[ends+1]-cm[ends-49] == 50
    valid[:, 10] = False
    return difference, valid


def independent_score(d, v, reference):
    usable = v & np.asarray(reference['active'], bool)
    normalized = np.full_like(d, np.nan)
    for c in np.flatnonzero(reference['active']):
        normalized[usable[:, c], c] = abs(d[usable[:, c], c])/reference['scale'][c]
    ordered = np.sort(np.where(usable, normalized, -np.inf), axis=1)
    score = ordered[:, -2]; score[usable.sum(1) < 2] = np.nan
    return score, normalized


def audit(root):
    read = lambda p: json.loads(p.read_text())
    done = read(root/'completed.json'); protocol = read(root/'protocol.json')
    for name, digest in done['artifacts'].items(): assert sha(root/name) == digest
    assert sha(Path(__file__).with_name('diagnostic.py')) == protocol['code_sha256']
    base, _ = load_parent(DEFAULT_PARENT)
    center, scale = [v.detach().cpu().numpy().astype('float64') for v in (base.center, base.scale)]
    rows = {r['shot']: r for r in read(root/'per_shot.json')}
    max_delta = max_score = 0.; checked = 0; safe_reference_shots = 0
    for plan in protocol['folds']:
        directory = root/f'fold{plan["fold"]}'; reference = read(directory/'reference.json')
        assert set(reference['safe_ids']) <= set(plan['fit'])
        assert not set(reference['safe_ids']) & set(plan['held'])
        safe_changes = []
        for sid in reference['safe_ids']:
            assert not io23.read_outcome('EAST', sid)[0]
            raw, signal = read_signal(sid)
            safe_changes.append(independent_change(signal, center, scale))
        quantiles = []
        for c in np.flatnonzero(reference['active']):
            quantiles.append(np.median([np.quantile(abs(d[v[:, c], c]), .95) for d, v in safe_changes]))
        np.testing.assert_allclose(quantiles, np.array(reference['scale'])[reference['active']], atol=1e-10, rtol=1e-7)
        peaks = [float(np.nanmax(independent_score(d, v, reference)[0])) for d, v in safe_changes]
        np.testing.assert_allclose(peaks, reference['safe_maxima'], atol=1e-5, rtol=1e-7)
        safe_reference_shots += len(peaks)
        for sid in plan['held']:
            raw, signal = read_signal(sid); d, v = independent_change(signal, center, scale)
            score, ratios = independent_score(d, v, reference)
            with np.load(directory/f'{sid}.npz') as stored:
                np.testing.assert_array_equal(v, stored['valid'])
                np.testing.assert_allclose(d, stored['delta'], atol=1e-10, rtol=1e-7)
                np.testing.assert_allclose(score, stored['score'], atol=1e-5, rtol=1e-7, equal_nan=True)
                np.testing.assert_allclose(ratios, stored['ratios'], atol=1e-5, rtol=1e-7, equal_nan=True)
                max_delta = max(max_delta, float(np.max(abs(d-stored['delta']))))
                max_score = max(max_score, float(np.nanmax(abs(score-stored['score']))))
                checked += len(score)
                risk = stored['static'].copy()
            disruptive, down = io23.read_outcome('EAST', sid)
            lead = np.round((down-raw['times'])*1000, 9) if disruptive else None
            valid_time = lead >= 0 if disruptive else np.ones(len(score), bool)
            above = np.isfinite(score) & (score > reference['boundary'])
            trigger = np.zeros(len(score), bool)
            count = 0
            for e, yes in enumerate(above):
                count = count+1 if yes else 0
                trigger[e] = count >= 3 and valid_time[e]
            valid_peak = valid_time & np.isfinite(score)
            critical = valid_peak & (lead > 30) & (lead <= 125) if disruptive else valid_peak
            r = rows[sid]
            assert r['is_disrupt'] == disruptive
            if disruptive: assert r['any_timely_exceedance'] == bool((trigger & (lead > 30)).any())
            for name, mask in [('first_exceedance', trigger), ('critical_first_exceedance', trigger & critical)]:
                indices = np.flatnonzero(mask); expected = r[name]
                if not len(indices):
                    assert expected is None
                    continue
                e = int(indices[0]); assert e == expected['end']
                assert float(risk[e]) == expected['static_risk']
                for fresh, key in [(False, 'next'), (True, 'complete_after')]:
                    schedule = [c for c in range(256, len(score), 128) if c-1 >= e and (not fresh or c-26 >= e)]
                    next_index = schedule[0] if schedule else None
                    assert next_index == expected[key+'_first_prediction']
                    if disruptive and next_index is not None:
                        assert float(lead[next_index]) == expected[key+'_lead_ms']
    assert len(rows) == 20 and sha(DEFAULT_PARENT) == PARENT_SHA256
    report = dict(status='VERIFIED', completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)), held_shots=20, reference_shot_fold_checks=safe_reference_shots,
        endpoint_checks=checked, max_change_difference=max_delta, max_score_difference=max_score,
        arithmetic='independent cumulative sums and sorting; independent trigger loop and update enumeration',
        source_unchanged=True, neural_updates=0, test360_accessed=False)
    _durable_json(root/'verification.json', report); print(report, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); audit(p.parse_args().root)
