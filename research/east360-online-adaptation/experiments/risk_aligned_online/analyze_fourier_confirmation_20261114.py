"""Compare each of the eleven original Fourier flips after the run is sealed."""

import argparse
import csv
from pathlib import Path

import numpy as np

from . import commissioning_fourier_confirmation_20261114 as run, simple_data
from .history_gradient_metric_20261102.run import read, sha, write


def analyze(folder):
    proto, fit, seal, metrics, per = (read(folder/f'{k}.json') for k in
                                     ('protocol','fit','seal','result','per_shot'))
    ref = read(run.REFERENCE/'seal.json')
    lock = run.prior.lag.load_inputs()
    cutoff = proto['calibration'][run.prior.MAIN]
    results = metrics['comparisons'][run.prior.MAIN]['fourier']['flips']
    cohort = (('rescue', results['rescued_D']), ('new_false_alarm', results['added_FP']))
    if (len(cohort[0][1]), len(cohort[1][1])) != (4, 7):
        raise ValueError('expected the existing four rescues and seven false alarms')
    rows = []
    for category, shots in cohort:
        for shot in shots:
            rank = proto['order'].index(shot)+1
            key = f'{rank:04d}_{shot}.npz'
            here, original = folder/'shots'/key, run.REFERENCE/'shots'/key
            if sha(here) != seal['curve_sha256'][key] or sha(original) != ref['curve_sha256'][key]:
                raise ValueError('comparison requires sealed risk curves')
            status = {arm: v[rank-1] for arm,v in per[run.prior.MAIN].items()}
            q = status['fourier']['first_idx']
            raw = simple_data.read_shot('EAST', shot, rate_contract='nominal')
            sig = run.prior.lag.prepare_actual(raw, rank, lock['mapping'])
            x, present = run.features(sig, np.array([q]))
            with np.load(here, allow_pickle=False) as f, np.load(original, allow_pickle=False) as past:
                t = f['times']
                np.testing.assert_array_equal(t, raw['times'])
                ttd = status['fourier']['down_s']
                valid = (np.round((ttd-t)*1000.,9)>30.) if ttd is not None else np.ones(len(t),bool)
                optional = valid & (f['fourier']>=cutoff) & (f['original']<cutoff)
                evidence_names = dict(confirmed_current='evidence_current', confirmed='evidence')
                maxima = {arm: float(f[k][optional & f['available']].max())
                          if (optional & f['available']).any() else None for arm,k in evidence_names.items()}
                # A classifier would need to separate shot-level maxima, not
                # merely find a difference at the first Fourier crossing.
                # Check the final implemented gate on a signal prefix at that crossing.
                n = q+1
                cut = {k: sig[k][:n] for k in ('values','available')}
                kscore, kval, kknown = run.predict_curves(f['original'][:n], f['fourier'][:n], cut, fit)
                for arm in run.ARMS:
                    np.testing.assert_array_equal(kscore[arm], f[arm][:n])
                for arm,k in evidence_names.items():
                    np.testing.assert_array_equal(kval[arm], f[k][:n])
                np.testing.assert_array_equal(kknown, f['available'][:n])
                n1 = np.abs(sig['values'][q-run.W+1:q+1, run.COLS[0]])
                rows.append(dict(shot=shot, category=category, fourier_first_s=float(t[q]),
                    original_score_at_fourier_first=float(f['original'][q]),
                    fourier_first_score=float(f['fourier'][q]), current_last32_kA=float(n1[-run.BLOCK:].mean()/1000.),
                    prior128_relative_changes_pct={k: float(100*x[0,i,0]) if present[0,i] else None
                                                  for i,k in enumerate(run.CHANNELS)},
                    step5_active_at_fourier_first=bool(past['dose'][q]>0), window_count=int(optional.sum()),
                    evidence_at_fourier_first={arm: float(f[k][q]) for arm,k in evidence_names.items()},
                    peak_confirmation_on_eligible_crossings=maxima,
                    decisions={k:{j:r[j] for j in ('first_s','lead_ms','timely','alarm')} for k,r in status.items()}))
    overlaps = {}
    for arm in run.ARMS[-2:]:
        p = [r['peak_confirmation_on_eligible_crossings'][arm] for r in rows if r['category']=='rescue']
        n = [r['peak_confirmation_on_eligible_crossings'][arm] for r in rows if r['category']=='new_false_alarm']
        cut = fit[arm]['cutoff']
        overlaps[arm] = dict(frozen_cutoff=cut, rescue_scores=p, false_alarm_scores=n,
            safe_shots_above_best_rescue=sum(score > max(p) for score in n))
    write(folder/'fourier_11_cases.json',dict(rows=rows, overlap=overlaps,
        change_definition='100 × (last 32 points mean − preceding 96 points mean) / whole 128-point mean, using magnitudes',
        prefix_witnesses=len(rows)))
    with (folder/'fourier_11_cases.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w = csv.writer(f)
        common = ('shot','category','fourier_first_s','original_score_at_fourier_first','fourier_first_score',
                  'current_last32_kA','step5_active_at_fourier_first')
        keys = ('first_s','lead_ms','timely','alarm')
        gate_arms = run.ARMS[-2:]
        w.writerow((*common, *(f'{c}_change_pct' for c in run.CHANNELS),
                    *(f'{k}_evidence_at_first' for k in gate_arms), *(f'{k}_peak_evidence' for k in gate_arms),
                    *(f'{k}_{s}' for k in run.ARMS for s in keys)))
        for r in rows:
            w.writerow([r[k] for k in common]+[r['prior128_relative_changes_pct'][c] for c in run.CHANNELS]
                       +[r['evidence_at_fourier_first'][k] for k in gate_arms]
                       +[r['peak_confirmation_on_eligible_crossings'][k] for k in gate_arms]
                       +[r['decisions'][k][s] for k in run.ARMS for s in keys])
    print(overlaps,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    analyze(args.out.resolve())
