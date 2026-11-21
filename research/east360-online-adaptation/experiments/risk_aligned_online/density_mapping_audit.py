"""Post-run, single-input density intervention; not online training.

Independently checks HDF5 outcomes and immutable seals for all matched shots.
Only the numerical ne_nG coordinate is interpolated at two already-emitted
peaks. These interventions diagnose MODEL sensitivity, not physical causality.
No interpolation value is selected for deployment and no threshold is fitted.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import io23, simple_data
from .alignment import fit_ended_shot_statistics
from .causal_input import apply_input_contract
from .ended_density_replay import EndedDensityReplay
from .persistent_ffn import PersistentFfnModel
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .simple_online import _durable_json, _hash
from .verify_ended_decoder import classify

ROOT = Path(__file__).parent
RAW = ROOT/'runs/gradient_transport_commission240_20261009/mask_static'
MAPPED = ROOT/'runs/ended_decoder120_20261010/static'
DENSITY = io23.RAW23.index('ne_nG')


def density_coordinates(values, *, source_center, source_scale, target_center, target_scale):
    scalars = np.array([source_center, source_scale, target_center, target_scale], float)
    if not np.isfinite(scalars).all() or source_scale <= 0 or target_scale <= 0:
        raise ValueError('finite centers and positive source/target scales required')
    v = np.asarray(values, dtype=np.float64)
    if not np.isfinite(v).all():
        raise ValueError('density observations must be finite')
    effective = max(float(target_scale), abs(float(target_center))*.01, 1e-6)
    return dict(raw_source=np.arcsinh((v-source_center)/source_scale),
                mapped_source=np.arcsinh((v-target_center)/effective),
                effective_target_scale=effective)


def eligible_peak(times, scores, *, disruptive, down):
    times, scores = np.asarray(times), np.asarray(scores)
    if (times.ndim != 1 or times.shape != scores.shape
            or not np.isfinite(times).all() or not np.isfinite(scores).all()):
        raise ValueError('matching finite 1D time and score arrays required')
    eligible = np.ones(len(times), bool)
    if disruptive:
        if down is None or not np.isfinite(down):
            raise ValueError('finite disruption time required')
        eligible &= np.round((down-times)*1000., 9) > 30.
    indices = np.flatnonzero(eligible)
    return int(indices[np.argmax(scores[indices])]) if len(indices) else None


def change_type(disruptive, before, after):
    if before == after:
        return 'unchanged'
    if disruptive:
        return 'lost' if before else 'rescued'
    return 'removed_fp' if before else 'new_fp'


def source_coordinate_path(raw, mapped, alphas, *, source_center, source_scale):
    """Interpolate density AFTER source asinh, then invert to file coordinates.

    Linear mixing of raw values ~10 and mapped values ~.002 leaves almost all
    intermediate points far outside source coordinates. This second path
    explicitly probes the normalized numerical input the embedding sees.
    Neither path estimates a physically correct density or a deployable map.
    """
    raw, mapped = np.asarray(raw), np.asarray(mapped)
    if (raw.shape != (128,23) or mapped.shape != raw.shape
            or not np.isfinite([source_center,source_scale]).all() or source_scale<=0
            or any(not 0<=a<=1 for a in alphas)):
        raise ValueError('matching W128 flat23 windows and finite source coordinates required')
    other = np.arange(23)!=DENSITY
    if not np.array_equal(raw[:,other], mapped[:,other]):
        raise ValueError('one-density-column intervention required')
    z0 = np.arcsinh((raw[:,DENSITY].astype(float)-source_center)/source_scale)
    z1 = np.arcsinh((mapped[:,DENSITY].astype(float)-source_center)/source_scale)
    result = np.broadcast_to(raw, (len(alphas),*raw.shape)).copy()
    for i, a in enumerate(alphas):
        if a==0:
            continue
        if a==1:
            result[i] = mapped
        else:
            result[i,:,DENSITY] = source_center+source_scale*np.sinh((1-a)*z0+a*z1)
    if not np.isfinite(result).all():
        raise FloatingPointError('nonfinite source-coordinate intervention')
    return result


def initial_density(signals):
    """First100ms of the SAVED RECORD, never called a safe/reference phase."""
    times = np.asarray(signals['times'])
    if times.ndim!=1 or not len(times) or not np.isfinite(times).all():
        raise ValueError('nonempty finite saved-record timestamps required')
    take = np.round((times-times[0])*1000.,9)<100.
    take &= signals['available'][:,DENSITY]
    v = signals['values'][take,DENSITY]
    return v[np.isfinite(v)]


def prefix_statistics(records):
    samples = [initial_density(s) for s in records]
    if not samples or not any(len(v) for v in samples):
        raise ValueError('observed initial-record density required')
    return dict(n_records=len(records), observed_points=sum(len(v) for v in samples),
        quartiles=np.quantile(np.concatenate(samples),[.25,.5,.75]).tolist(),
        record_start_time_quantiles=np.quantile([s['times'][0] for s in records],[0,.5,1]).tolist(),
        observed_count_quantiles=np.quantile([len(v) for v in samples],[0,.5,1]).tolist(),
        same_physical_phase_not_established=True)


def read_trace(folder, index, sid, raw, threshold):
    seal = json.loads((folder/f'{index:04d}_{sid}_seal.json').read_text())
    audit = json.loads((folder/f'{index:04d}_{sid}_audit.json').read_text())
    if (seal['shot'] != sid or seal['threshold'] != threshold
            or _hash(folder/seal['npz']) != seal['sha256']
            or audit['seal_sha256'] != seal['sha256']):
        raise AssertionError('archived prediction seal identity differs')
    with np.load(folder/seal['npz'], allow_pickle=False) as trace:
        times, scores = trace['times'].copy(), trace['scores'].copy()
        if (int(trace['shot']) != sid or not np.array_equal(times, raw['times'])
                or scores.shape != times.shape or not np.isfinite(scores).all()):
            raise AssertionError('signal/trace identity or finite shape differs')
    if (audit['model_parameter_change_norm'] != 0 or audit['model_buffer_change_norm'] != 0
            or not audit['fast_discarded']):
        raise AssertionError('not an unchanged-source static prediction')
    return times, scores


def counts(rows):
    d = sum(r['is_disrupt'] for r in rows)
    s = len(rows)-d
    tp = sum(r['timely'] for r in rows)
    fp = sum(r['alarm'] and not r['is_disrupt'] for r in rows)
    return dict(n=len(rows), n_disrupt=d, n_safe=s, tp=tp, fp=fp,
                tpr=tp/d, fpr=fp/s, J=tp/d-fp/s,
                early_over_3000=sum(r['timely'] and r['lead_ms']>3000 for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not (MAPPED.parent/'independent_verification.json').is_file():
        parser.error('completed verified traces and a new output receipt required')
    torch.set_num_threads(4)
    order = json.loads((MAPPED.parent/'order.json').read_text())
    if json.loads((RAW.parent/'order.json').read_text())[:len(order)] != order:
        raise AssertionError('paired streams have different order')
    base, source = load_parent(DEFAULT_PARENT)
    model = PersistentFfnModel(base).to('cuda').eval().requires_grad_(False)
    threshold = float(source['threshold'])
    originals = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    calibration = [apply_input_contract(simple_data.read_shot('EAST', r['shot'],
        rate_contract='nominal')) for r in order[:30]]
    statistics = fit_ended_shot_statistics(calibration,
        ended_ids=[r['shot'] for r in calibration])
    provider = EndedDensityReplay(source['source_center'], source['source_scale'],
        output_dir=MAPPED, order=[r['shot'] for r in order], preserve_observed_channels=True)
    # POST-RUN reconstruction only. This never invokes the predictor's online
    # read/reveal lifecycle or supplies diagnostic outcomes to a learner.
    provider.statistics = statistics
    scale_args = dict(source_center=float(source['source_center'][DENSITY]),
        source_scale=float(source['source_scale'][DENSITY]),
        target_center=float(statistics['center'][DENSITY]),
        target_scale=float(statistics['scale'][DENSITY]))
    paired, recounted = [], dict(raw=[], mapped=[])
    errors, startup_error = [], 0.
    alphas = (0., .25, .5, .75, 1.)  # Sensitivity grid, NOT a deployed search.
    for index, item in enumerate(order, 1):
        sid = item['shot']
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        disruptive, down = io23.read_outcome('EAST', sid)
        traces = [read_trace(folder, index, sid, raw, threshold) for folder in (RAW, MAPPED)]
        for name, (times, scores) in zip(recounted, traces):
            recounted[name].append(dict(shot=sid, **classify(times, scores,
                np.ones(len(times), bool), threshold=threshold, disruptive=disruptive, down=down)))
        if index <= 30:
            startup_error = max(startup_error, float(np.max(np.abs(traces[0][1]-traces[1][1]))))
        left, right = recounted['raw'][-1], recounted['mapped'][-1]
        field = 'timely' if disruptive else 'alarm'
        category = change_type(disruptive, left[field], right[field])
        if category == 'unchanged':
            continue
        if index <= 30:
            raise AssertionError('density switch changed a pre-calibration shot')
        mapped = provider.reencode(raw)
        if (not np.array_equal(mapped['available'], raw['available'])
                or not np.array_equal(mapped['values'][:, np.arange(23)!=DENSITY],
                                      raw['values'][:, np.arange(23)!=DENSITY])):
            raise AssertionError('density intervention changed masks or another feature')
        observations = raw['values'][raw['available'][:, DENSITY], DENSITY]
        coordinates = density_coordinates(observations, **scale_args)
        queries = [eligible_peak(t, p, disruptive=disruptive, down=down) for t,p in traces]
        queries = sorted({q for q in queries if q is not None})
        peak_rows = []
        for query in queries:
            x, mask = simple_data.window(raw, query, 128)
            mapped_x, mapped_mask = simple_data.window(mapped, query, 128)
            if not np.array_equal(mask, mapped_mask):
                raise AssertionError('causal window mask changed')
            xs = np.stack([x.astype(np.float64)*(1-a)+mapped_x.astype(np.float64)*a
                           for a in alphas]).astype(np.float32)
            with torch.no_grad():
                prob = model(torch.as_tensor(xs, device='cuda'), torch.as_tensor(
                    np.broadcast_to(mask, xs.shape).copy(), device='cuda'))['logit'].sigmoid().cpu().numpy()
                zpath = source_coordinate_path(x, mapped_x, alphas,
                    source_center=scale_args['source_center'], source_scale=scale_args['source_scale'])
                zprob = model(torch.as_tensor(zpath, device='cuda'), torch.as_tensor(
                    np.broadcast_to(mask, zpath.shape).copy(), device='cuda'))['logit'].sigmoid().cpu().numpy()
            error = max(abs(float(prob[0])-float(traces[0][1][query])),
                        abs(float(prob[-1])-float(traces[1][1][query])))
            if error > 2e-6:
                raise AssertionError(f'zero-update intervention replay differs for {sid}: {error}')
            errors.append(error)
            qcoord = density_coordinates(raw['values'][query:query+1, DENSITY], **scale_args)
            peak_rows.append(dict(index=query, time=float(raw['times'][query]),
                lead_ms=round((down-float(raw['times'][query]))*1000., 9) if disruptive else None,
                raw_density=float(raw['values'][query, DENSITY]),
                original_source_coordinate=float(qcoord['raw_source'][0]),
                mapped_source_coordinate=float(qcoord['mapped_source'][0]),
                raw_score=float(traces[0][1][query]), mapped_score=float(traces[1][1][query]),
                interpolation_scores=[float(p) for p in prob],
                normalized_coordinate_interpolation_scores=[float(p) for p in zprob]))
        paired.append(dict(index=index, shot=sid, category=category, raw=left, mapped=right,
            raw_density_quantiles=np.quantile(observations, [.25,.5,.75]).tolist(),
            original_coordinate_quantiles=np.quantile(coordinates['raw_source'], [.25,.5,.75]).tolist(),
            mapped_coordinate_quantiles=np.quantile(coordinates['mapped_source'], [.25,.5,.75]).tolist(),
            peak_rows=peak_rows))
    if startup_error != 0. or _hash(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError('startup control or qualified source changed')
    if any(not torch.equal(v.detach().cpu(), originals[k]) for k,v in model.state_dict().items()):
        raise AssertionError('forensic forward changed model state')
    roles = simple_data.source_roles()
    # Source TRAIN signals only. This is a descriptive phase-comparability
    # check; no VAL/TEST reads, outcome reads, transform fitting or deployment.
    source_prefixes=[]
    for sid in roles['train']:
        s = simple_data.read_shot('J-TEXT',sid)
        take=np.round((s['times']-s['times'][0])*1000.,9)<100.
        source_prefixes.append(dict(times=s['times'][take], values=s['values'][take],
                                    available=s['available'][take]))
    receipt = dict(source_sha256=PARENT_SHA256, code_sha256=_hash(Path(__file__)),
        threshold=threshold, calibration_ids=[s['shot'] for s in calibration],
        statistics=dict(**scale_args, target_count=int(statistics['counts'][DENSITY])),
        first30_max_curve_error=startup_error, reconstructed_peak_max_error=max(errors),
        initial_record100ms=dict(source_train=prefix_statistics(source_prefixes),
                                ended_east30=prefix_statistics(calibration)),
        independently_verified_seals=2*len(order),
        metrics={name:{period:counts(rows[a:b]) for period,a,b in
            (('all',0,len(order)),('first30',0,30),('after30',30,len(order)))}
            for name,rows in recounted.items()},
        changes={k:[r['shot'] for r in paired if r['category']==k]
                 for k in ('lost','rescued','removed_fp','new_fp')},
        interpolation_alphas=list(alphas), changed_shots=paired,
        no_learning_no_threshold_search=True, input_intervention_only=True,
        no_physical_causality_claim=True, development_only=True)
    _durable_json(args.output, receipt)
    print(json.dumps({k:receipt[k] for k in ('statistics','metrics','changes',
        'first30_max_curve_error','reconstructed_peak_max_error')}), flush=True)


if __name__ == '__main__':
    main()
