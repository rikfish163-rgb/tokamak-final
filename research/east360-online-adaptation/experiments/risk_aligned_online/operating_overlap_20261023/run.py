"""Preflight on source TRAIN and ended EAST prefix20; no target outcome access.

This does NOT fit a model or report a TPR/FPR gain. It measures whether the
available physical-response descriptors can support conditional alignment.
"""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .diagnostics import describe_window, caliper_match, covariance_decomposition, qa_geometry_check

MAPPING = io23.ROOT/'experiments/risk_aligned_online/runs/deepcoral_prefix_test360_v2_20261022/density_map.json'
MAPPING_SHA = '35938a3a541ebe83f2fbae23e850b44ac11f50cd5b91556766dc462362292a3a'
PER_SHOT = 32


def collect(device, ids, model, mapping):
    windows, masks, rows, excluded, shot_meta = [], [], [], [], []
    for count, sid in enumerate(ids):
        s = apply_input_contract(simple_data.read_shot(device, sid))
        raw = {k:s[k] for k in ('values', 'available')}
        signal = raw if device == 'J-TEXT' else transform(raw, mapping)
        n = len(s['values'])
        shot_meta.append(dict(shot=int(sid), n=n, first_time=float(s['times'][0]),
            last_time=float(s['times'][-1]),
            qa_geometry=qa_geometry_check(raw, radius=.25 if device=='J-TEXT' else .45,
                                         major_radius=1.05 if device=='J-TEXT' else 1.85)))
        if n < 128:
            excluded.append(dict(shot=int(sid), reason='shorter_than_W128', n=n))
            continue
        ends = np.unique(np.linspace(127, n-1, min(PER_SHOT, n-127), dtype=int))
        start = len(rows)
        for end in ends:
            desc = describe_window(raw, int(end))
            if 'excluded' in desc:
                excluded.append(dict(shot=int(sid), end=int(end), **desc))
                continue
            rows.append(dict(shot=int(sid), end=int(end), time=float(s['times'][end]), **desc))
            x, m = simple_data.window(signal, int(end), 128)
            windows.append(x); masks.append(m)
        for row in rows[start:]:
            row['weight'] = 1./(len(rows)-start)
        if (count+1) % 300 == 0:
            print(json.dumps(dict(stage='signal_read',device=device,shots=count+1)),flush=True)
    if not rows:
        raise ValueError('no observed current/field windows')
    # Each retained shot has equal mass regardless of duration or window count.
    total = sum(r['weight'] for r in rows)
    for r in rows:
        r['weight'] /= total
    h = []
    with torch.no_grad():
        for lo in range(0, len(rows), 256):
            x = torch.as_tensor(np.stack(windows[lo:lo+256]),device='cuda')
            m = torch.as_tensor(np.stack(masks[lo:lo+256]),device='cuda')
            h.append(model(x,m,mode='risk_state')['risk_state'].cpu().numpy())
    return rows, np.concatenate(h).astype(float), excluded, shot_meta


def quantile_summary(rows):
    out = {}
    for key in ('ip_A','Bt_magnitude','relative_current_change','qa_raw','qa_Ip_over_Bt'):
        valid = [r for r in rows if r[key] is not None]
        if not valid:
            out[key] = dict(observations=0)
            continue
        x = np.array([r[key] for r in valid]); w = np.array([r['weight'] for r in valid])
        order = np.argsort(x,kind='stable'); x,w = x[order],w[order]/w.sum()
        # Discrete weighted CDF, no temporal samples counted as independent shots.
        q = x[np.searchsorted(np.cumsum(w),[0,.01,.05,.5,.95,.99,1],side='left').clip(0,len(x)-1)]
        out[key] = dict(observations=len(x), quantiles_0_1_5_50_95_99_100=q.tolist())
    return out


def covariance_summary(rows, h):
    w = np.array([r['weight'] for r in rows]); ids = np.array([r['shot'] for r in rows])
    w /= w.sum()
    drift = np.array([r['relative_current_change'] for r in rows])
    # These are RESPONSE bins, not verified programmed operating phases.
    phase = np.where(drift < -.01, -1, np.where(drift > .01, 1, 0))
    out = {}
    for name,groups in (('shot',ids),('observed_current_response',phase)):
        total,within,between = covariance_decomposition(h,w,groups)
        err = float(np.max(np.abs(total-within-between)))
        if err > 1.e-10:
            raise AssertionError('weighted covariance identity failed')
        trace = float(np.trace(total))
        out[name] = dict(total_trace=trace,within_trace=float(np.trace(within)),
            between_trace=float(np.trace(between)),
            between_fraction=None if trace == 0. else float(np.trace(between)/trace),
            fraction_status='undefined_zero_total_variance' if trace == 0. else 'defined',
            identity_max_abs=err,groups=[dict(group=int(g),windows=int((groups==g).sum()),
                shots=len(set(ids[groups==g])),mass=float(w[groups==g].sum())) for g in np.unique(groups)])
    return out


def match_summary(source, target, *, coordinates='absolute_current_field'):
    if coordinates == 'absolute_current_field':
        keys = ('ip_A','Bt_magnitude','relative_current_change')
        a,b = [np.array([[r[k] for k in keys] for r in rows]) for rows in (source,target)]
    elif coordinates == 'stored_qa_proxy':
        # Geometry-adjusted proxy is exploratory, NOT q95 or full regime.
        # A unit dummy second column reuses the identical relative caliper.
        if any(r['qa_raw'] is None or r['qa_raw']<=0 for r in source+target):
            raise ValueError('qa matching requires observed positive stored proxy in every retained window')
        a,b = [np.array([[r['qa_raw'],1.,r['relative_current_change']] for r in rows])
               for rows in (source,target)]
    else:
        raise ValueError('unknown matching coordinates')
    output = []
    tw = np.array([r['weight'] for r in target]); tw /= tw.sum()
    for caliper in (.05,.10,.20):
        for drift in (None,.01):
            counts,nearest = caliper_match(a,b,caliper,drift_tolerance=drift)
            covered = nearest >= 0
            source_masses = {}
            for index, weight in zip(nearest[covered],tw[covered]):
                sid = source[index]['shot']
                source_masses[sid] = source_masses.get(sid,0.)+float(weight)
            mass = sum(source_masses.values())
            ess = mass*mass/sum(x*x for x in source_masses.values()) if mass else 0.
            per_shot = []
            for sid in sorted({r['shot'] for r in target}):
                ix = np.array([r['shot']==sid for r in target])
                per_shot.append(dict(shot=sid,windows=int(ix.sum()),covered=int((ix&covered).sum())))
            output.append(dict(coordinates=coordinates,relative_caliper=caliper,drift_tolerance=drift,
                covered_weight=float(tw[covered].sum()),covered_windows=int(covered.sum()),
                total_windows=len(target),target_shots_with_any_match=sum(r['covered']>0 for r in per_shot),
                unique_nearest_source_shots=len(source_masses),nearest_source_effective_shots=ess,
                source_shot_mass=source_masses,per_target_shot=per_shot,
                counts=counts.tolist(),nearest_source_row=nearest.tolist()))
    return output


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    if sha(MAPPING) != MAPPING_SHA:
        raise ValueError('mapping changed')
    mapping = json.loads(MAPPING.read_text())
    source_ids = io23.source_roles()['train']; target_ids = io23.cohort_ids('EAST')[:20]
    protocol = dict(source='JTEXT TRAIN1353 signals only',target='chronological fixed EAST TEST360 first20 signals only',
        source_ids=source_ids,target_ids=target_ids,parent_sha256=PARENT_SHA256,mapping_sha256=MAPPING_SHA,
        labels_accessed=0,model_updates=0,source_window=128,max_windows_per_shot=PER_SHOT,
        matching_calipers=[.05,.10,.20],relative_current_change_caliper=.01,
        matching_calipers_are_descriptive_not_certified_physics=True,
        matching_coordinates=['absolute_current_field','stored_qa_proxy'],
        qa_formula_probe='2*pi*a^2*Bt/(R*Ip_A); nominal JT a=.25 R=1.05; EAST a=.45 R=1.85',
        target_preparation='only density20 empirical gain for representation evaluation; descriptors unchanged',
        model_parameters_unchanged=True,status='DEVELOPMENT_DIAGNOSTIC_NOT_PERFORMANCE_EXPERIMENT',
        files={str(q.relative_to(io23.ROOT)):sha(q) for q in sorted(Path(__file__).parent.glob('*.py'))})
    _durable_json(args.output/'protocol.json',protocol)
    torch.set_num_threads(4)
    parent, _ = load_parent(DEFAULT_PARENT)
    # level_scale belongs only to the unused reconstruction head. The
    # risk_state path is exactly the existing base encoder/readout coordinate.
    model = PairedFeedback(parent, np.ones(23)).to('cuda'); model.set_stage('frozen')
    before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    with patch.object(io23,'read_outcome',side_effect=AssertionError('outcome access forbidden')):
        source,h_source,se,sm = collect('J-TEXT',source_ids,model,mapping)
        target,h_target,te,tm = collect('EAST',target_ids,model,mapping)
    for k,v in model.state_dict().items():
        if not torch.equal(v.cpu(),before[k]): raise AssertionError('model changed')
    _durable_json(args.output/'source_rows.json',source); _durable_json(args.output/'target_rows.json',target)
    _durable_json(args.output/'sampling.json',dict(source_excluded=se,target_excluded=te,source_shots=sm,target_shots=tm))
    np.savez_compressed(args.output/'states.npz',source=h_source,target=h_target)
    source_summary = dict(shots=len({r['shot'] for r in source}),windows=len(source),
                          descriptors=quantile_summary(source),covariance=covariance_summary(source,h_source))
    targets = {}
    for count in (10,20):
        selected = np.array([r['shot'] in set(target_ids[:count]) for r in target])
        rows = [r for r,keep in zip(target,selected) if keep]
        targets[str(count)] = dict(shots=count,windows=len(rows),descriptors=quantile_summary(rows),
            representation_mapping_fit_prefix=20,
            representation_statistics='retrospective subset in fixed prefix20 coordinates, not causal prefix10 preparation',
            covariance=covariance_summary(rows,h_target[selected]),matching=match_summary(source,rows),
            qa_proxy_matching=match_summary(source,rows,coordinates='stored_qa_proxy'))
    result = dict(source=source_summary,target_prefix=targets,seconds=time.monotonic()-started,
        outcome_reader_guard='no calls',model_unchanged=True,parent_hash_after=sha(DEFAULT_PARENT))
    _durable_json(args.output/'summary.json',result)
    _durable_json(args.output/'receipt.json',{q.name:sha(q) for q in sorted(args.output.iterdir()) if q.is_file()})
    print(json.dumps(dict(source_windows=len(source),target_windows=len(target),seconds=result['seconds'],
        prefix20_matches=[{k:r[k] for k in ('coordinates','relative_caliper','drift_tolerance',
            'covered_weight','unique_nearest_source_shots','nearest_source_effective_shots')}
            for r in targets['20']['matching']+targets['20']['qa_proxy_matching']]),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
