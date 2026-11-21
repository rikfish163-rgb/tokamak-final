"""Retrospective frozen-computation diagnostic, NOT an online adaptation arm.

Pre-fixed sparse roster, endpoint-local matching, one observed ne_nG trajectory
swap. No optimizer, scale fitting, outcome-based matching or risk correction.
Real donor trajectories do not guarantee an on-manifold joint counterfactual.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import io23, simple_data
from .causal_input import apply_input_contract
from .ffn_intra_20261005.model import FfnModel
from .score_shift_audit import sha
from .simple23 import Config, Simple23

CHANNEL = 12  # ne_nG, not raw density; fixed before examining effects.
CONTEXT_LIMIT = .25  # RMS in source-asinh coordinates; not a physical tolerance.
MIN_DOSE = .05
LOGIT_ATOL = 5e-5
PROB_ATOL = 1e-5


def stratified_indices():
    result = []
    for start, end in zip((0, 30, 120, 600, 1500), (30, 120, 600, 1500, 2953)):
        result.extend(np.linspace(start, end-1, min(32, end-start), dtype=int).tolist())
    return result


def choose_donor(z, masks, index):
    """Nearest earlier same-mask context, independent of outcome/risk/dose."""
    candidates = []
    context = masks[index].copy()
    context[:, CHANNEL] = False
    for j in range(index):
        if (not np.array_equal(masks[j], masks[index]) or not context.any()
                or not masks[index, :, CHANNEL].any()):
            continue
        distance = float(np.sqrt(np.mean((z[j][context]-z[index][context])**2)))
        candidates.append((distance, j))
    if not candidates:
        return dict(accepted=False, reason='no_same_mask_prior')
    distance, j = min(candidates)
    observed = masks[index, :, CHANNEL]
    dose = float(np.mean(np.abs(z[j, observed, CHANNEL]-z[index, observed, CHANNEL])))
    reason = ('context_distance' if distance > CONTEXT_LIMIT else
              'insufficient_dose' if dose < MIN_DOSE else 'accepted')
    return dict(donor=j, distance=distance, dose=dose,
                accepted=reason == 'accepted', reason=reason)


def swap_observed_channel(values, donor, available):
    changed = values.clone()
    changed[..., CHANNEL] = torch.where(available[..., CHANNEL],
        donor[..., CHANNEL], values[..., CHANNEL])
    return changed


def capture(model, values, available):
    """Capture last FFN CLS branch and its residual, on the existing grad path."""
    layer = model.base.encoder.layers[-1]
    if not layer.norm_first or layer.dropout2.p != 0.:
        raise ValueError('audited pre-LN zero-dropout path required')
    saved = {}

    def save_r(_module, _inputs, output):
        saved['r'] = output[:, -1].detach().clone()

    def save_y(_module, _inputs, output):
        saved['y'] = output[:, -1].detach().clone()

    handles = [layer.linear2.register_forward_hook(save_r), layer.register_forward_hook(save_y)]
    try:
        out = model(values, available, model.initial_delta())
        logit = out['logit'].detach()
    finally:
        for handle in handles:
            handle.remove()
    return dict(logit=logit, h=saved['y']-saved['r'], r=saved['r'])


def readout(model, h, r):
    gain, bias = model.base.initial_fast(len(h)).chunk(2, -1)
    x = model.base.norm(h+r) * (1 + .1*gain.tanh()) + .1*bias.tanh()
    return model.base.risk_head(x).squeeze(-1).detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.output.with_suffix('.manifest.json')
    if args.output.exists() or manifest_path.exists():
        raise FileExistsError('preserve the locked diagnostic and any incomplete attempt')
    torch.set_num_threads(4)
    torch.manual_seed(131)
    device = 'cuda'  # Explicit local GPU experiment, no silent CPU substitution.
    paired = json.loads(args.paired.read_text())
    parent_sha = sha(args.checkpoint)
    if parent_sha != paired['source_sha256'] or not paired['verification']['full2953_verified']:
        raise ValueError('unchanged qualified source and paired full2953 required')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    base = Simple23(Config(**state['model_config']), state['source_center'], state['source_scale'])
    base.load_state_dict(state['model'], strict=True)
    model = FfnModel(base.to(device))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    threshold = float(state['threshold'])
    if threshold != paired['threshold'] or io23.RAW23[CHANNEL] != 'ne_nG':
        raise ValueError('threshold or channel contract mismatch')
    receipts = sorted(Path(paired['runs']['static']).glob('*_seal.json'))
    if len(receipts) != 2953:
        raise ValueError('complete static roster required')
    indices = stratified_indices()
    records = [json.loads(receipts[i].read_text()) for i in indices]
    ids = [int(r['shot']) for r in records]
    if ids != sorted(set(ids)):
        raise ValueError('static roster not unique ascending shot order')
    manifest = dict(scope='RETROSPECTIVE_COMPUTATION_ONLY_NOT_ADAPTATION',
        indices=[i+1 for i in indices], shots=ids, endpoints=[25, 127],
        channel='ne_nG', context_rms_limit=CONTEXT_LIMIT, min_dose=MIN_DOSE,
        logit_atol=LOGIT_ATOL, probability_atol=PROB_ATOL,
        donor_rule='nearest earlier SELECTED shot with identical mask; other22 observed RMS',
        source_sha256=parent_sha, code_sha256=sha(__file__), paired_sha256=sha(args.paired),
        mask_interventions=False, classifier_updates=False, label_selection=False,
        warning='Joint support not certified; trajectory swaps are not scale calibration. '
                'Reverse-direction swaps are retrospective controls, never deployable actions.')
    with manifest_path.open('x') as handle:
        json.dump(manifest, handle, indent=2)
    start = time.monotonic()
    windows, masks, timelines, sealed, provenance = [], [], [], [], []
    for index, receipt in zip(indices, records):
        sid = int(receipt['shot'])
        s = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        npz = receipts[index].parent / receipt['npz']
        if sha(npz) != receipt['sha256'] or float(receipt['threshold']) != threshold:
            raise ValueError('static seal changed')
        with np.load(npz, allow_pickle=False) as trace:
            if int(trace['shot']) != sid or not np.array_equal(trace['times'], s['times']):
                raise ValueError('static identity/time axis changed')
            sealed.append(trace['scores'][:128].copy())
        if len(s['times']) < 128:
            raise ValueError('prelocked endpoint unavailable; no silent shot replacement')
        pair = [simple_data.window(s, t, 128) for t in manifest['endpoints']]
        windows.append([p[0] for p in pair])
        masks.append([p[1] for p in pair])
        timelines.append(s['times'][:128].copy())
        provenance.append(dict(shot=sid, data_sha256=sha(io23.path_for('EAST', sid)),
                               static_seal_sha256=receipt['sha256']))
    windows, masks = np.asarray(windows), np.asarray(masks)
    clean = np.where(masks, windows, np.asarray(state['source_center']))
    z = np.arcsinh((clean - np.asarray(state['source_center']))/np.asarray(state['source_scale']))
    match_rows, baseline_rows, effects = [], [], []
    identity_error, seal_error = 0., 0.
    # All matches are frozen before a model output or outcome enters this audit.
    for e, endpoint in enumerate(manifest['endpoints']):
        for i, sid in enumerate(ids):
            match_rows.append(dict(shot=sid, endpoint=endpoint, index=i, endpoint_index=e,
                                    **choose_donor(z[:, e], masks[:, e], i)))
    for item in match_rows:
        i, e, t = item['index'], item['endpoint_index'], item['endpoint']
        x = torch.as_tensor(windows[i:i+1, e], device=device)
        m = torch.as_tensor(masks[i:i+1, e], device=device)
        observed = capture(model, x, m)
        p = float(observed['logit'].sigmoid())
        identity_error = max(identity_error, float((readout(model, observed['h'], observed['r'])
                                                   - observed['logit']).abs().max()))
        seal_error = max(seal_error, abs(p-float(sealed[i][t])))
        baseline_rows.append(dict(shot=ids[i], endpoint=t, probability=p,
            archived_probability=float(sealed[i][t]), time_s=float(timelines[i][t]),
            archived_prefix_alarm=bool((sealed[i][:t+1] >= threshold).any())))
        if not item['accepted']:
            continue  # Explicitly reported unmatched cases, not a zero-effect imputation.
        j = item['donor']
        # Time-grid discrepancies would invalidate timepoint-wise exchanges.
        a, b = timelines[i][:t+1], timelines[j][:t+1]
        if not np.allclose(a-a[-1], b-b[-1], atol=1e-3, rtol=0):
            raise ValueError('matched prefix time grids differ by >1ms')
        for receiver, donor, direction in ((i, j, 'prior_to_current'), (j, i, 'reverse_control')):
            xx = torch.as_tensor(windows[receiver:receiver+1, e], device=device)
            dd = torch.as_tensor(windows[donor:donor+1, e], device=device)
            mm = torch.as_tensor(masks[receiver:receiver+1, e], device=device)
            original = capture(model, xx, mm)
            changed = capture(model, swap_observed_channel(xx, dd, mm), mm)
            effects.append(dict(receiver=ids[receiver], donor=ids[donor], endpoint=t,
                direction=direction, context_distance=item['distance'], dose=item['dose'],
                base_logit=float(original['logit']), changed_logit=float(changed['logit']),
                branch_only_logit=float(readout(model, original['h'], changed['r'])),
                residual_only_logit=float(readout(model, changed['h'], original['r']))))
    if identity_error > LOGIT_ATOL or seal_error > PROB_ATOL:
        raise ValueError(f'identity parity failed: readout={identity_error}, seal={seal_error}')
    if sha(args.checkpoint) != parent_sha or not all(
            torch.equal(v, before[k]) for k, v in model.state_dict().items()):
        raise RuntimeError('frozen source changed')
    # Outcomes enter only now, for retrospective interpretation of fixed cases.
    outcomes = {sid: io23.read_outcome('EAST', sid) for sid in ids}
    for row in baseline_rows:
        disruptive, down = outcomes[row['shot']]
        row.update(disruptive=bool(disruptive), lead_ms=(down-row['time_s'])*1000.
                   if disruptive else None)
    counts = {str(t): {reason: sum(r['endpoint'] == t and r['reason'] == reason for r in match_rows)
              for reason in ('accepted', 'context_distance', 'insufficient_dose', 'no_same_mask_prior')}
              for t in manifest['endpoints']}
    report = dict(manifest=manifest, manifest_sha256=sha(manifest_path), threshold=threshold,
        gpu=torch.cuda.get_device_name(), runtime_s=time.monotonic()-start,
        matching=counts, matches=match_rows, baseline=baseline_rows, interventions=effects,
        provenance=provenance, identity_max_abs_logit_error=identity_error,
        static_seal_max_abs_probability_error=seal_error, frozen_source_unchanged=True,
        scope='Sparse endpoint sensitivity, NOT full-shot TPR/FPR or physical causality',
        conclusions_not_tested=['mask mechanism', 'scale calibration', 'online adaptation benefit',
                                'ability to preserve timely disruption warnings'])
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in ('matching', 'gpu', 'runtime_s',
        'identity_max_abs_logit_error', 'static_seal_max_abs_probability_error')}), flush=True)


if __name__ == '__main__':
    main()
