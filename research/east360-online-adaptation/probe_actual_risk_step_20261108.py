"""Post-seal CPU diagnostics of the last actual update on 85 fixed error queries.

Replay every native release from the shot start; labels measure risk only.
This is neither online performance nor a causal explanation of shot outcomes.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import analyze_risk_oracle_direction_20261108 as d
from risk_feature_intra_20261107 import FeatureIntra, update

j = d.j
OUTPUT = d.FIT / 'east_actual_step_cpu.json'
REAL = 'recorded_sampling_real'


def probe_prefix(base, decoder, raw, rank, mapping, q, diagnostic_label):
    """Return JSON readings, an isolated model, and the last actual delta W.

    The caller configures CPU determinism. AdamW keeps its native fixed-size
    moments throughout this prefix; diagnostic gradients never enter .grad.
    """
    d.need(type(q) is int and 0 <= q < len(raw['values']) and rank > 20,
           'an eligible post-preparation query is required')
    d.need(diagnostic_label in (0, 1), 'binary diagnostic label required')
    d.need(all(p.device.type == 'cpu' for m in (base, decoder) for p in m.parameters()),
           'CPU templates required')
    prefix = {key: raw[key] for key in ('device', 'shot')}
    prefix.update({key: raw[key][:q+1].copy() for key in ('values', 'available', 'times')})
    signal = j.r.prepare_actual(prefix, rank, mapping)
    d.need(len(signal['values']) == q+1 and not signal['available'][:, 2].any(),
           'prefix length or native MNM mask changed')
    model = FeatureIntra(base, decoder)
    optimizer = torch.optim.AdamW([model.weight], lr=1e-4, weight_decay=0., foreach=False)
    x, a = [torch.as_tensor(v[None]) for v in j.simple_data.window(signal, q, 128)]
    label = torch.tensor([float(diagnostic_label)], dtype=torch.float32)
    releases = list(range(153, q+1, 32))
    row = dict(status='last_native_update' if releases else 'noupdate',
               updates=len(releases), releases=releases, last_release=None, origin=None,
               target_range=None, ssl_before=None, ssl_after=None,
               actual_step_risk_gradient_dot=None, actual_step_risk_gradient_cosine=None,
               actual_step_norm=0.)
    delta = torch.zeros_like(model.weight)
    with sdpa_kernel(SDPBackend.MATH):
        if not releases:
            row['before'] = row['after'] = d.output_at_query(model, x, a, label)
        for release in releases:
            episode = d.make_episode(signal, release-26, q)
            if release == releases[-1]:
                before_weight = model.weight.detach().clone()
                logit = model.risk(x, a)['logit']
                risk = F.binary_cross_entropy_with_logits(logit, label)
                gradient, = torch.autograd.grad(risk, model.weight)
                row['before'] = dict(logit=float(logit.detach()),
                                     probability=float(logit.detach().sigmoid()), bce=float(risk.detach()))
            stats = update(model, optimizer, episode, feedback='real')
        if releases:
            delta = model.weight.detach() - before_weight
            with torch.no_grad():
                ssl_after = float(model.signal_loss(episode, feedback='real'))
            d.need(bool(torch.isfinite(gradient).all() and torch.isfinite(delta).all()),
                   'nonfinite diagnostic risk gradient or actual update')
            row.update(last_release=release, origin=release-26,
                       target_range=[release-25, release-1], ssl_before=stats['signal_loss'],
                       ssl_after=ssl_after, after=d.output_at_query(model, x, a, label),
                       actual_step_risk_gradient_dot=float((gradient.double()*delta.double()).sum()),
                       actual_step_risk_gradient_cosine=d.cosine(gradient, delta),
                       actual_step_norm=float(delta.double().norm()), update_stats=stats)
    return row, model, delta


def summarize(cases):
    result = {}
    for group in ('all', 'tardy', 'miss', 'fp'):
        rows = [v for v in cases if group == 'all' or v['static_status'] == group]
        measured = [v for v in rows if v['status'] != 'noupdate']
        result[group] = dict(n=len(rows), nomatured=len(rows)-len(measured),
            ssl_decreased=sum(v['ssl_after'] < v['ssl_before'] for v in measured),
            risk_increased=sum(v['after']['bce'] > v['before']['bce'] for v in measured),
            ssl_down_risk_up=sum(v['ssl_after'] < v['ssl_before'] and
                                v['after']['bce'] > v['before']['bce'] for v in measured),
            positive_dot_risk_up=sum(v['actual_step_risk_gradient_dot'] > 0 and
                                    v['after']['bce'] > v['before']['bce'] for v in measured))
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    started, start_utc = time.monotonic(), datetime.now(timezone.utc).isoformat()
    d.configure_cpu()
    d.need(not OUTPUT.exists(), f'refusing to overwrite {OUTPUT}')
    fixed = j.r.read(d.OUTPUT)
    d.need(fixed['status'] == 'COMPLETE_CPU_POSTHOC_LOCAL_DIRECTION'
           and len(fixed['cases']) == 85, 'the sealed fixed 85 eligible cases are required')
    cfg, mapping, errors = d.load_inputs()
    by_case = {(v['rank'], v['shot']): v for v in errors}
    keys = [(v['rank'], v['shot']) for v in fixed['cases']]
    d.need(len(set(keys)) == 85 and all(k in by_case for k in keys), 'fixed case identities differ')
    files = {k: d.checkpoint_files()[k] for k in ('source', 'control')}
    hashes = {k: j.r.sha(path) for k, path in files.items()}
    base, parent = j.r.load_parent(files['source'])
    d.need(parent['threshold'] == d.THRESHOLD, 'qualified parent threshold differs')
    decoder = d.MetaOrderedTrajectory(base.center, base.scale, dim=base.cfg.dim, scenarios=4).eval()
    decoder.load_state_dict(torch.load(files['control'], map_location='cpu', weights_only=True), strict=True)
    decoder.requires_grad_(False)
    source_state, decoder_state = d.snapshot(base), d.snapshot(decoder)
    d.need(source_state[d.WEIGHT].numel() == 8192, 'the native 8192-scalar plastic matrix is required')
    cases = []
    for old in fixed['cases']:
        case, raw = d.choose_and_verify(by_case[(old['rank'], old['shot'])], cfg)
        d.need(not case['exclusion_reasons'] and all(old[k] == v for k, v in case.items()),
               f'fixed query identity changed: {old["shot"]}')
        q = case['query_index']
        readings, model, _ = probe_prefix(base, decoder, raw, case['rank'], mapping, q,
                                          case['native_y_at_query'])
        curve_file = d.SEALED / 'shots' / f'{case["rank"]:04d}_{case["shot"]}.npz'
        with np.load(curve_file, allow_pickle=False) as saved:
            reference = float(saved[REAL][q])
        difference = readings['after']['probability'] - reference
        d.need(np.isfinite(reference) and 0 <= reference <= 1 and abs(difference) <= 5e-6,
               f'sealed real parity failed: shot={case["shot"]} q={q} '
               f'cpu={readings["after"]["probability"]} sealed={reference} abs_error={abs(difference)}')
        cases.append(dict(**case, **readings, sealed_real_probability=reference,
                          sealed_real_probability_difference=difference,
                          sealed_real_probability_abs_error=abs(difference), sealed_curve_file=str(curve_file)))
        d.check_state(model.risk, source_state, excluded=(d.WEIGHT,))
        d.check_state(model.decoder, decoder_state)
        if len(cases) % 10 == 0:
            print(f'fixed actual-step queries: {len(cases)}/85', flush=True)
    d.check_state(base, source_state)
    d.check_state(decoder, decoder_state)
    d.need({k: j.r.sha(path) for k, path in files.items()} == hashes, 'source or G checkpoint changed')
    source_paths = [Path(__file__), Path(d.__file__), Path(j.__file__),
                    Path(update.__code__.co_filename), Path(d.MetaOrderedTrajectory.loss.__code__.co_filename),
                    Path(type(decoder).forward.__code__.co_filename), Path(type(base).forward.__code__.co_filename),
                    Path(j.simple_data.window.__code__.co_filename), Path(j.r.prepare_actual.__code__.co_filename)]
    result = dict(status='COMPLETE_CPU_POSTHOC_ACTUAL_RISK_STEP', started_utc=start_utc,
        seconds=time.monotonic()-started, command=[sys.executable, *sys.argv],
        inputs=dict(fixed_cases=str(d.OUTPUT), error_diagnostic=str(d.DIAGNOSTIC),
                    sealed_reference=str(d.SEALED), sealed_curve_key=REAL,
                    fixed_map=str(j.r.COLD/'input_map.json'),
                    source_and_G={k: str(v) for k, v in files.items()}, source_and_G_sha256=hashes),
        contract=dict(cpu_only=True, prefix_through_query_only=True, window=128, horizon=25,
                      releases='153,185,...<=q', origin='release-26', truth='origin+1:origin+26',
                      optimizer='native AdamW lr1e-4 wd0; accumulated constant-size moments; clip1; .01 shot-start prior',
                      diagnostic='last release only; same matured episode and same q risk window',
                      labels='post-seal diagnostic BCE and autograd.grad only; never update inputs',
                      cosine='actual delta W versus pre-update q BCE gradient; positive is locally harmful',
                      interpretation='fixed posthoc cases, not online performance or a causal conclusion'),
        verification=dict(eligible=85, excluded=len(fixed['exclusions']), real_probability_tolerance=5e-6,
                          max_real_probability_abs_error=max(v['sealed_real_probability_abs_error'] for v in cases),
                          source_and_G_unchanged=True, each_nonplastic_source_tensor_unchanged=True),
        summary=summarize(cases), cases=cases, exclusions=fixed['exclusions'],
        source_files={str(path): path.read_text() for path in source_paths})
    json.dumps(result, allow_nan=False)
    j.r.write_new(OUTPUT, result)
    print(json.dumps(dict(status=result['status'], output=str(OUTPUT), summary=result['summary'])), flush=True)


if __name__ == '__main__':
    main()
