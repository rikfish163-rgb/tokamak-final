"""Fixed-budget source experiment for partial feedback and its delayed control.

Same qualified classifier, original auxiliary head, 60 source fit pairs and 32
source selection shots as the previous matched experiment. Reuse its completed
signal-only heads (20/40/60), not extra fitting or target-driven selection.
"""
import argparse
import json
import math
from pathlib import Path
import time
import zipfile

import torch

from ..simple_online import _hash, _durable_json
from ..redesign_20261004.prepare_prior import (
    load_parent, load_source_train, classifier_state, assert_classifier_unchanged, DEFAULT_PARENT,
)
from ..ffn_intra_20261005.model import FfnModel
from ..ffn_intra_20261005.source import split_shots
from ..decoder_intra_20261005.matched_meta import enable_head_only
from ..decoder_intra_20261005.prepare import risk_probe
from ..decoder_intra_20261005.prepare_whole import publish_admitted
from .source import selection_report, train_pair


def admission(candidate, references, initial_maintenance, delayed, signal_only):
    def better_than(ref):
        c, r = candidate['whole'], ref['whole']
        return (c['metrics30']['tp'] >= r['metrics30']['tp']
                and c['metrics30']['fp'] <= r['metrics30']['fp']
                and c['balanced_loss'] < r['balanced_loss']
                and candidate['later']['mean_loss'] < ref['later']['mean_loss'])
    source_gate = (candidate['maintenance'] <= 1.10 * initial_maintenance
                   and all(better_than(r) for r in references))
    clock_gain = better_than(delayed)
    risk_training_gain = better_than(signal_only)
    return dict(source_gate=bool(source_gate), clock_gain=bool(clock_gain),
                risk_training_gain=bool(risk_training_gain),
                admitted=bool(source_gate and clock_gain and risk_training_gain))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matched', type=Path, required=True)
    p.add_argument('--whole', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    old = json.loads((args.matched/'summary.json').read_text()); previous = old['protocol']
    whole = json.loads((args.whole/'summary.json').read_text())
    parent_hash = _hash(DEFAULT_PARENT)
    if (not previous['source_only'] or previous['parent_sha256'] != parent_hash
            or whole['protocol']['parent_sha256'] != parent_hash
            or (previous['steps'], previous['lr'], previous['step'], previous['radius'], previous['stride'])
            != (60, .00003, .1, 1., 26)):
        raise ValueError('completed fixed-budget original-head source references required')
    if whole['protocol']['matched_summary_sha256'] != _hash(args.matched/'summary.json'):
        raise ValueError('whole reference was not the same matched experiment')
    signal_paths = {i: args.matched/'signal_only'/f'candidate_{i:03d}.pt' for i in (20, 40, 60)}
    if not all(path.is_file() for path in signal_paths.values()):
        raise ValueError('all three fixed-budget signal-only controls required')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(previous['seed']); started = time.monotonic()
    shots, ids = load_source_train(); lookup = {s['shot']: s for s in shots}
    fit, selection = split_shots(shots, previous['seed'])
    if (ids != previous['source_train_ids'] or [s['shot'] for s in fit] != previous['fit_ids']
            or sorted(s['shot'] for s in selection) != previous['selection_ids']):
        raise ValueError('canonical source split changed')
    val = [(lookup[e['shot']], e) for e in previous['validation_episodes']]
    fit_pairs = [[(lookup[e['shot']], e) for e in pair] for pair in previous['fit_episodes']]
    if (len(val) != 32 or len(fit_pairs) != 60
            or not {s['shot'] for s, _ in val} <= set(previous['selection_ids'])
            or any(s['shot'] not in previous['fit_ids'] for pair in fit_pairs for s, _ in pair)):
        raise ValueError('exact disjoint 60 source pairs / 32 selection shots required')
    base, state = load_parent(DEFAULT_PARENT); base = base.to(args.device)
    model = FfnModel(base); enable_head_only(model); before = classifier_state(base)
    original = {k: v.detach().cpu().clone() for k, v in base.forecast_head.state_dict().items()}
    threshold = float(state['threshold']); boundary = math.log(threshold/(1-threshold))
    reference_risk = risk_probe(model, val[0][0])
    root = Path(__file__).parents[1]
    dependencies = [root/f for f in ('simple23.py', 'simple_data.py', 'simple_train.py', 'simple_online.py', 'fast_update.py', 'io23.py')]
    dependencies += [f for folder in ('partial_intra_20261005', 'decoder_intra_20261005', 'ffn_intra_20261005', 'redesign_20261004')
                     for f in (root/folder).glob('*.py') if not f.name.startswith('test_')]
    code = {str(f.resolve()): _hash(f) for f in dependencies}
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for f in dependencies:
            archive.write(f, str(f.relative_to(root)))
    protocol = dict(previous, arms=['partial_signal_only_reused', 'partial_risk_aligned', 'same_head_delayed'],
        objective='partial whole-shot causal source risk + .20 unchanged zero-state H25 prediction',
        outer_positive='whole: all source D endpoints strict lead>30ms, no125ms upper bound; original source point labels unchanged',
        later_positive='diagnostic inherited episode queries: strict >30ms and <=125ms; actual emitted risk, not training peak scope',
        probability_loss='whole and later use emitted float32 probability, effective float32 clip[1e-7,1-1e-7], then threshold-relative softplus',
        clocks={'partial': [5, 25], 'delayed': [25, 25], 'snapshots': [5, 25], 'first_effect': [6, 26]},
        segment_steps=[.02, .08], radius=1., origin_stride=26, target_level_partition=['h1..5', 'h6..25'],
        difference_partition=['1->2 through 4->5', '5->6 through 24->25'],
        differentiated_updates='last at most4 executed signal events; real earlier burn-in detached',
        derivative_duration='not matched duration to previous four H25 events; same truncation within P/D',
        matched_summary_sha256=_hash(args.matched/'summary.json'), whole_summary_sha256=_hash(args.whole/'summary.json'),
        reused_signal_head_sha256={str(path.resolve()): _hash(path) for path in signal_paths.values()},
        unchanged_classifier=True, alarm_tuning=False, threshold=threshold, code_sha256=code,
        gate='whole+causal-later improve vs static, initialP, originalH25 and whole60; TP>=,FP<=,maintenance<=1.10 initial; additionally beats same-headD and same-budget signal-onlyP',
        no_east_reads='source-only loader/code path, not an independent filesystem sandbox',
        torch_version=str(torch.__version__), hardware=torch.cuda.get_device_name() if args.device.startswith('cuda') else 'cpu',
        interpretation='development only; no EAST gain, independent validation or physics discovery established')
    _durable_json(args.output/'protocol.json', protocol)
    baselines = {}
    for mode in ('static', 'partial', 'delayed'):
        baselines[mode] = selection_report(model, val, threshold, mode=mode)
        print(json.dumps(dict(stage='baseline', mode=mode, whole=baselines[mode]['whole']['balanced_loss'],
            later=baselines[mode]['later']['mean_loss'], tp=baselines[mode]['whole']['metrics30']['tp'],
            fp=baselines[mode]['whole']['metrics30']['fp'])), flush=True)
    for part, key in [('whole', 'balanced_loss'), ('later', 'mean_loss')]:
        if abs(baselines['static'][part][key] - old['static'][part][key]) > 2e-6:
            raise RuntimeError('static source baseline differs from reference')
    _durable_json(args.output/'baseline.json', baselines)
    signal_reports = {}
    for iteration, path in signal_paths.items():
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['arm'] != 'signal_only' or checkpoint['iteration'] != iteration or checkpoint['protocol'] != previous:
            raise ValueError('signal-only checkpoint identity/provenance mismatch')
        base.forecast_head.load_state_dict(checkpoint['forecast_head'])
        signal_reports[str(iteration)] = {mode: selection_report(model, val, threshold, mode=mode)
                                         for mode in ('partial', 'delayed')}
        print(json.dumps(dict(stage='signal_control', iteration=iteration,
            whole=signal_reports[str(iteration)]['partial']['whole']['balanced_loss'])), flush=True)
    _durable_json(args.output/'signal_controls.json', signal_reports)
    base.forecast_head.load_state_dict(original); enable_head_only(model)
    optimizer = torch.optim.Adam(base.forecast_head.parameters(), lr=.00003)
    references = [baselines['static'], baselines['partial'], old['initial'], whole['selections'][-1]]
    history = []; selections = []; best = None; best_step = None; best_loss = float('inf')
    for iteration, pair in enumerate(fit_pairs, 1):
        row = train_pair(model, pair, optimizer, objective='risk_aligned', boundary=boundary)
        history.append(dict(iteration=iteration, **row))
        if iteration % 10 == 0:
            print(json.dumps(dict(stage='partial_fit', iteration=iteration, objective=row['objective'],
                                  elapsed_seconds=time.monotonic()-started)), flush=True)
        if iteration % 20 == 0:
            report = selection_report(model, val, threshold, mode='partial')
            delayed = selection_report(model, val, threshold, mode='delayed')
            gates = admission(report, references, baselines['partial']['maintenance'], delayed,
                              signal_reports[str(iteration)]['partial'])
            current = {k: v.detach().cpu().clone() for k, v in base.forecast_head.state_dict().items()}
            torch.save(dict(forecast_head=current, protocol=protocol, iteration=iteration,
                            source_gate_passed=gates['admitted']), args.output/f'candidate_{iteration:03d}.pt')
            selections.append(dict(iteration=iteration, partial=report, delayed=delayed, **gates))
            if gates['admitted'] and report['whole']['balanced_loss'] < best_loss:
                best = current; best_step = iteration; best_loss = report['whole']['balanced_loss']
            print(json.dumps(dict(stage='partial_selection', iteration=iteration, **gates,
                whole=report['whole']['balanced_loss'], later=report['later']['mean_loss'],
                delayed_whole=delayed['whole']['balanced_loss'], maintenance=report['maintenance'],
                tp=report['whole']['metrics30']['tp'], fp=report['whole']['metrics30']['fp'])), flush=True)
            _durable_json(args.output/'training.json', dict(history=history, selections=selections))
    assert_classifier_unchanged(base, before)
    if not torch.equal(reference_risk, risk_probe(model, val[0][0])):
        raise RuntimeError('auxiliary fitting changed static risk output')
    external_hashes = {args.matched/'summary.json': protocol['matched_summary_sha256'],
                      args.whole/'summary.json': protocol['whole_summary_sha256'],
                      **{Path(path): h for path, h in protocol['reused_signal_head_sha256'].items()}}
    if any(_hash(path) != h for path, h in external_hashes.items()):
        raise RuntimeError('fixed reference asset changed during experiment')
    publish_admitted(args.output, best, protocol, best_step, parent_hash, code)
    _durable_json(args.output/'summary.json', dict(protocol=protocol, baselines=baselines,
        signal_controls=signal_reports, selections=selections, selected_step=best_step,
        source_gate_passed=best is not None, classifier_unchanged=True, static_risk_probe_bit_exact=True,
        elapsed_seconds=time.monotonic()-started,
        interpretation='matched source screening; no EAST inference experiment performed'))


if __name__ == '__main__':
    main()
