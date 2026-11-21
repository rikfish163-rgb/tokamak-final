"""Predeclared source-only CLS A/C/S screening; fixed final step, no target tuning."""
import argparse
import json
import math
from pathlib import Path
import time
import zipfile

import torch

from ..simple_online import _hash, _durable_json
from ..redesign_20261004.prepare_prior import load_parent, load_source_train, DEFAULT_PARENT
from ..ffn_intra_20261005.source import split_shots
from ..decoder_intra_20261005.prepare import risk_probe
from .model import ClsModel
from .ridge import fit_decoder
from .source import enable_decoder_only, selection_report, train_pair


def admission(a, c, static, initial_maintenance):
    comparisons = {}
    for name, ref in [('S', static), ('C', c)]:
        comparisons[name] = dict(tp_not_lower=a['metrics30']['tp'] >= ref['metrics30']['tp'],
            fp_not_higher=a['metrics30']['fp'] <= ref['metrics30']['fp'],
            risk_lower=a['balanced_loss'] < ref['balanced_loss'])
    maintenance = a['maintenance'] <= 1.10 * initial_maintenance
    return dict(comparisons=comparisons, maintenance_passed=maintenance,
        source_gate_passed=bool(maintenance and all(all(r.values()) for r in comparisons.values())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matched', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old = json.loads((args.matched / 'summary.json').read_text()); previous = old['protocol']
    if (not previous['source_only'] or previous['parent_sha256'] != _hash(DEFAULT_PARENT)
            or previous['steps'] != 60 or previous['lr'] != .00003):
        raise ValueError('locked matched source protocol required')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(previous['seed']); started = time.monotonic()
    shots, ids = load_source_train(); lookup = {s['shot']: s for s in shots}
    fit, selection = split_shots(shots, previous['seed'])
    if (ids != previous['source_train_ids'] or [s['shot'] for s in fit] != previous['fit_ids']
            or sorted(s['shot'] for s in selection) != previous['selection_ids']):
        raise ValueError('source split changed')
    val = [(lookup[e['shot']], e) for e in previous['validation_episodes']]
    fit_pairs = [[(lookup[e['shot']], e) for e in pair] for pair in previous['fit_episodes']]
    if (len(fit) != 1083 or len(val) != 32 or len(fit_pairs) != 60
            or not {s['shot'] for s, _ in val} <= set(previous['selection_ids'])
            or any(s['shot'] not in previous['fit_ids'] for pair in fit_pairs for s, _ in pair)):
        raise ValueError('disjoint fixed source fit and selection identities required')
    base, state = load_parent(DEFAULT_PARENT); model = ClsModel(base.to('cuda'))
    enable_decoder_only(model)
    before = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    original_probe = risk_probe(model, val[0][0])
    threshold = float(state['threshold']); boundary = math.log(threshold / (1 - threshold))
    root = Path(__file__).parents[1]
    dependencies = [root/f for f in ('simple23.py', 'simple_data.py', 'simple_train.py', 'simple_online.py', 'fast_update.py', 'io23.py')]
    dependencies += [f for folder in ('cls_intra_20261005', 'partial_intra_20261005', 'decoder_intra_20261005', 'ffn_intra_20261005', 'redesign_20261004')
                     for f in (root/folder).glob('*.py') if not f.name.startswith('test_')]
    code = {str(f.resolve()): _hash(f) for f in dependencies}
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for f in dependencies: archive.write(f, str(f.relative_to(root)))
    protocol = dict(parent_sha256=_hash(DEFAULT_PARENT), matched_summary_sha256=_hash(args.matched/'summary.json'),
        source_only=True, source_train_ids=ids, fit_ids=previous['fit_ids'], selection_ids=previous['selection_ids'],
        fit_episodes=previous['fit_episodes'], validation_episodes=previous['validation_episodes'],
        seed=previous['seed'], steps=60, lr=.00003, threshold=threshold, arms=['S', 'C', 'A'],
        decoder='same risk-head CLS, linear64->575', decoder_parameters=37375,
        ridge='all1083 source-fit shots, complete H25 stride26; mean MSE + .001 weight penalty, intercept unpenalized',
        C='arrived signal only inner feedback; whole-source-risk + .20 maintenance outer',
        A='same inner plus same-endpoint frozen task metric; identical outer budget',
        segment_steps=[.02, .08], radius=1., stride=26, earliest_effect=[6, 26],
        source_positive_outer='all eligible source D endpoints strict lead>30ms; unchanged original max25/AdvancedTime labels',
        truncation='last4 executed updates; earlier causal burn-in detached',
        selection='fixed final step60 only; source32 is a development screen, not independent validation',
        gate='A TP>=S,C; FP<=S,C; whole balanced risk<S,C; maintenance<=1.10 ridge initial',
        risk_parameters_updated=False, alarm_parameters_updated=False, target_labels_accessed=0,
        code_sha256=code, torch_version=str(torch.__version__), hardware=torch.cuda.get_device_name())
    _durable_json(args.output/'protocol.json', protocol)
    ridge = fit_decoder(model, fit)
    initial = {k: v.detach().cpu().clone() for k, v in model.decoder.state_dict().items()}
    torch.save(dict(decoder=initial, protocol=protocol, ridge=ridge), args.output/'ridge_initial.pt')
    _durable_json(args.output/'ridge.json', ridge)
    reports = {}
    for arm, intra, use_task in [('S', False, False), ('C0', True, False), ('A0', True, True)]:
        report = selection_report(model, val, threshold, use_task=use_task, intra=intra)
        reports[arm] = report
        print(json.dumps(dict(stage='initial', arm=arm, risk=report['balanced_loss'], maintenance=report['maintenance'],
                              tp=report['metrics30']['tp'], fp=report['metrics30']['fp'])), flush=True)
    if abs(reports['S']['balanced_loss'] - old['static']['whole']['balanced_loss']) > 2e-6:
        raise RuntimeError('static source baseline differs from existing evidence')
    _durable_json(args.output/'initial.json', reports)
    history = {}
    for arm, use_task in [('C', False), ('A', True)]:
        model.decoder.load_state_dict(initial); enable_decoder_only(model)
        optimizer = torch.optim.Adam(model.decoder.parameters(), lr=.00003)
        history[arm] = []
        for iteration, pair in enumerate(fit_pairs, 1):
            row = train_pair(model, pair, optimizer, use_task=use_task, boundary=boundary)
            history[arm].append(dict(iteration=iteration, **row))
            if iteration % 10 == 0:
                print(json.dumps(dict(stage='fit', arm=arm, iteration=iteration, objective=row['objective'],
                                      elapsed_seconds=time.monotonic()-started)), flush=True)
                _durable_json(args.output/'training.json', history)
        reports[arm] = selection_report(model, val, threshold, use_task=use_task)
        checkpoint = dict(decoder={k: v.detach().cpu().clone() for k, v in model.decoder.state_dict().items()},
                          protocol=protocol, arm=arm, iteration=60)
        torch.save(checkpoint, args.output/f'{arm}_final60.pt')
        _durable_json(args.output/f'{arm}_selection.json', reports[arm])
        print(json.dumps(dict(stage='final_selection', arm=arm, risk=reports[arm]['balanced_loss'],
            maintenance=reports[arm]['maintenance'], tp=reports[arm]['metrics30']['tp'],
            fp=reports[arm]['metrics30']['fp'])), flush=True)
    for k, value in base.state_dict().items():
        if not torch.equal(value.detach().cpu(), before[k]): raise RuntimeError(f'qualified source changed: {k}')
    if not torch.equal(original_probe, risk_probe(model, val[0][0])):
        raise RuntimeError('static risk probe changed')
    if any(_hash(Path(path)) != sha for path, sha in code.items()):
        raise RuntimeError('code changed during fixed experiment')
    gates = admission(reports['A'], reports['C'], reports['S'], reports['A0']['maintenance'])
    _durable_json(args.output/'summary.json', dict(protocol=protocol, ridge=ridge, reports=reports,
        **gates, source_all_state_bit_exact=True, static_risk_probe_bit_exact=True,
        previous_all_token_maintenance=old['initial']['maintenance'],
        elapsed_seconds=time.monotonic()-started, interpretation='source-only screen; no EAST inference performed'))
    print(json.dumps(dict(stage='complete', **gates, elapsed_seconds=time.monotonic()-started)), flush=True)


if __name__ == '__main__': main()
