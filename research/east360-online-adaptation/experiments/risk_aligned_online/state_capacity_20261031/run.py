"""Known-label point-capacity diagnostics; never produces an online score."""
import argparse
import copy
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import load_context_memory
from ..delayed_recurrent_memory_20261025.learning import batch
from ..east_context_operator_20261031.run import (
    COLD_SHA, REFERENCE, REFERENCE_SHA, configure, install_operator, read,
)
from ..lifecycle_stream_20261025.run import build_adapter
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .optimize import probe, risk_with_shift
from .prepare import enumerate_errors, prepare_cases

OPERATOR = io23.ROOT/'probe_outputs/east_operator360_v1'
OPERATOR_SHA = '3edfa9b2a3025f327533ab252b61cde95d550f219447215eccce90a8cf147c33'
PLAN = io23.ROOT/'STATE_CAPACITY_PLAN_20261031.md'


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k:cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu(v) for v in value]
    return value


@torch.no_grad()
def replay_witnesses(models, cases, data, outputs, mapping, threshold):
    records = []
    for i, case in enumerate(cases):
        raw = apply_input_contract(simple_data.read_shot('EAST', case['shot'], rate_contract='nominal'))
        signal = transform({k:raw[k] for k in ('values', 'available')}, mapping)
        ends = range(case['original_batch_start'], case['original_batch_stop'])
        model = models['source_operator']
        x, available = batch(signal, ends, model.base.center.device)
        h, valid = model.tokens(x, available)
        j = case['endpoint']-case['original_batch_start']
        for operator, result in outputs.items():
            previous_loss = None
            for path, found in result['paths'].items():
                shift = found['best_shift'][i]
                radius = float(result['radius'][i])
                assert torch.isfinite(shift).all() and float(shift.norm()) <= radius+2e-6*max(1., radius)
                assert torch.count_nonzero(shift[~valid[j, :-1]]) == 0
                if not case['enabled']:
                    assert torch.count_nonzero(shift) == 0
                if path == 'context8':
                    assert float(found['best_parameter'][i].norm()) <= 1.+1e-6
                injected = h.new_zeros(len(h), 16, 64)
                injected[j] = shift
                z = risk_with_shift(models[operator], h, valid, injected)[j]
                probability = float(z.sigmoid())
                batched_probability = float(found['best_logits'][i].sigmoid())
                drift = abs(probability-batched_probability)
                if drift > 2e-6:
                    raise AssertionError(('original-batch witness mismatch', case['shot'], operator, path, drift))
                satisfied = probability >= threshold if case['disruptive'] else probability < threshold
                loss = float(torch.nn.functional.binary_cross_entropy_with_logits(z, data['labels'][i]))
                if previous_loss is not None:
                    assert loss <= previous_loss+2e-6, ('nested witness loss', case['shot'], operator, path)
                previous_loss = loss
                parameter = found['best_parameter'][i]
                actual = data['operators'][operator]['actual_context'][i]
                cosine = None
                if path == 'context8' and float(parameter.norm()*actual.norm()) > 0:
                    cosine = float(torch.dot(parameter, actual)/(parameter.norm()*actual.norm()))
                records.append(dict(shot=case['shot'], operator=operator, path=path,
                    disruptive=case['disruptive'], endpoint=case['endpoint'], enabled=case['enabled'],
                    prior_alarm_before_first_effect=case['prior_alarm_before_first_effect'],
                    label_disagreement=case['label_disagreement'],
                    original_cold_probability=case['cold_probability'],
                    witness_probability=probability, satisfies_selected_endpoint=satisfied,
                    original_batch_probability_drift=drift,
                    near_threshold=abs(probability-threshold) <= 2e-6,
                    shift_norm=float(shift.norm()), shift_radius=radius,
                    parameter_norm=float(parameter.norm()),
                    actual_context_norm=float(actual.norm()), context_cosine_to_actual=cosine,
                    best_init=int(found['best_init'][i]), best_step=int(found['best_step'][i])))
    return records


def main(root):
    if root.exists():
        raise FileExistsError(root)
    configure()
    start = time.monotonic()
    assert sha(REFERENCE/'completed.json') == REFERENCE_SHA
    assert sha(REFERENCE/'models/cold.pt') == COLD_SHA
    assert sha(OPERATOR/'completed.json') == OPERATOR_SHA
    assert read(OPERATOR/'verification.json')['complete']
    p = read(REFERENCE/'protocol.json')
    base, _ = load_parent(DEFAULT_PARENT)
    source = build_adapter(base.cuda(), readout='full', feedback='progressive', context=True)
    load_context_memory(source, Path(p['source_memory']['path']))
    source.restore(torch.load(REFERENCE/'models/cold.pt', map_location='cuda', weights_only=True))
    learned = copy.deepcopy(source)
    activation = read(OPERATOR/'activation.json')
    assert sha(OPERATOR/'models/operator.pt') == activation['operator_sha256']
    install_operator(learned, torch.load(OPERATOR/'models/operator.pt', map_location='cuda',
                                        weights_only=True)['operator'])
    models = dict(source_operator=source, learned_operator=learned)
    for model in models.values():
        model.requires_grad_(False)
        model.eval()
    before = {name:{k:v.clone() for k, v in model.state_dict().items()} for name, model in models.items()}
    root.mkdir(parents=True)
    files = set(Path(__file__).parent.glob('*.py')) | {PLAN}
    for name, module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module, '__file__', None):
            files.add(Path(module.__file__))
    protocol = dict(scope='POSTHOC_KNOWN_LABEL_ENDPOINT_ORACLE_NOT_ONLINE_PERFORMANCE',
        source_sha256=PARENT_SHA256, parent=str(REFERENCE), parent_completed_sha256=REFERENCE_SHA,
        parent_cold_sha256=COLD_SHA, target_operator_completed_sha256=OPERATOR_SHA,
        target_operator_sha256=activation['operator_sha256'], plan_sha256=sha(PLAN),
        threshold=p['threshold'], steps=100, starts_per_path=2, learning_rate=.1,
        radius_calculation='float64 matrix product and SVD; final cast to model dtype',
        source_memory=p['source_memory'], seed=131, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        first20_excluded_from_optimization_not_removed_from_online_score=True,
        earliest_action_index=16, supervised_model_updates=0, independent_confirmation=False,
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files})
    _durable_json(root/'protocol.json', protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, str(f.relative_to(io23.ROOT)))
    errors, cases = enumerate_errors(REFERENCE)
    _durable_json(root/'all88_errors.json', errors)
    _durable_json(root/'cases78.json', cases)
    print(dict(stage='prepare', errors=len(errors), cases=len(cases),
               disabled=sum(not r['enabled'] for r in cases)), flush=True)
    mapping = read(REFERENCE/'input_map.json')
    data, receipts = prepare_cases(models, REFERENCE, cases, mapping)
    torch.save(cpu(data), root/'prepared.pt')
    _durable_json(root/'preparation.json', receipts)
    outputs, progress_rows = {}, []
    for name, model in models.items():
        def progress(row):
            record = dict(operator=name, seconds=time.monotonic()-start, **row)
            progress_rows.append(record)
            _durable_json(root/'progress.json', record)
            _durable_json(root/'optimization_progress.json', progress_rows)
            print(record, flush=True)

        result = probe(model, data['h'], data['valid'], data['operators'][name]['basis'],
                       data['operators'][name]['actual_context'], data['labels'],
                       enabled=data['enabled'], threshold=p['threshold'],
                       steps=100, lr=.1, progress=progress)
        zero_error = max(abs(float(z)-case['cold_probability'])
                         for z, case in zip(result['zero_logits'].sigmoid(), cases, strict=True))
        assert zero_error <= 2e-6, ('batched zero probability', zero_error)
        torch.save(cpu(result), root/f'{name}.pt')
        outputs[name] = result
    records = replay_witnesses(models, cases, data, outputs, mapping, p['threshold'])
    _durable_json(root/'witness_replays.json', records)
    assert all(torch.equal(value, before[name][key]) for name, model in models.items()
               for key, value in model.state_dict().items())
    assert sha(DEFAULT_PARENT) == PARENT_SHA256 and sha(REFERENCE/'models/cold.pt') == COLD_SHA
    summaries = {}
    for name in models:
        summaries[name] = {}
        for path in ('context8', 'free_last', 'free_sequence'):
            group = [r for r in records if r['operator'] == name and r['path'] == path]
            summaries[name][path] = dict(
                selected_positive_endpoints=36, selected_negative_endpoints=42,
                positive_endpoint_targets_met=sum(r['disruptive'] and r['satisfies_selected_endpoint'] for r in group),
                negative_endpoint_targets_met=sum(not r['disruptive'] and r['satisfies_selected_endpoint'] for r in group),
                negative_met_without_prior_early_alarm=sum(not r['disruptive'] and r['satisfies_selected_endpoint']
                    and not r['prior_alarm_before_first_effect'] for r in group),
                disabled_endpoints=sum(not r['enabled'] for r in group),
                median_witness_shift_norm=float(np.median([r['shift_norm'] for r in group])),
                near_threshold_witnesses=sum(r['near_threshold'] for r in group))
    _durable_json(root/'completed.json', dict(status='COMPLETED_POSTHOC_DIAGNOSTIC',
        not_online_performance=True, not_a_global_capacity_bound=True, not_a_deployable_policy=True,
        startup_errors_retained=10, selected_cases=78, witness_replays=468, summaries=summaries,
        model_tensors_unchanged=True, seconds=time.monotonic()-start,
        prepared_sha256=sha(root/'prepared.pt'), witness_replays_sha256=sha(root/'witness_replays.json'),
        independent_confirmation=False))
    print(summaries, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
