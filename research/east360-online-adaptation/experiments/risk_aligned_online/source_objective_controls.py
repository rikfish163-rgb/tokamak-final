"""Paired source-objective/history diagnostics; no EAST optimization.

Four copies receive the same 512 source minibatches. Each is evaluated on full
TEST220/TEST360 irrespective of its source score. These retrospective controls
explain a failed preparation recipe; none is automatically promoted and no
static improvement is called an intra-shot gain.
"""
import argparse
from copy import deepcopy
from functools import partial
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

from . import simple_data, simple_train
from .aligned_encoder import encoder_parameters, source_step, predict_shot
from .aligned_encoder_experiment import source_test
from .cross_channel_stable import FourViewModel, HEAD
from .fixed_test_error_audit import diagnose, paired_changes
from .fixed_test_protocol import load_test_contract, score_fixed_test
from .operating_curve_audit import load_sealed
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent, load_source_train
from .redesign_20261004.run import run_arm
from .simple_online import _durable_json, _hash

CONTROLS = {
    'joint_mixed': dict(objective='joint', short_history=True),
    'no_signal_mixed': dict(objective='no_signal', short_history=True),
    'no_supervised_mixed': dict(objective='no_supervised', short_history=True),
    'joint_full': dict(objective='joint', short_history=False),
}
LENGTHS = (8, 16, 32, 64, 128, 128, 128, 128)


def controlled_mask(available, step, *, short_history):
    if (available.dtype != torch.bool or available.ndim != 3
            or available.shape[1:] != (128, 23) or type(step) is not int
            or step < 0 or type(short_history) is not bool):
        raise ValueError('explicit history control and source availability required')
    length = LENGTHS[step % len(LENGTHS)] if short_history else 128
    mask = available.clone()
    mask[:, :128-length] = False
    return mask, length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent-run', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('new output directory required')
    if not torch.cuda.is_available():
        raise RuntimeError('rl_env CUDA required')
    historical = json.loads((args.reference/'completed.json').read_text())
    if historical['status'] != 'COMPLETED' or historical['meets_intra_goal']:
        raise ValueError('completed failed source preparation is required')
    gate = json.loads((args.reference/'source_gate.json').read_text())
    if _hash(args.reference/'prepared.pt') != gate['candidate_sha256']:
        raise ValueError('historical candidate hash changed')
    reference = torch.load(args.reference/'prepared.pt', map_location='cpu', weights_only=True)
    if reference['parent_sha256'] != PARENT_SHA256:
        raise ValueError('historical preparation used a different source parent')
    torch.set_num_threads(4)
    torch.manual_seed(131)
    rng = np.random.default_rng(131)
    base, parent = load_parent(DEFAULT_PARENT)
    teacher = FourViewModel(base, site='embedding').cuda().eval().requires_grad_(False)
    head = torch.load(HEAD, map_location='cpu', weights_only=True)
    if head['parent_sha256'] != PARENT_SHA256 or head['branch'] != 'signal':
        raise ValueError('original source-only reconstruction head required')
    teacher.decoder.load_state_dict(head['decoder'])
    original = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
    models, optimizers = {}, {}
    for name in CONTROLS:
        student = deepcopy(teacher)
        params = encoder_parameters(student)
        for parameter in params.values():
            parameter.requires_grad_(True)
        student.decoder.requires_grad_(True)
        models[name] = student
        optimizers[name] = torch.optim.AdamW([
            dict(params=list(params.values()), lr=1e-5),
            dict(params=list(student.decoder.parameters()), lr=1e-4),
        ], weight_decay=0.)
    threshold = float(parent['threshold'])
    args.output.mkdir(parents=True)
    started = time.monotonic()
    root = Path(__file__).parent
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in ('source_objective_controls.py', 'aligned_encoder.py', 'aligned_encoder_experiment.py',
                     'fixed_test_protocol.py', 'fixed_test_error_audit.py', 'operating_curve_audit.py',
                     'cross_channel_prefix.py', 'cross_channel_stable.py', 'cross_channel_shot_screen.py',
                     'simple23.py', 'simple_data.py', 'simple_train.py', 'simple_online.py', 'io23.py',
                     'causal_input.py', 'redesign_20261004/prepare_prior.py',
                     'redesign_20261004/intra.py', 'redesign_20261004/run.py'):
            archive.write(root/name, name)
    roles = simple_data.source_roles()
    _durable_json(args.output/'protocol.json', dict(
        scope='RETROSPECTIVE SOURCE-OBJECTIVE CONTROLS; FULL DEVELOPMENT-EXPOSED TESTS',
        parent_sha256=PARENT_SHA256, auxiliary_head_sha256=_hash(HEAD),
        reference_candidate_sha256=gate['candidate_sha256'], controls=CONTROLS,
        code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'), source_train_ids=roles['train'],
        steps=512, batch_size=32, seed=131, encoder_lr=1e-5, decoder_lr=1e-4,
        weight_decay=0., gradient_clip=1., short_history_lengths=LENGTHS,
        signal_mask_phase='step % 4; held identical across all controls',
        common_batches=True, feature_alignment='64D squared error to frozen original',
        original_risk_head_frozen=True, source_threshold=threshold, threshold_reselection=False,
        source_test=load_test_contract('J-TEXT'), target_test=load_test_contract('EAST'),
        target_training=False, intra=False, inter=False, source_gate_filters_east=False,
        selection='all four declared controls; no candidate promotion',
        interpretation='single-factor removal versus joint_mixed, not a complete interaction factorial'))
    print(json.dumps(dict(stage='loading_source_train', shots=len(roles['train']))), flush=True)
    shots, ids = load_source_train()
    if ids != roles['train']:
        raise AssertionError('source TRAIN identity changed')
    pools = simple_train.pools(shots)
    histories = {name: [] for name in CONTROLS}
    batches = hashlib.sha256()
    for step in range(512):
        x, available, y, _, _ = simple_train.batch(shots, pools, rng, 32, teacher.cfg, 'cuda')
        for tensor in (x, available, y):
            batches.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        for name, control in CONTROLS.items():
            mask, length = controlled_mask(available, step, short_history=control['short_history'])
            row = source_step(models[name], teacher, optimizers[name], x, mask, y.float(),
                              phase=step % 4, objective=control['objective'])
            histories[name].append(dict(step=step+1, length=length, **row))
        if (step+1) % 64 == 0:
            _durable_json(args.output/'training.json', histories)
            print(json.dumps(dict(stage='train_controls', step=step+1,
                                  losses={k: v[-1]['total'] for k, v in histories.items()})), flush=True)
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    reproduction = {}
    for key, value in models['joint_mixed'].state_dict().items():
        old = reference['model'][key]
        diff = float((value.detach().cpu().double()-old.double()).abs().max())
        if diff:
            reproduction[key] = diff
    _durable_json(args.output/'training_audit.json', dict(
        source_batches_sha256=batches.hexdigest(), original_teacher_unchanged=True,
        joint_reference_tensor_identical=not reproduction,
        joint_reference_max_abs_difference=max(reproduction.values(), default=0.),
        joint_reference_changed_tensors=reproduction))
    order = [dict(shot=sid) for sid in load_test_contract('EAST')['shot_ids']]
    provider = SimpleNamespace(read_shot=partial(simple_data.read_shot, rate_contract='nominal'),
                               attach_outcome=simple_data.attach_outcome, window=simple_data.window)
    # Outcomes below are evaluation-only and never passed into the predictor.
    signals = [simple_data.attach_outcome(provider.read_shot('EAST', r['shot']), 'EAST', {}) for r in order]
    parent_rows, _ = load_sealed(args.parent_run/'east_test360/stream', order, signals, threshold)
    parent_analysis = diagnose(parent_rows, threshold)
    analyses, results = {}, {}
    for name, student in models.items():
        student.requires_grad_(False)
        folder = args.output/name
        folder.mkdir()
        state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        changed = [k for k, v in state.items() if not torch.equal(v, original[k].cpu())]
        if not changed or any(not k.startswith(('base.embedding.', 'base.encoder.', 'base.norm.', 'decoder.'))
                              for k in changed):
            raise AssertionError('undeclared or missing source update')
        with (folder/'candidate.pt').open('xb') as handle:
            torch.save(dict(model=state, parent_sha256=PARENT_SHA256, control=CONTROLS[name]), handle)
        print(json.dumps(dict(stage='full_test', control=name)), flush=True)
        source = source_test(student, folder/'jtext_test220', threshold)
        emitted = run_arm(student, order=order, arm='static', threshold=threshold,
                          output_dir=folder/'stream', stride=32, data_api=provider, predictor=predict_shot)
        rows, seals = load_sealed(folder/'stream', order, signals, threshold)
        filtered = []
        for row in rows:
            valid = row['valid'].copy()
            if row['is_disrupt']:
                valid &= row['times'] <= row['down']
            filtered.append(dict(row, times=row['times'][valid], scores=row['scores'][valid]))
        if score_fixed_test('EAST', filtered, threshold) != emitted['metrics']:
            raise AssertionError('full sealed test recount differs')
        analysis = diagnose(rows, threshold)
        analyses[name] = analysis
        paired = paired_changes(parent_analysis, analysis)
        for key, value in student.state_dict().items():
            torch.testing.assert_close(value.cpu(), state[key], rtol=0, atol=0)
        if emitted['accepted_intra_updates'] or emitted['accepted_inter_updates']:
            raise AssertionError('source-objective diagnostic has no online update')
        results[name] = dict(source={k: v for k, v in source.items() if k != 'predictions'},
                            east=analysis['metrics'], early_fp=analysis['fp_before_elapsed_ms'],
                            paired_original={k: v for k, v in paired.items() if k != 'changed_shots'})
        _durable_json(folder/'verification.json', dict(seals=seals, parameters_unchanged=True,
                      diagnostic=analysis, paired_original=paired, candidate_sha256=_hash(folder/'candidate.pt')))
        _durable_json(args.output/'progress.json', results)
        print(json.dumps(dict(stage='control_done', control=name, results=results[name])), flush=True)
    pairs = {name: paired_changes(analyses['joint_mixed'], analysis) for name, analysis in analyses.items()}
    if _hash(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError('original source changed')
    _durable_json(args.output/'completed.json', dict(
        status='COMPLETED', results=results, paired_joint=pairs,
        full_test_curves=4*(220+360), elapsed_seconds=time.monotonic()-started,
        source_file_unchanged=True, source_batches_sha256=batches.hexdigest(),
        joint_reference_tensor_identical=not reproduction, no_online_adaptation_claim=True,
        independent_confirmation=False, promoted_candidate=None))
    print(json.dumps(dict(stage='completed', results=results)), flush=True)


if __name__ == '__main__':
    main()
