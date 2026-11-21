"""Pure intra controls without the harmful extra source classification loss.

This runs the unchanged whole-encoder update once from the qualified original
and once from the no-supervised preparation copy. Corresponding static traces
already exist on exactly TEST360. Original and candidate are never overwritten.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

from . import simple_data
from .aligned_encoder import predict_shot
from .cross_channel_stable import FourViewModel, HEAD
from .fixed_test_error_audit import diagnose, paired_changes
from .fixed_test_protocol import load_test_contract, score_fixed_test
from .operating_curve_audit import load_sealed, operating_curve
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .redesign_20261004.run import run_arm
from .simple_online import _durable_json, _hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', choices=('original', 'no_supervised'), required=True)
    parser.add_argument('--scope', choices=('encoder', 'decoder', 'both'), default='encoder')
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--parent-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('new output directory required')
    if args.start != 'original' and args.scope != 'encoder':
        parser.error('decoder-scope comparison is declared only for the unchanged original parent')
    if not torch.cuda.is_available():
        raise RuntimeError('rl_env CUDA required')
    controls = json.loads((args.controls/'completed.json').read_text())
    control_protocol = json.loads((args.controls/'protocol.json').read_text())
    if (controls['status'] != 'COMPLETED' or not controls['source_file_unchanged']
            or control_protocol['parent_sha256'] != PARENT_SHA256):
        raise ValueError('completed same-parent source controls required')
    torch.set_num_threads(4)
    torch.manual_seed(131)
    base, parent = load_parent(DEFAULT_PARENT)
    model = FourViewModel(base, site='embedding').cuda().eval().requires_grad_(False)
    head = torch.load(HEAD, map_location='cpu', weights_only=True)
    if head['parent_sha256'] != PARENT_SHA256 or head['branch'] != 'signal':
        raise ValueError('original source-only auxiliary head required')
    model.decoder.load_state_dict(head['decoder'])
    if args.start == 'original':
        source = json.loads((args.parent_run/'completed.json').read_text())['results']['J-TEXT']
        reference_dir = args.parent_run/'east_test360/stream'
        checkpoint_sha = PARENT_SHA256
    else:
        folder = args.controls/'no_supervised_mixed'
        verification = json.loads((folder/'verification.json').read_text())
        checkpoint_sha = _hash(folder/'candidate.pt')
        if checkpoint_sha != verification['candidate_sha256']:
            raise ValueError('source candidate changed after static evaluation')
        state = torch.load(folder/'candidate.pt', map_location='cpu', weights_only=True)
        if (state['parent_sha256'] != PARENT_SHA256
                or state['control'] != dict(objective='no_supervised', short_history=True)):
            raise ValueError('only the source no-supervised control is authorized')
        for key, value in model.state_dict().items():
            if not key.startswith(('base.embedding.', 'base.encoder.', 'base.norm.', 'decoder.')):
                torch.testing.assert_close(value.cpu(), state['model'][key], rtol=0, atol=0)
        model.load_state_dict(state['model'], strict=True)
        source = controls['results']['no_supervised_mixed']['source']
        reference_dir = folder/'stream'
    if not (source['tpr'] >= .90 and source['fpr'] < .20):
        raise ValueError('complete TEST220 minimum qualification required')
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    threshold = float(parent['threshold'])
    contract = load_test_contract('EAST')
    if threshold != source['threshold'] or contract != control_protocol['target_test']:
        raise ValueError('fixed threshold/test contract differs')
    order = [dict(shot=sid) for sid in contract['shot_ids']]
    provider = SimpleNamespace(read_shot=partial(simple_data.read_shot, rate_contract='nominal'),
                               attach_outcome=simple_data.attach_outcome, window=simple_data.window)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    root = Path(__file__).parent
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in ('encoder_intra_controls.py', 'aligned_encoder.py', 'fixed_test_protocol.py',
                     'fixed_test_error_audit.py', 'operating_curve_audit.py', 'cross_channel_prefix.py',
                     'cross_channel_stable.py', 'simple23.py', 'simple_data.py', 'simple_train.py',
                     'simple_online.py', 'io23.py', 'causal_input.py', 'redesign_20261004/prepare_prior.py',
                     'redesign_20261004/intra.py', 'redesign_20261004/run.py'):
            archive.write(root/name, name)
    _durable_json(args.output/'protocol.json', dict(
        scope='PURE INTRA; FIXED DEVELOPMENT-EXPOSED TEST360', start=args.start,
        parameter_scope=args.scope,
        parent_sha256=PARENT_SHA256, checkpoint_sha256=checkpoint_sha, head_sha256=_hash(HEAD),
        source_test=source, target_test=contract, threshold=threshold,
        source_controls_completion_sha256=_hash(args.controls/'completed.json'),
        code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'),
        static_reference=str(reference_dir.resolve()),
        objective='mean four-view arrived-signal reconstruction; no target risk label',
        update_scope=dict(encoder='temporary embedding + both Transformer layers + final normalization',
                          decoder='temporary auxiliary decoder only; risk path unchanged',
                          both='temporary encoder and auxiliary decoder')[args.scope],
        update_lr=.001, gradient_clip=1., parameter_radius=.5, first_update_index=7, update_stride=32,
        clipping_and_radius_are_per_group=True,
        risk_head_frozen=True, auxiliary_decoder_frozen=args.scope=='encoder', reset_every_shot=True,
        inter=False, alarm_tuning=False, selection='two declared starts, same fixed update rule',
        acceptance='TPR30 and J improve over matched static AND FPR<=.20',
        posthoc_caps_are_not_deployed_thresholds=True))
    emitted = run_arm(model, order=order, arm='intra', threshold=threshold,
                      output_dir=args.output/'stream', stride=32, data_api=provider,
                      predictor=partial(predict_shot,scope=args.scope))
    signals = [simple_data.attach_outcome(provider.read_shot('EAST', row['shot']), 'EAST', {}) for row in order]
    references, ref_seals = load_sealed(reference_dir, order, signals, threshold)
    predictions, seals = load_sealed(args.output/'stream', order, signals, threshold)
    filtered = []
    for row in predictions:
        valid = row['valid'].copy()
        if row['is_disrupt']:
            valid &= row['times'] <= row['down']
        filtered.append(dict(row, times=row['times'][valid], scores=row['scores'][valid]))
    if score_fixed_test('EAST', filtered, threshold) != emitted['metrics']:
        raise AssertionError('full TEST360 sealed recount differs')
    static, intra = diagnose(references, threshold), diagnose(predictions, threshold)
    old, new = static['metrics'], intra['metrics']
    trace_difference=0.
    for previous,current in zip(references,predictions):
        if previous['shot'] != current['shot'] or not np.array_equal(previous['times'],current['times']):
            raise AssertionError('paired complete trace identities differ')
        trace_difference=max(trace_difference,float(np.max(np.abs(previous['scores']-current['scores']))))
    if args.scope=='decoder':
        # Existing static traces used different inference batch sizes. Permit
        # float32 batching noise, not changed decisions or a risk-path update.
        if trace_difference>1e-6 or old!=new:
            raise AssertionError('decoder-only control unexpectedly changed risk decisions')
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
    if _hash(DEFAULT_PARENT) != PARENT_SHA256 or emitted['accepted_inter_updates']:
        raise AssertionError('original parent or inter policy changed')
    _durable_json(args.output/'verification.json', dict(reference_seals=ref_seals, intra_seals=seals,
                  full_test_recount=True, restored_after_every_shot=True,
                  max_score_difference_from_static=trace_difference,
                  decoder_only_risk_invariance_checked=args.scope=='decoder'))
    curves = {name: operating_curve(rows, threshold)['posthoc_best_at_cap']
              for name, rows in (('static', references), ('intra', predictions))}
    result = dict(status='COMPLETED', start=args.start, parameter_scope=args.scope,
        source=source, static=static, intra=intra,
        paired=paired_changes(static, intra), accepted_intra_updates=emitted['accepted_intra_updates'],
        accepted_inter_updates=0, original_parent_unchanged=True,
        meets_intra_goal=new['tpr'] > old['tpr'] and new['fpr'] <= .20 and new['J'] > old['J'],
        elapsed_seconds=time.monotonic()-started, independent_confirmation=False,
        posthoc_caps=curves, posthoc_warning='all test outcomes used after sealing; not online operating points')
    _durable_json(args.output/'completed.json', result)
    print(json.dumps(dict(stage='completed', start=args.start, static=old, intra=new,
                          meets_intra_goal=result['meets_intra_goal'])), flush=True)


if __name__ == '__main__':
    main()
