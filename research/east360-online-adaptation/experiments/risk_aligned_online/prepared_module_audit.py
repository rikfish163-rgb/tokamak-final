"""Single-module replacement controls after source preparation failed on EAST.

These are retrospective structural diagnostics, not online adaptation or a
search for a replacement source model. All four controls use full TEST220 and
TEST360, fixed source threshold and no target optimization. Nonlinear module
interactions mean their effects must not be added to explain the full model.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import torch

from . import simple_data
from .aligned_encoder import predict_shot
from .aligned_encoder_experiment import source_test
from .cross_channel_stable import FourViewModel,HEAD
from .fixed_test_error_audit import diagnose,paired_changes
from .fixed_test_protocol import load_test_contract,score_fixed_test
from .operating_curve_audit import load_sealed,operating_curve
from .redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_parent
from .redesign_20261004.run import run_arm
from .simple_online import _durable_json,_hash

GROUPS=dict(embedding='base.embedding.',block0='base.encoder.layers.0.',
            block1='base.encoder.layers.1.',norm='base.norm.')


def hybrid_state(original,prepared,group):
    if group not in GROUPS:raise ValueError('one explicit module control required')
    if set(original)!=set(prepared):raise ValueError('candidate state keys differ')
    for name,old in original.items():
        new=prepared[name]
        if old.shape!=new.shape or old.dtype!=new.dtype:
            raise ValueError('candidate shape/dtype differs')
        if not torch.isfinite(old).all() or not torch.isfinite(new).all():
            raise ValueError('finite candidate tensors required')
        if not name.startswith((*GROUPS.values(),'decoder.')) and not torch.equal(old,new):
            raise ValueError('undeclared preparation change')
    prefix=GROUPS[group]
    if not any(name.startswith(prefix) and not torch.equal(old,prepared[name]) for name,old in original.items()):
        raise ValueError('requested module is unchanged')
    return {name:(prepared[name] if name.startswith(prefix) else old).clone()
            for name,old in original.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--parent-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('new output directory required')
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    completion=json.loads((args.candidate/'completed.json').read_text())
    if completion['status']!='COMPLETED' or completion['meets_intra_goal']:
        raise ValueError('this diagnostic is scoped to the completed failed candidate')
    gate=json.loads((args.candidate/'source_gate.json').read_text())
    if _hash(args.candidate/'prepared.pt')!=gate['candidate_sha256']:
        raise ValueError('prepared candidate changed')
    prepared=torch.load(args.candidate/'prepared.pt',map_location='cpu',weights_only=True)
    base,parent=load_parent(DEFAULT_PARENT)
    if prepared['parent_sha256']!=PARENT_SHA256:raise ValueError('different original parent')
    model=FourViewModel(base,site='embedding').cuda().eval().requires_grad_(False)
    head=torch.load(HEAD,map_location='cpu',weights_only=True)
    if head['parent_sha256']!=PARENT_SHA256:raise ValueError('different diagnostic head parent')
    model.decoder.load_state_dict(head['decoder'])
    original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    states={name:hybrid_state(original,prepared['model'],name) for name in GROUPS}
    torch.set_num_threads(4);torch.manual_seed(131)
    threshold=float(parent['threshold']);contract=load_test_contract('EAST')
    order=[dict(shot=sid) for sid in contract['shot_ids']]
    provider=SimpleNamespace(read_shot=partial(simple_data.read_shot,rate_contract='nominal'),
        attach_outcome=simple_data.attach_outcome,window=simple_data.window)
    # Complete reference predictions already exist. Outcome access is only for
    # describing diagnostics; the static predictor never receives those fields.
    signals=[simple_data.attach_outcome(provider.read_shot('EAST',r['shot']),'EAST',{}) for r in order]
    parent_rows,parent_seals=load_sealed(args.parent_run/'east_test360/stream',order,signals,threshold)
    reference=diagnose(parent_rows,threshold)
    args.output.mkdir(parents=True);started=time.monotonic();root=Path(__file__).parent
    with zipfile.ZipFile(args.output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in ('prepared_module_audit.py','aligned_encoder.py','aligned_encoder_experiment.py',
            'fixed_test_error_audit.py','fixed_test_protocol.py','cross_channel_prefix.py',
            'cross_channel_stable.py','cross_channel_shot_screen.py','simple23.py','simple_data.py',
            'io23.py','causal_input.py','simple_train.py','simple_online.py','operating_curve_audit.py',
            'redesign_20261004/prepare_prior.py','redesign_20261004/intra.py','redesign_20261004/run.py'):
            archive.write(root/name,name)
    _durable_json(args.output/'protocol.json',dict(
        scope='RETROSPECTIVE MODULE CONTROLS; NOT ONLINE ADAPTATION OR INDEPENDENT CONFIRMATION',
        parent_sha256=PARENT_SHA256,candidate_sha256=gate['candidate_sha256'],
        code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'),threshold=threshold,
        groups=GROUPS,jtext=load_test_contract('J-TEXT'),east=contract,
        source_training_steps=0,target_training_steps=0,intra=False,inter=False,
        comparison='replace ONE prepared module in the original source model; all other tensors original',
        purpose='locate harmful source-preparation changes; no sum-of-module-effects claim',
        source_gate_filters_east_diagnostic=False,posthoc_caps_are_not_online_thresholds=True))
    results={};seals={}
    for group,state in states.items():
        folder=args.output/group;folder.mkdir();model.load_state_dict(state)
        print(json.dumps(dict(stage='module_control',group=group)),flush=True)
        source=source_test(model,folder/'jtext_test220',threshold)
        emitted=run_arm(model,order=order,arm='static',threshold=threshold,output_dir=folder/'stream',
            stride=32,data_api=provider,predictor=predict_shot)
        rows,seals[group]=load_sealed(folder/'stream',order,signals,threshold)
        filtered=[]
        for row in rows:
            valid=row['valid'].copy()
            if row['is_disrupt']:valid &= row['times']<=row['down']
            filtered.append(dict(row,times=row['times'][valid],scores=row['scores'][valid]))
        exact=score_fixed_test('EAST',filtered,threshold)
        if exact!=emitted['metrics']:raise AssertionError('full test recount differs')
        analysis=diagnose(rows,threshold);pair=paired_changes(reference,analysis)
        curve=operating_curve(rows,threshold)
        for k,v in model.state_dict().items():torch.testing.assert_close(v.cpu(),state[k],rtol=0,atol=0)
        if emitted['accepted_intra_updates'] or emitted['accepted_inter_updates']:
            raise AssertionError('diagnostic must not update parameters')
        results[group]=dict(source={k:v for k,v in source.items() if k!='predictions'},
            east=analysis['metrics'],paired={k:v for k,v in pair.items() if k!='changed_shots'},
            early_fp=analysis['fp_before_elapsed_ms'],posthoc_caps=curve['posthoc_best_at_cap'])
        _durable_json(folder/'analysis.json',dict(diagnostic=analysis,paired=pair,
            operating_diagnostic={k:v for k,v in curve.items() if k!='curve'},parameters_unchanged=True))
        _durable_json(args.output/'progress.json',results)
        print(json.dumps(dict(stage='module_done',group=group,results=results[group])),flush=True)
    if _hash(DEFAULT_PARENT)!=PARENT_SHA256:raise AssertionError('original source changed')
    _durable_json(args.output/'completed.json',dict(status='COMPLETED',results=results,
        elapsed_seconds=time.monotonic()-started,full_test_curves=len(GROUPS)*(220+360),
        reference_seals=parent_seals,seals=seals,source_file_unchanged=True,
        independent_confirmation=False,no_neural_adaptation_gain_claim=True))
    print(json.dumps(dict(stage='completed',results=results)),flush=True)


if __name__=='__main__':main()
