"""Matched optimizer-step pilot for a finite reference/emitted peak bag."""
import argparse
import json
from pathlib import Path
import zipfile

import torch

from .continual_decoder import RollingDecoder, head_digest
from .continual_decoder_experiment import ROOT, REFERENCE, BASELINE
from .ended_context_experiment import side_metrics
from .ended_decoder import DecoderFfnModel
from .ended_density_replay import EndedDensityReplay
from .peak_bag_decoder import BagRollingDecoder
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .redesign_20261004.run import run_arm
from .simple_online import _durable_json, _hash

PEAK_ARMS=('single_early_fixed','bag_early_fixed','bag_early_continual')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=PEAK_ARMS,required=True)
    parser.add_argument('--limit',type=int,default=120)
    args=parser.parse_args()
    full=json.loads((REFERENCE/'order.json').read_text())
    if args.output.exists() or not 30<args.limit<=len(full):
        parser.error('new output and prefix31..240 required')
    order=full[:args.limit]; args.output.mkdir(parents=True)
    torch.set_num_threads(4); torch.manual_seed(131)
    base,state=load_parent(DEFAULT_PARENT)
    model=DecoderFfnModel(base).to('cuda').eval()
    policy,mode=args.arm.split('_',1)
    learner=(RollingDecoder if policy=='single' else BagRollingDecoder)(model,mode=mode)
    provider=EndedDensityReplay(state['source_center'],state['source_scale'],
        output_dir=args.output/'stream',order=[r['shot'] for r in order],
        preserve_observed_channels=True)
    files=[Path(__file__),ROOT/'peak_bag_decoder.py',ROOT/'continual_decoder.py',
        ROOT/'ended_decoder.py',ROOT/'ended_context_rule.py',ROOT/'ended_context_experiment.py',
        ROOT/'ended_density_replay.py',ROOT/'alignment.py',ROOT/'persistent_ffn.py',
        ROOT/'fast_update.py',ROOT/'simple23.py',ROOT/'simple_data.py',ROOT/'simple_online.py',
        ROOT/'causal_input.py',ROOT/'channel_gradient_audit.py',ROOT/'ffn_intra_20261005/model.py',
        ROOT/'ffn_intra_20261005/loss.py',ROOT/'redesign_20261004/intra.py',
        ROOT/'redesign_20261004/run.py',ROOT/'redesign_20261004/prepare_prior.py']
    with zipfile.ZipFile(args.output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for file in files:archive.write(file,str(file.relative_to(ROOT)))
    threshold=float(state['threshold'])
    _durable_json(args.output/'order.json',order)
    _durable_json(args.output/'protocol.json',dict(
        scope='DEVELOPMENT finite bag pilot, not independent or mechanistic validation',
        arm=args.arm,mode=mode,peak_policy=policy,parent_sha256=PARENT_SHA256,
        threshold=threshold,order_sha256=_hash(REFERENCE/'order.json'),selected_shots=args.limit,
        online_steps=4,bootstrap_steps=40,lr=.0003,seed=131,replay_shots=32,
        actual_optimizer_steps_matched_to_single_schedule=True,flops_matched=False,
        numerical_input_mapping='same ended-first30 density mapping, preserve observed channels',
        mapping_approved_physical_calibration=False,input_coordinate_transition_index=30,
        transition_action='same reset auxiliary head/Adam and refit40 exact ended first30',
        natural_postseal_outcomes=True,risk_head_changed=False,source_changed=False,
        persistent_ffn_training=False,same_shot_delta_discarded=True,
        stride=100,horizon=25,step=.05,radius=1.,source_risk_threshold_fixed=True,
        alarm_tuning=False,outer='max adapted logit over selected peaks plus .20 signal maintenance',
        bag_selection='ended reference plus own emitted operationally eligible peak, deduplicated',
        bag_is_exhaustive_trajectory=False,code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'),
        previous_static_and_intra=str(BASELINE.resolve()),
        previous_schedule=str((ROOT/'runs/continual_decoder120_20261010').resolve())))

    def updater(m,shot,replay,**kwargs):
        return learner(m,shot,replay,provider=provider,**kwargs)

    result=run_arm(model,order=order,arm='both',threshold=threshold,
        output_dir=args.output/'stream',data_api=provider,updater=updater)
    if model.memory.any() or _hash(DEFAULT_PARENT)!=PARENT_SHA256:
        raise AssertionError('persistent/source state changed')
    with (args.output/'final_head.pt').open('xb') as handle:
        torch.save(dict(decoder=model.decoder.state_dict(),optimizer=learner.optimizer.state_dict(),
            optimizer_steps=learner.optimizer_steps,ended_ids=provider.ended_ids,
            parent_sha256=PARENT_SHA256,mode=mode,first_unused_future_index=len(order)+1),handle)
    periods={p:side_metrics(args.output/'stream',order,threshold,a,b)
             for p,a,b in (('all',0,args.limit),('first30',0,30),('after30',30,args.limit))}
    _durable_json(args.output/'comparison.json',periods)
    _durable_json(args.output/'completed.json',dict(source_sha256_unchanged=True,
        persistent_ffn_unchanged=True,head_sha256=head_digest(model),
        model_meta_optimizer_steps=learner.optimizer_steps,observed_shots=result['observed_count'],
        all_metric_seal_hashes_verified=True,temporary_retained=False))
    print(json.dumps(dict(completed=args.arm,metrics={p:{k:v for k,v in m.items() if k!='predictions'}
        for p,m in periods.items()})),flush=True)


if __name__=='__main__':main()
