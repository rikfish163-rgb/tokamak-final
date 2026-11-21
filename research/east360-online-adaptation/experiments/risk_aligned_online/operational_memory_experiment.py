"""Locked-prefix control: static, intra rule, persistent FFN and combination.

All arms learn the same auxiliary head on the first30 ended shots. The density
transition has an explicit robust or full-point mean/std numerical contract.
Neither is certified physical calibration. Post30 only the new
registered FFN memory may learn. The control is inter-trained intra, not purely
label-free source-only intra. No target outcome is read before its durable seal.
"""
import argparse
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import torch

from .continual_decoder import RollingDecoder, head_digest
from .ended_context_experiment import REFERENCE, side_metrics
from .ended_decoder import DecoderFfnModel,SourceTokenDecoderModel,SIGNAL_CONTRACTS
from .ended_density_replay import EndedDensityReplay
from .efd_normalization import EfdReplay, load_source_statistics
from .operational_memory import OperationalMemory
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .redesign_20261004.run import run_arm
from .redesign_20261004.source_replay import load_source_bank
from .simple_online import _durable_json, _hash
from . import simple_data

ROOT=Path(__file__).parent
BANK=ROOT/'runs/redesign_source_bank_20261004.pt'
ARMS=('mapped_static','intra_control','inter_only','both')
NORMALIZATIONS=('density_robust','efd_density_zscore')
TEMPORARY_SITES=('final_ffn','source_tokens')


def arm_flags(arm):
    if arm not in ARMS:raise ValueError('explicit experiment arm required')
    return arm in ('intra_control','both'),arm in ('inter_only','both')


def validate_normalization(normalization,source_moments):
    if normalization not in NORMALIZATIONS or ((normalization=='efd_density_zscore')!=(source_moments is not None)):
        raise ValueError('EFD density requires source TRAIN moments; robust density does not use them')


def validate_temporary_site(site,arm):
    if site not in TEMPORARY_SITES or arm not in ARMS or (site=='source_tokens' and arm in ('inter_only','both')):
        raise ValueError('source-token pilot is static/intra only; persistent FFN is a separate declared route')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--limit',type=int,default=120)
    parser.add_argument('--signal-contract',choices=SIGNAL_CONTRACTS,default='all23')
    parser.add_argument('--normalization',choices=NORMALIZATIONS,default='density_robust')
    parser.add_argument('--source-moments',type=Path)
    parser.add_argument('--temporary-site',choices=TEMPORARY_SITES,default='final_ffn')
    args=parser.parse_args()
    try:
        validate_normalization(args.normalization,args.source_moments)
        validate_temporary_site(args.temporary_site,args.arm)
    except ValueError as error:parser.error(str(error))
    order=json.loads((REFERENCE/'order.json').read_text())
    if args.output.exists() or not 30<args.limit<=len(order):
        parser.error('new output and a declared31..240 locked prefix required')
    args.output.mkdir(parents=True)
    order=order[:args.limit]
    torch.set_num_threads(4);torch.manual_seed(131)
    base,state=load_parent(DEFAULT_PARENT)
    source_moments=None
    if args.normalization=='efd_density_zscore':
        source_moments,_=load_source_statistics(args.source_moments,
            expected_parent_sha256=PARENT_SHA256,train_ids=simple_data.source_roles()['train'])
        shutil.copyfile(args.source_moments,args.output/'source_moments.json')
    original={k:v.detach().clone() for k,v in base.state_dict().items()}
    threshold=float(state['threshold'])
    factory=DecoderFfnModel if args.temporary_site=='final_ffn' else SourceTokenDecoderModel
    model=factory(base,signal_contract=args.signal_contract).to('cuda').eval()
    head=RollingDecoder(model,mode='early_fixed')
    intra,persistent=arm_flags(args.arm)
    bank=load_source_bank(BANK)
    learner=OperationalMemory(model,bank=bank,intra=intra) if persistent else None
    provider_args=dict(output_dir=args.output/'stream',order=[r['shot'] for r in order])
    provider=(EndedDensityReplay(state['source_center'],state['source_scale'],
        preserve_observed_channels=True,**provider_args) if source_moments is None else
        EfdReplay(state['source_center'],state['source_scale'],source_statistics=source_moments,
                  channels=('ne_nG',),statistics_mode='fixed',**provider_args))
    _durable_json(args.output/'order.json',order)
    files=[Path(__file__),ROOT/'operational_memory.py',ROOT/'continual_decoder.py',
        ROOT/'ended_decoder.py',ROOT/'ended_context_rule.py',ROOT/'ended_context_experiment.py',
        ROOT/'ended_density_replay.py',ROOT/'efd_normalization.py',ROOT/'alignment.py',ROOT/'persistent_ffn.py',
        ROOT/'fast_update.py',ROOT/'simple23.py',ROOT/'simple_data.py',ROOT/'simple_online.py',
        ROOT/'causal_input.py',ROOT/'channel_gradient_audit.py',ROOT/'ffn_intra_20261005/model.py',
        ROOT/'ffn_intra_20261005/loss.py',ROOT/'redesign_20261004/intra.py',
        ROOT/'redesign_20261004/run.py',ROOT/'redesign_20261004/source_replay.py',
        ROOT/'redesign_20261004/prepare_prior.py']
    with zipfile.ZipFile(args.output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for path in files:archive.write(path,str(path.relative_to(ROOT)))
    _durable_json(args.output/'protocol.json',dict(
        scope='SINGLE PARENT DEVELOPMENT CAUSAL PILOT, not independent future validation',
        arm=args.arm,parent_sha256=PARENT_SHA256,threshold=threshold,
        order_sha256=_hash(REFERENCE/'order.json'),selected_shots=args.limit,seed=131,
        source_bank_file_sha256=_hash(BANK),source_bank_content_sha256=bank['bank_sha256'],
        input_mapping=('same ended-first30 density mapping; preserve observed channels' if source_moments is None else
                       'ended-first30 full-point mean/std density-only bridge; other22 and observed masks unchanged'),
        normalization=args.normalization,normalization_channels=['ne_nG'],
        source_moments_file_sha256=None if source_moments is None else _hash(args.output/'source_moments.json'),
        mapping_approved_physical_calibration=False,first30_steps=4,bootstrap_steps=40,
        coordinate_transition_index=30,aux_head_reset_at_transition=True,
        auxiliary_head_frozen_after30=True,source_retrained=False,risk_head_changed=False,
        auxiliary_postseal_feedback_in_all_arms=True,
        signal_contract=args.signal_contract,input_feature_count=23,
        auxiliary_excluded_features=[] if args.signal_contract=='all23' else ['mode_number_n'],
        auxiliary_target_restriction_is_not_input_deletion=True,
        mode_number_n_is_assumed_categorical=False,
        temporary_site=args.temporary_site,temporary_count=model.temporary_dim,
        temporary_is_offset_from_parent_origin=True,
        persistent_site=('original final Transformer FFN linear1/linear2, additive16576'
                         if args.temporary_site=='final_ffn' else 'not_present'),
        persistent_enabled=learner is not None,intra=intra,temporary_discarded=True,
        persistent_starts_after_ended_index=31 if persistent else None,
        persistent_proposals_per_ended_shot=1 if persistent else 0,
        persistent_step=.03,persistent_radius=1.,source_bce_weight=.20,
        persistent_task='balanced current+opposite ended shot full-trajectory operational maximum',
        gradient_unroll_depth=4,gradient_is_full_untruncated=False,
        history_guard='actual full causal curves of recent32 ended shots: no lost TP or TN',
        source_guard='all128 original TRAIN bank points; not whole J-TEXT retention',
        acceptance='paired full-curve max loss improves AND history/source guards pass',
        future_retention_guarantee=False,stride=100,horizon=25,intra_step=.05,intra_radius=1.,
        lead_ms_strict=30.,alarm_tuning=False,natural_outcomes_after_seal=True,
        first30_startup_errors_included=True,compute_matched_comparison=False,
        code_snapshot_sha256=_hash(args.output/'code_snapshot.zip')))

    def updater(m,shot,replay,**kwargs):
        head_receipt=head(m,shot,replay,provider=provider,**kwargs)
        memory_receipt=(dict(accepted=False,actual_steps=0,reason=f'{args.arm}_no_memory')
            if learner is None else learner(m,shot,replay,provider=provider,**kwargs))
        return dict(accepted=head_receipt['accepted'] or memory_receipt['accepted'],
                    head=head_receipt,memory=memory_receipt)

    deployed='both' if persistent and intra else 'inter' if persistent else 'intra' if intra else 'static'
    result=run_arm(model,order=order,arm=deployed,threshold=threshold,
                   output_dir=args.output/'stream',data_api=provider,updater=updater,
                   postseal_feedback=True)
    _durable_json(args.output/'normalization.json',
        {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in provider.statistics.items()})
    for k,v in model.base.state_dict().items():
        torch.testing.assert_close(v.detach().cpu(),original[k],rtol=0,atol=0)
    if _hash(DEFAULT_PARENT)!=PARENT_SHA256:raise AssertionError('original source file changed')
    if learner is None:model.assert_persistent_inactive()
    periods={p:side_metrics(args.output/'stream',order,threshold,a,b)
             for p,a,b in (('all',0,args.limit),('first30',0,30),('after30',30,args.limit))}
    _durable_json(args.output/'comparison.json',periods)
    memory_rows=[json.loads(p.read_text())['persistent']['memory']
                 for p in sorted((args.output/'stream').glob('*_audit.json'))]
    accepted=sum(r['accepted'] for r in memory_rows)
    with (args.output/'final_state.pt').open('xb') as handle:
        torch.save(dict(memory=model.memory.detach().cpu() if args.temporary_site=='final_ffn' else None,
            decoder=model.decoder.state_dict(),
            head_optimizer=None if head.optimizer is None else head.optimizer.state_dict(),
            head_optimizer_steps=head.optimizer_steps,parent_sha256=PARENT_SHA256,
            ended_ids=provider.ended_ids),handle)
    _durable_json(args.output/'completed.json',dict(source_file_and_tensors_unchanged=True,
        head_sha256=head_digest(model),head_optimizer_steps=head.optimizer_steps,
        observed_shots=result['observed_count'],memory_updates=accepted,
        memory_proposals=sum(r['actual_steps'] for r in memory_rows),
        memory_norm=float(model.memory.detach().double().norm()) if args.temporary_site=='final_ffn' else 0.,
        all_metric_seal_hashes_verified=True,temporary_retained=False))
    print(json.dumps(dict(completed=args.arm,memory_updates=accepted,
        metrics={p:{k:v for k,v in m.items() if k!='predictions'} for p,m in periods.items()})),flush=True)


if __name__=='__main__':main()
