"""Matched EAST replay with ended-density input control and internal updates.

Not V3C: no support110, expert weights, risk correction head or alarm tuning.
The startup shots are included. The four neural ablation arms share exactly
the same label-blind input statistics; mask_static isolates numeric mapping.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import torch

from . import simple_data
from .ended_density_replay import EndedDensityReplay
from .redesign_20261004.inter import postshot_update
from .redesign_20261004.run import (DEFAULT_ORDER, DEFAULT_ROSTER, DEFAULT_SOURCE,
    load_parent, load_roster, validate_order, validate_roster_rate_contract, run_arm)
from .redesign_20261004.source_replay import load_source_bank, evaluate_source_guard
from .simple_online import _durable_json, _hash


def update_in_coordinates(model, shot, replay, *, coordinates, data_api, **kwargs):
    """Past EAST replay follows the current numeric mapping; source stays raw."""
    if data_api is not coordinates:
        raise ValueError('runner and updater must share the same coordinate provider')
    if (not coordinates.ended_ids or shot['shot'] != coordinates.ended_ids[-1]
            or '_sealed_scores' not in shot):
        raise ValueError('current sealed and revealed shot required')
    ids = [s['shot'] for s in replay]
    if (len(ids) != len(set(ids)) or not set(ids) <= set(coordinates.ended_ids[:-1])
            or any('_sealed_scores' not in s for s in replay)):
        raise ValueError('only past sealed history may be replayed')
    return postshot_update(model, coordinates.reencode(shot),
        [coordinates.reencode(s) for s in replay], data_api=coordinates, **kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=120)
    parser.add_argument('--order', type=Path, default=DEFAULT_ORDER,
                        help='locked full order, including offline commissioning simulations')
    parser.add_argument('--roster', type=Path, default=DEFAULT_ROSTER,
                        help='explicit cohort/QC/ordering provenance matching --order')
    parser.add_argument('--internal-site',choices=('embedding','feature_affine','reversible_affine','window_gain','persistent_ffn','routed_ffn','routed_ffn_weight','gradient_transport'),default='embedding')
    parser.add_argument('--gradient-operator',type=Path,help='source TRAIN-only ridge rule; gradient_transport only')
    parser.add_argument('--router-artifact',type=Path,help='fixed J-TEXT TRAIN hidden keys; routed FFN sites only')
    parser.add_argument('--routing',choices=('local','uniform'),default='local',
                        help='source-keyed local update or same-parameter non-local mechanism control')
    parser.add_argument('--objective',choices=('level','forward_holdout'),default='level',
                        help='fixed chronological12/13 forecast split or archived all25 fit')
    parser.add_argument('--qa-convention',choices=('storage','source_ratio'),default='storage',
                        help='explicit diagnostic reconstruction from current Bt/Ip; not physical q calibration')
    parser.add_argument('--source-signal-weights',type=Path,
                        help='source-only fixed loss weights; only reversible_affine / level')
    parser.add_argument('--source-guard-policy',choices=('crossings','diagnostic'),default='crossings',
                        help='explicit source-veto ablation; source replay BCE always retained')
    parser.add_argument('--preserve-observed-channels',action='store_true',
                        help='density-only mapping retains genuine other-channel observations, even if prefix IQR=0')
    parser.add_argument('--normalization', choices=('density_robust','efd_zscore','efd_zscore_preserve_controls'),
                        default='density_robust', help='explicit numeric input intervention; original source buffers stay fixed')
    parser.add_argument('--statistics-mode', choices=('fixed','cumulative'), default='fixed',
                        help='EFD uses first30 ended shots or all past ended signals; never future signals')
    parser.add_argument('--source-moments', type=Path,
                        help='source TRAIN-only full-observation mean/std artifact, required for EFD')
    parser.add_argument('--arms', nargs='+', choices=('mask_static','static','intra','inter','both'),
                        default=['mask_static','static','intra','inter','both'])
    args=parser.parse_args()
    if args.output.exists() or args.limit < 0 or len(set(args.arms)) != len(args.arms):
        parser.error('new output, nonnegative limit and distinct arms required')
    is_efd=args.normalization!='density_robust'
    if is_efd != (args.source_moments is not None):
        parser.error('EFD requires explicit source moments; density_robust does not use them')
    if not is_efd and args.statistics_mode!='fixed':
        parser.error('cumulative statistics are scoped to EFD')
    if is_efd and args.preserve_observed_channels:
        parser.error('EFD always preserves observation masks; omit the density-only mask flag')
    if args.source_signal_weights is not None and (args.internal_site!='reversible_affine' or args.objective!='level'):
        parser.error('source signal weights require reversible_affine with level objective')
    if (args.internal_site=='gradient_transport') != (args.gradient_operator is not None):
        parser.error('gradient_transport requires an explicit gradient operator; other sites do not use one')
    if args.source_guard_policy=='diagnostic' and args.internal_site not in ('persistent_ffn','routed_ffn','routed_ffn_weight','gradient_transport'):
        parser.error('source-veto ablation is scoped to internal FFN candidates')
    if (args.internal_site in ('routed_ffn','routed_ffn_weight')) != (args.router_artifact is not None):
        parser.error('routed FFN requires an explicit source-only router; other sites do not use one')
    if args.routing!='local' and args.internal_site not in ('routed_ffn','routed_ffn_weight'):
        parser.error('routing control only applies to routed FFN')
    root=Path(__file__).parent
    bank_path=root/'runs/redesign_source_bank_20261004.pt'
    order=json.loads(args.order.read_text())
    ids, roster=load_roster(args.roster)
    validate_order(order, ids); validate_roster_rate_contract(roster,'nominal')
    selected=order[:args.limit] if args.limit else order
    if len(selected)<=30 or args.limit>len(order):
        parser.error('evaluation must include more than the 30 calibration shots')
    source_moments=None; source_moments_protocol=None
    if is_efd:
        from .efd_normalization import EfdReplay, load_source_statistics
        source_moments,source_moments_protocol=load_source_statistics(args.source_moments,
            expected_parent_sha256=_hash(DEFAULT_SOURCE),train_ids=simple_data.source_roles()['train'])
    torch.set_num_threads(4); torch.manual_seed(131)
    args.output.mkdir(parents=True)
    if is_efd:
        shutil.copyfile(args.source_moments,args.output/'source_moments.json')
    transport=None; transport_receipt=None
    if args.gradient_operator is not None:
        transport=torch.load(args.gradient_operator,map_location='cpu',weights_only=True)
        tp=transport['protocol']; train_ids=simple_data.source_roles()['train']
        if (tp['parent_sha256']!=_hash(DEFAULT_SOURCE) or tp['source_only'] is not True
                or tp['no_east_reads'] is not True or tp['source_classifier_and_decoder_frozen'] is not True
                or set(tp['source_train_ids'])!=set(train_ids)
                or set(tp['fit_ids']) & set(tp['selection_ids'])
                or set(tp['fit_ids']) | set(tp['selection_ids']) != set(train_ids)):
            raise ValueError('same-parent source TRAIN-only gradient operator required')
        shutil.copyfile(args.gradient_operator,args.output/'source_gradient_operator.pt')
        transport_receipt=dict(artifact_sha256=_hash(args.output/'source_gradient_operator.pt'),
            protocol=tp,used_for_persistent_gradient=False)
    router=None; router_receipt=None
    if args.router_artifact is not None:
        from .routed_ffn import load_router
        original_bank=load_source_bank(bank_path,expected_parent_sha256=_hash(DEFAULT_SOURCE))
        router=load_router(args.router_artifact,bank_sha256=original_bank['bank_sha256'])
        shutil.copyfile(args.router_artifact,args.output/'source_router.pt')
        router_receipt=dict(artifact_sha256=_hash(args.output/'source_router.pt'),
            source_role=router['source_role'],source_shot_ids=router['source_shot_ids'],
            hidden_vectors=router['hidden_vectors'],clusters=8,top_k=2,temperature=.1,
            routing=args.routing,target_records_accessed=0,label_used_for_key_fitting=False,
            source_bank_content_sha256=router['source_bank_content_sha256'])
        router_receipt['gate_normalization']='unit_l2' if args.internal_site=='routed_ffn_weight' else 'sum_one'
    qa_receipt=None
    signal_weights=None; signal_weight_protocol=None
    if args.source_signal_weights is not None:
        from .affine_signal_meta import load_weights
        signal_weights,signal_weight_protocol=load_weights(args.source_signal_weights,
            expected_parent_sha256=_hash(DEFAULT_SOURCE))
    if args.qa_convention=='source_ratio':
        from .qa_convention import fit_source_qa_convention
        qa_receipt=fit_source_qa_convention(
            simple_data.read_shot('J-TEXT',sid) for sid in simple_data.source_roles()['train'])
        _durable_json(args.output/'qa_source_convention.json',qa_receipt)
    _durable_json(args.output/'order.json', order)
    _durable_json(args.output/'roster.json', roster)
    files=[root/name for name in ('density_online.py','commissioning_stream.py','feature_affine.py','persistent_ffn.py','routed_ffn.py','routed_ffn_weight.py','gradient_transport.py','affine_signal_meta.py','qa_convention.py','ended_density_replay.py','efd_normalization.py','alignment.py',
        'causal_input.py','simple_data.py','simple23.py','simple_train.py','simple_online.py',
        'fast_update.py','io23.py','source_gate.py')]
    files.extend((root/'redesign_20261004').glob('*.py'))
    files.append(root/'ffn_intra_20261005/model.py')
    with zipfile.ZipFile(args.output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, str(path.relative_to(root)))
    results={}; initial=None
    for name in args.arms:
        model,state,gate,_=load_parent(DEFAULT_SOURCE,'cuda',persistent_site='embedding')
        if args.internal_site in ('feature_affine','reversible_affine','window_gain'):
            from .feature_affine import FeatureAffineModel
            if signal_weights is not None:
                from .affine_signal_meta import WeightedFeatureAffineModel
                model=WeightedFeatureAffineModel(model.base,signal_weights).to('cuda').eval()
            else:
                model=FeatureAffineModel(model.base,forecast_coordinates=(
                    'source' if args.internal_site=='feature_affine' else 'observed'),
                    temporary_mode='window_gain' if args.internal_site=='window_gain' else 'affine').to('cuda').eval()
        elif args.internal_site=='persistent_ffn':
            from .persistent_ffn import PersistentFfnModel
            model=PersistentFfnModel(model.base).to('cuda').eval()
        elif args.internal_site=='gradient_transport':
            from .gradient_transport import TransportFfnModel
            model=TransportFfnModel(model.base,transport['keys'],transport['coefficients']).to('cuda').eval()
        elif args.internal_site=='routed_ffn':
            from .routed_ffn import RoutedFfnModel
            model=RoutedFfnModel(model.base,router['router'],routing=args.routing).to('cuda').eval()
        elif args.internal_site=='routed_ffn_weight':
            from .routed_ffn_weight import RoutedWeightFfnModel
            model=RoutedWeightFfnModel(model.base,router['router'],routing=args.routing).to('cuda').eval()
        model.replay_policy='error_aware'; model.update_policy='error_only'; model.guard_policy='crossings'
        model.source_bank=load_source_bank(bank_path, expected_parent_sha256=gate['checkpoint_sha256'])
        model.source_replay_weight=1.; model.source_threshold=float(state['threshold'])
        model.source_rng=np.random.default_rng(132); model.source_backtracking=8
        model.source_guard_policy=args.source_guard_policy
        if initial is None:
            initial=dict(source_sha256=gate['checkpoint_sha256'],source_gate=gate,
                threshold=float(state['threshold']),model_config=state['model_config'],
                source_bank_sha256=_hash(bank_path),source_bank_content_sha256=model.source_bank['bank_sha256'],
                source_guard_policy=args.source_guard_policy,
                order_sha256=_hash(args.output/'order.json'),roster_sha256=_hash(args.output/'roster.json'),
                input_order_sha256=_hash(args.order),input_roster_sha256=_hash(args.roster),
                stream_selection_protocol=roster.get('selection_protocol',{}),
                code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'),
                code_sha256={str(p.relative_to(root)):_hash(p) for p in files},
                calibration_count=30,calibration_rule='first30 ended signals, no outcome selection',
                normalization=('density-only source-coordinate mapping; preserve original observation masks'
                    if args.preserve_observed_channels else
                    'density-only source-coordinate mapping; same unusable mask after30'),
                preserve_observed_channels=args.preserve_observed_channels,
                normalization_candidate=args.normalization,statistics_mode=args.statistics_mode,
                source_moments=(None if not is_efd else dict(
                    artifact_sha256=_hash(args.output/'source_moments.json'),protocol=source_moments_protocol)),
                first_shot_target_information=0,source_buffers_unchanged=True,
                future_labels_used=False,source_training=False,jtext_online=False,support110=False,
                natural_outcomes=True,paid_query_budget=False,alarm_tuning=False,
                qa_convention=args.qa_convention,qa_source_receipt=qa_receipt,
                source_router=router_receipt,
                source_gradient_transport=transport_receipt,
                source_signal_weights=(None if signal_weights is None else dict(
                    artifact_sha256=_hash(args.source_signal_weights),protocol=signal_weight_protocol,
                    weights=signal_weights.tolist(),used_for_persistent_loss=False)),
                risk_correction_head=False,forecast_prior='original qualified source head',
                intra=dict(site='source tokens before Transformer',count=128,horizon=25,
                           stride=100,step=.05,radius=1.,target='matured real signal, not autoregression',
                           objective=args.objective,
                           acceptance=('all25 in-sample decrease' if args.objective=='level' else
                                       'first12 fit; last13 accept/reject; both loss decreases')),
                inter=dict(site='original input embedding',count=23616,lr=.03,radius=1.,steps=4,
                           source_replay_weight=1.,guard='sampled history/source, not a guarantee'),
                selected_shots=len(selected),roster_shots=len(order),arms=args.arms,
                scope='DEVELOPMENT MATCHED CAUSAL REPLAY; startup costs included; not independent validation')
            if is_efd:
                initial['normalization']='feature-wise device mean/std bridge to original source coordinates; masks preserved'
                initial['preserve_observed_channels']=True
                initial['calibration_rule']=('first30 ended signals, fixed afterwards' if args.statistics_mode=='fixed'
                    else 'start after30; all past ended signals accumulated between shots')
            initial['internal_site']=args.internal_site
            initial['signal_forecast_coordinates']=getattr(model,'forecast_coordinates','source')
            initial['temporary_mode']=getattr(model,'temporary_mode','tokens')
            initial['intra'].update(site=getattr(model,'adaptation_site','source tokens before Transformer'),
                                    count=getattr(model,'temporary_dim',128))
            initial['inter'].update(site=(model.adaptation_site if args.internal_site in ('persistent_ffn','routed_ffn','routed_ffn_weight','gradient_transport') else
                                         'normalized flat23 before embedding' if args.internal_site!='embedding'
                                         else 'original input embedding'),
                                    count=sum(v.numel() for v in model.persistent_parameters()))
            if args.internal_site in ('persistent_ffn','routed_ffn','routed_ffn_weight','gradient_transport'):
                initial['intra']['state_receipt']='count/dtype/L2/SHA256; temporary state not retained'
                initial['inter']['memory_artifact']='final_'+args.internal_site+'.pt'
            _durable_json(args.output/'protocol.json', initial)
        source_guard=evaluate_source_guard(model,model.source_bank,'cuda',threshold=float(state['threshold']))
        if not source_guard['passed']:
            raise ValueError('original source replay guard fails before stream')
        provider_args=dict(output_dir=args.output/name,order=[r['shot'] for r in selected],
            aligned=name!='mask_static',qa_coefficient=None if qa_receipt is None else qa_receipt['coefficient'])
        provider=(EfdReplay(state['source_center'],state['source_scale'],
            source_statistics=source_moments,statistics_mode=args.statistics_mode,
            preserve_controls=args.normalization=='efd_zscore_preserve_controls',**provider_args)
            if is_efd else EndedDensityReplay(state['source_center'],state['source_scale'],
                preserve_observed_channels=args.preserve_observed_channels,**provider_args))
        updater=partial(update_in_coordinates,coordinates=provider)
        results[name]=run_arm(model,order=order,arm='static' if name=='mask_static' else name,
            threshold=float(state['threshold']),output_dir=args.output/name,limit=args.limit,
            data_api=provider,updater=updater,device='cuda',stride=100,step=.05,radius=1.,
            objective=args.objective,inter_lr=.03,inter_radius=1.,inter_steps=4)
        _durable_json(args.output/name/'alignment.json',
            {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in provider.statistics.items()})
        if not all(torch.equal(v.detach().cpu(),state['model'][k]) for k,v in model.base.state_dict().items()):
            raise ValueError('original source model or buffers changed')
        if args.internal_site in ('persistent_ffn','routed_ffn','routed_ffn_weight','gradient_transport'):
            artifact=args.output/name/('final_'+args.internal_site+'.pt')
            with artifact.open('xb') as handle:
                torch.save(dict(memory=model.memory.detach().cpu(),
                    source_sha256=gate['checkpoint_sha256'],site=model.adaptation_site,
                    source_guard=evaluate_source_guard(model,model.source_bank,'cuda',
                        threshold=float(state['threshold'])),
                    observed_shots=[r['shot'] for r in selected],temporary_retained=False),handle)
        _durable_json(args.output/'comparison.json', results)
        print(json.dumps(dict(completed_arm=name,
            metrics={k:v for k,v in results[name]['metrics'].items() if k!='predictions'})),flush=True)
    if _hash(DEFAULT_SOURCE)!=initial['source_sha256'] or _hash(bank_path)!=initial['source_bank_sha256']:
        raise ValueError('source artifact changed')


if __name__=='__main__':
    main()
