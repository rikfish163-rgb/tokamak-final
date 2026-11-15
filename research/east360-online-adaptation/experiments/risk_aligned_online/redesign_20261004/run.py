"""Same-parent four-arm EAST replay, independent of the legacy V3C pipeline.

Only the source model and numerical/data helpers are reused. Prediction,
temporary gradients, persistent gradients and objectives are redesigned.
Outcome reveal is natural at shot end, NOT an active label-query budget.
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..simple23 import Config, Simple23
from ..simple_online import _durable_json, _hash, seal_predictions
from ..simple_train import metrics
from ..source_gate import qualify_source

ARMS = ('static', 'intra', 'inter', 'both')
DEFAULT_SOURCE = Path(__file__).parents[1]/'runs/gradient_aligned512_20260929/source.pt'
DEFAULT_STREAM = Path(__file__).parents[1]/'runs/east_all_nominal_stream_20261004'
DEFAULT_ORDER = DEFAULT_STREAM/'order.json'
DEFAULT_ROSTER = DEFAULT_STREAM/'roster.json'


def load_roster(path):
    """An explicit label-blind selection, not an inferred balanced test set."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError('explicit roster must be an object')
    ids = payload.get('shot_ids')
    if (not isinstance(ids, list) or not ids
            or any(type(sid) is not int or sid <= 0 for sid in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('explicit roster requires distinct positive integer shot IDs')
    if not isinstance(payload.get('selection'), str) or not payload['selection'].strip():
        raise ValueError('explicit roster requires selection provenance')
    return ids, payload


def validate_order(order, ids):
    actual = [int(row['shot']) for row in order]
    if len(set(actual)) != len(actual):
        raise ValueError('duplicate shot in locked order')
    if len(actual) != len(ids) or set(actual) != set(ids):
        raise ValueError('locked order does not match full roster')


def validate_roster_rate_contract(provenance, contract):
    expected=io23.rate_tolerance_hz('EAST',contract)
    selection=(provenance or {}).get('selection_protocol',{})
    if not isinstance(selection,dict):
        raise ValueError('invalid roster selection protocol')
    # Older strict manifests record the numeric tolerance without a name.
    # Check that field too; never silently run an archived strict list nominally.
    if (('rate_contract' in selection and selection['rate_contract'] != contract)
            or ('rate_atol_hz' in selection and selection['rate_atol_hz'] != expected)):
        raise ValueError('roster and target reader rate contracts differ')


def run_arm(model, *, order, arm, threshold, output_dir, limit=0, device='cuda',
            stride=100, step=.05, radius=1., objective='level', inter_lr=.03,
            inter_radius=1., inter_steps=4, seed=131, data_api=simple_data,
            predictor=None, updater=None, postseal_feedback=False):
    if arm not in ARMS:
        raise ValueError('unknown arm')
    if postseal_feedback and updater is None:
        raise ValueError('postseal auxiliary feedback requires an explicit updater')
    if predictor is None:
        from .intra import predict_shot
        predictor = predict_shot
    if updater is None:
        from .inter import postshot_update
        updater = postshot_update
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    selected = order[:limit] if limit else order
    replay, rng = deque(maxlen=32), np.random.default_rng(seed)
    rows, audits = [], []
    start_time = time.monotonic()
    for index, item in enumerate(selected, 1):
        # Data read contains diagnostic signals only; no HDF5 outcome fields.
        shot = apply_input_contract(data_api.read_shot('EAST', int(item['shot'])))
        scores, audit = predictor(model, shot, intra=arm in ('intra', 'both'),
            device=device, stride=stride, step=step, radius=radius,
            objective=objective, data_api=data_api)
        scores = np.asarray(scores, np.float32)
        if (scores.shape != shot['times'].shape or not np.isfinite(scores).all()
                or np.any((scores < 0)|(scores > 1))):
            raise ValueError('invalid emitted probability trace')
        receipt = seal_predictions(output_dir, index, shot['shot'], shot['times'], scores,
                                   arm=arm, threshold=threshold)
        # Everything above executes before the first access to this outcome.
        revealed = data_api.attach_outcome(shot, 'EAST', {})
        if float(revealed['horizon']) != 125.:
            raise ValueError('EAST training label contract must be fixed 125ms')
        revealed['_sealed_scores'] = scores.copy()
        valid = np.asarray(revealed['valid'], bool).copy()
        if revealed['is_disrupt']:
            valid &= shot['times'] <= revealed['down']
        rows.append(dict(shot=shot['shot'], is_disrupt=revealed['is_disrupt'],
            down=revealed['down'], times=shot['times'][valid], scores=scores[valid]))
        current = metrics(rows, threshold, lead=30.)
        persistent = (updater(model, revealed, replay, device=device, rng=rng,
            threshold=threshold, lr=inter_lr, radius=inter_radius, steps=inter_steps,
            data_api=data_api) if arm in ('inter', 'both') or postseal_feedback else None)
        # Current experience enters replay only after its own postshot update.
        replay.append(revealed)
        audit = dict(audit, shot=shot['shot'], seal_sha256=receipt['sha256'],
            prediction=current['predictions'][-1], persistent=persistent,
            label_access='natural reveal AFTER whole-shot durable seal')
        audits.append(audit)
        _durable_json(output_dir/f'{index:04d}_{shot["shot"]}_audit.json', audit)
        if index % 5 == 0 or index == len(selected):
            progress = dict(arm=arm, observed_count=index, elapsed_seconds=time.monotonic()-start_time,
                tp=current['tp'], fp=current['fp'], n_disrupt=current['n_disrupt'], n_safe=current['n_safe'],
                accepted_intra_updates=sum(a.get('accepted_updates', 0) for a in audits),
                accepted_inter_updates=sum(bool(a['persistent'] and a['persistent']['accepted']) for a in audits))
            _durable_json(output_dir/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    result = dict(arm=arm, metrics=metrics(rows, threshold, lead=30.), observed_count=len(selected),
        roster_count=len(order), accepted_intra_updates=sum(a.get('accepted_updates', 0) for a in audits),
        accepted_inter_updates=sum(bool(a['persistent'] and a['persistent']['accepted']) for a in audits),
        elapsed_seconds=time.monotonic()-start_time,
        evidence=('DEVELOPMENT_FULL360' if len(selected) == len(order) == 360 else
                  'DEVELOPMENT_LOCKED_STREAM' if len(selected) == len(order) else
                  'DEVELOPMENT_PREFIX'))
    _durable_json(output_dir/'summary.json', result)
    return result


def load_parent(checkpoint, device, *, context_dim=8, context_mode='endpoint',
                persistent_site='tokens', prior=None):
    if persistent_site not in ('tokens','embedding'):
        raise ValueError('unknown internal persistent site')
    gate = qualify_source(checkpoint)
    if not gate['qualified']:
        raise ValueError('source classifier did not qualify')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not state.get('forecast_trained'):
        raise ValueError('source direct forecast head required')
    base = Simple23(Config(**state['model_config']), state['source_center'], state['source_scale'])
    base.load_state_dict(state['model'], strict=True)
    prior_protocol = None
    if prior is not None:
        prepared = torch.load(prior, map_location='cpu', weights_only=False)
        prior_protocol = prepared['protocol']
        if (prior_protocol['parent_sha256'] != gate['checkpoint_sha256']
                or not prior_protocol['source_only']
                or not prior_protocol['static_classifier_bitwise_unchanged']):
            raise ValueError('forecast prior violates source-only parent contract')
        base.forecast_head.load_state_dict(prepared['forecast_head'], strict=True)
    if persistent_site == 'tokens':
        from .model import ContextModel
        wrapper = ContextModel(base, context_dim=context_dim,context_mode=context_mode)
    else:
        from .embedding_model import EmbeddingModel
        wrapper = EmbeddingModel(base)
    wrapper = wrapper.to(device).eval()
    return wrapper, state, gate, prior_protocol


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--order', type=Path, default=DEFAULT_ORDER)
    p.add_argument('--roster', type=Path, default=DEFAULT_ROSTER,
                   help='explicit roster.json with shot_ids and label-blind selection provenance')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--rate-contract',choices=('strict','nominal'),default='nominal',
                   help='explicit target-only sampling contract; keep declared times/rates')
    p.add_argument('--seed', type=int, default=131)
    p.add_argument('--context-dim', type=int, default=8)
    p.add_argument('--context-mode', choices=('endpoint','ordered'), default='endpoint')
    p.add_argument('--persistent-site', choices=('tokens','embedding'), default='tokens',
                   help='internal parameter scope; embedding replaces the context-to-token map')
    p.add_argument('--replay-policy', choices=('uniform','error_aware'), default='uniform')
    p.add_argument('--update-policy', choices=('all','error_only'), default='all')
    p.add_argument('--guard-policy', choices=('margin','crossings'), default='margin')
    p.add_argument('--stride', type=int, default=100)
    p.add_argument('--step', type=float, default=.05)
    p.add_argument('--radius', type=float, default=1.)
    p.add_argument('--objective', choices=('level','innovation','shape','forward_holdout'), default='level')
    p.add_argument('--inter-lr', type=float, default=.03)
    p.add_argument('--inter-radius', type=float, default=1.)
    p.add_argument('--inter-steps', type=int, default=4)
    p.add_argument('--prior', type=Path)
    p.add_argument('--source-bank', type=Path,
                   help='optional fixed J-TEXT TRAIN-only replay; no target support initialization')
    p.add_argument('--source-replay-weight', type=float, default=1.)
    p.add_argument('--source-backtracking', type=int, default=0,
                   help='0..12 step halvings until revealed-history/source guards pass')
    args = p.parse_args(argv)
    if args.objective == 'shape' and (args.stride != 100 or args.step != .05 or args.radius != 1.):
        p.error('shape hypothesis requires stride100, step.05 and radius1')
    if (args.limit < 0 or len(set(args.arms)) != len(args.arms)
            or args.stride <= 25 or min(args.step,args.radius,args.inter_lr,args.inter_radius) <= 0
            or args.inter_steps < 1 or args.context_dim < 1
            or not np.isfinite(args.source_replay_weight) or args.source_replay_weight <= 0
            or not 0 <= args.source_backtracking <= 12
            or args.source_backtracking and args.source_bank is None):
        p.error('invalid limit, duplicate arms, adaptation budget or forecast stride')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    model, state, gate, prior_protocol = load_parent(args.checkpoint,args.device,
        context_dim=args.context_dim,context_mode=args.context_mode,
        persistent_site=args.persistent_site,prior=args.prior)
    model.replay_policy = args.replay_policy
    model.update_policy = args.update_policy
    model.guard_policy = args.guard_policy
    source_bank_sha256 = None
    if args.source_bank is not None:
        from .source_replay import load_source_bank
        model.source_bank = load_source_bank(args.source_bank,
                                            expected_parent_sha256=gate['checkpoint_sha256'])
        model.source_replay_weight = args.source_replay_weight
        model.source_threshold = float(state['threshold'])
        model.source_rng = np.random.default_rng(args.seed + 1)
        model.source_backtracking = args.source_backtracking
        source_bank_sha256 = _hash(args.source_bank)
    order = json.loads(args.order.read_text())
    ids, roster_provenance = (load_roster(args.roster) if args.roster else
                             (io23.cohort_ids('EAST'), None))
    validate_order(order, ids)
    if (roster_provenance and roster_provenance.get('data_root')
            and Path(roster_provenance['data_root']).resolve() !=
            (io23.DATA_ROOT/'EAST'/'unified_hdf5').resolve()):
        p.error('explicit roster root differs from the configured EAST reader root')
    if args.limit > len(order):
        p.error('limit exceeds the locked stream roster')
    try:
        validate_roster_rate_contract(roster_provenance,args.rate_contract)
    except ValueError as exc:
        p.error(str(exc))
    target_data=SimpleNamespace(read_shot=partial(simple_data.read_shot,rate_contract=args.rate_contract),
                               window=simple_data.window,attach_outcome=simple_data.attach_outcome)
    order_sha256 = _hash(args.order)
    roster_sha256 = _hash(args.roster) if args.roster else None
    args.output.mkdir(parents=True, exist_ok=False)
    _durable_json(args.output/'order.json', order)
    if roster_provenance is not None:
        _durable_json(args.output/'roster.json', roster_provenance)
    site_module = 'model.py' if args.persistent_site == 'tokens' else 'embedding_model.py'
    local_names = (site_module,'intra.py','inter.py','run.py') + (
        ('error_replay.py',) if args.replay_policy == 'error_aware' else ()) + (
        ('source_replay.py',) if args.source_bank is not None else ()) + (
        ('shape_intra.py',) if args.objective == 'shape' else ())
    code_hash = {name:_hash(Path(__file__).with_name(name)) for name in local_names}
    helper_names = ('simple23.py','simple_data.py','io23.py','causal_input.py',
                    'simple_train.py','simple_online.py','fast_update.py')
    helper_hash = {n:_hash(Path(__file__).parents[1]/n) for n in helper_names}
    # Executable source snapshots make uncommitted developmental runs auditable
    # even after later surgical fixes. This archive contains no data or secrets.
    with zipfile.ZipFile(args.output/'code_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for n in local_names:
            archive.write(Path(__file__).with_name(n),arcname=f'redesign_20261004/{n}')
        for n in helper_names:
            archive.write(Path(__file__).parents[1]/n,arcname=n)
    classifier = {name:value.detach().cpu().clone() for name,value in model.base.state_dict().items()
                  if not name.startswith('forecast_head.')}
    _durable_json(args.output/'protocol.json', dict(
        lineage='INDEPENDENT_EAST_INTERNAL_REDESIGN_NOT_V3C',
        source_gate=gate, source_sha256=gate['checkpoint_sha256'],
        checkpoint=str(args.checkpoint.resolve()), model_config=state['model_config'],
        threshold=float(state['threshold']), threshold_origin='fixed J-TEXT VAL160',
        source_label='max(25ms,AdvancedTime)', east_label_ms=125., lead_rule='strict >30ms',
        source_offline_only=True, target_labels_before_first_shot=0, support110_used=False,
        support110_used_for_initialization=False,
        support110_role='no preinitialization; roster-overlap shots follow natural postshot reveal',
        label_access='natural postshot reveal after whole-shot seal, NOT paid active query',
        input_contract='flat23 + masks; CIII=-10 candidate sentinel only; no target scaler fit',
        rate_contract=args.rate_contract,
        rate_atol_hz=io23.rate_tolerance_hz('EAST',args.rate_contract),
        rate_and_times='declared metadata retained, never rounded/resampled; H25 means 25 sample intervals',
        alarm_tuning=False, final_score_correction_head=False, global_LayerNorm_updates=False,
        persistent_site=args.persistent_site,
        persistent_parameter_count=sum(p.numel() for p in model.persistent_parameters()),
        update_policy=args.update_policy, guard_policy=args.guard_policy,
        context_summary_used=args.context_mode if args.persistent_site == 'tokens' else None,
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        roster_count=len(order), roster_provenance=roster_provenance,
        order_sha256=order_sha256, roster_sha256=roster_sha256,
        prior_protocol=prior_protocol, prior_sha256=_hash(args.prior) if args.prior else None,
        source_bank_sha256=source_bank_sha256,
        source_bank_content_sha256=(model.source_bank['bank_sha256']
                                   if args.source_bank is not None else None),
        code_sha256=code_hash, helper_code_sha256=helper_hash,
        code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'),
        torch_version=torch.__version__, device=args.device,
        gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
        interpretation='all data used for development; no independent validation or physics discovery claim'))
    results = {}
    for arm in args.arms:
        candidate = deepcopy(model)
        results[arm] = run_arm(candidate, order=order, arm=arm, threshold=float(state['threshold']),
            output_dir=args.output/arm, limit=args.limit, device=args.device, stride=args.stride,
            step=args.step,radius=args.radius,objective=args.objective,inter_lr=args.inter_lr,
            inter_radius=args.inter_radius,inter_steps=args.inter_steps,seed=args.seed,
            data_api=target_data)
        current = candidate.base.state_dict()
        if any(not torch.equal(current[n].detach().cpu(),v) for n,v in classifier.items()):
            raise AssertionError('immutable source classifier changed')
        torch.save({n:v.detach().cpu() for n,v in candidate.state_dict().items()
                    if not n.startswith('base.')}, args.output/arm/'persistent.pt')
        _durable_json(args.output/'summary.json',results)
    if _hash(args.checkpoint) != gate['checkpoint_sha256']:
        raise AssertionError('qualified source checkpoint overwritten')
    if args.source_bank is not None and _hash(args.source_bank) != source_bank_sha256:
        raise AssertionError('fixed TRAIN-only source replay bank overwritten')
    if _hash(args.order) != order_sha256 or (args.roster and _hash(args.roster) != roster_sha256):
        raise AssertionError('stream selection or order changed during experiment')
    if any(_hash(Path(__file__).with_name(n)) != h for n,h in code_hash.items()):
        raise AssertionError('implementation changed during experiment; results not version-locked')
    if any(_hash(Path(__file__).parents[1]/n) != h for n,h in helper_hash.items()):
        raise AssertionError('input/model helper changed during experiment; results not version-locked')
    return results


if __name__ == '__main__':
    main()
