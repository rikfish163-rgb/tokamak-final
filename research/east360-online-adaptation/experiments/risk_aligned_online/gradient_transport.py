"""Source-trained signal-gradient to task-gradient rule; unchanged risk graph.

Only offline TRAIN pairs supply future risk labels. A target step receives
the matured real-signal gradient, never a target outcome or virtual input.
The fixed ridge rule acts on original final-FFN disposable parameters.
"""
import argparse
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from . import simple_data, fast_meta
from .fast_direction import episode
from .fast_update import normalized_step
from .persistent_ffn import PersistentFfnModel
from .ffn_intra_20261005.source import split_shots
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent, load_source_train
from .redesign_20261004.intra import predict_source_diagnostic
from .simple_online import _durable_json, _hash, seal_predictions
from .simple_train import metrics


def fit_transport(signal, risk, *, ridge=.1):
    if (signal.ndim != 2 or signal.shape != risk.shape or not len(signal)
            or signal.device != risk.device or signal.dtype != risk.dtype
            or not torch.isfinite(signal).all() or not torch.isfinite(risk).all()
            or not np.isfinite(ridge) or ridge <= 0
            or not (signal.double().norm(dim=-1) > 0).all()
            or not (risk.double().norm(dim=-1) > 0).all()):
        raise ValueError('paired finite nonzero parameter gradients and positive ridge required')
    keys = F.normalize(signal.double(), dim=-1)
    targets = F.normalize(risk.double(), dim=-1)
    gram = keys @ keys.T + ridge*torch.eye(len(keys), device=keys.device, dtype=keys.dtype)
    coefficients = torch.linalg.solve(gram, targets)
    if not torch.isfinite(coefficients).all(): raise FloatingPointError('nonfinite gradient transport fit')
    return keys.to(signal), coefficients.to(signal)


def transport_gradient(gradient, keys, coefficients):
    if (gradient.ndim != 2 or keys.ndim != 2 or keys.shape != coefficients.shape
            or gradient.shape[1] != keys.shape[1]
            or gradient.device != keys.device or coefficients.device != keys.device
            or gradient.dtype != keys.dtype or coefficients.dtype != keys.dtype
            or not torch.isfinite(gradient).all() or not torch.isfinite(keys).all()
            or not torch.isfinite(coefficients).all()):
        raise ValueError('matching finite source-space gradient transport tensors required')
    unit = gradient / gradient.double().norm(dim=-1, keepdim=True).clamp_min(1e-12).to(gradient)
    mapped = (unit @ keys.T) @ coefficients
    # Remove only the component that increases the SIGNAL loss to first order.
    # This is an explicit cone constraint, not a risk guarantee. A zero
    # projected direction stays zero; no hidden identity-gradient fallback.
    conflict = (mapped*unit).sum(-1, keepdim=True).clamp(max=0)
    result = mapped-conflict*unit
    if not torch.isfinite(result).all(): raise FloatingPointError('nonfinite transported direction')
    return result


class TransportFfnModel(PersistentFfnModel):
    fast_gradient_transform = 'source TRAIN ridge gradient transport'

    def __init__(self, base, keys, coefficients):
        super().__init__(base)
        if (keys.ndim != 2 or not len(keys) or keys.shape != coefficients.shape
                or keys.shape[1] != self.temporary_dim or not torch.isfinite(keys).all()
                or not torch.isfinite(coefficients).all()):
            raise ValueError('finite original-FFN gradient operator required')
        self.register_buffer('transport_keys', keys.detach().clone().to(self.center))
        self.register_buffer('transport_coefficients', coefficients.detach().clone().to(self.center))

    def intra_gradient_transform(self, gradient):
        return transport_gradient(gradient, self.transport_keys, self.transport_coefficients)


def source_pairs(model, shots, *, count, seed):
    pools = fast_meta.episode_pools(shots, model.cfg)
    rng = np.random.default_rng(seed)
    signals, risks, records = [], [], []
    for index in range(count):
        group = pools[index % 2]
        sid, origins = group[int(rng.integers(len(group)))]
        shot = shots[sid]; origin = int(rng.choice(origins))
        e = episode(shot, origin, model.cfg, model.center.device)
        fast = model.initial_fast(1).requires_grad_()
        output = model(e['x'], e['a'], fast)
        loss = model.forecast_loss(output['forecast'], e['truth'], e['mask'])
        signal, = torch.autograd.grad(loss, fast)
        fast = model.initial_fast(1).requires_grad_()
        risk_loss = F.binary_cross_entropy_with_logits(model(e['qx'], e['qa'], fast)['logit'], e['y'])
        risk, = torch.autograd.grad(risk_loss, fast)
        if not signal.double().norm() or not risk.double().norm():
            raise ValueError('sampled source pair has no usable gradient; no silent resampling')
        signals.append(signal.detach()[0].cpu()); risks.append(risk.detach()[0].cpu())
        records.append(dict(shot=shot['shot'], origin=origin, query=origin+26,
                            source_point_label=float(e['y'][0]),
                            signal_loss=float(loss.detach()), risk_loss=float(risk_loss.detach())))
        if (index+1) % 128 == 0:
            print(json.dumps(dict(source_pairs=index+1, total=count)), flush=True)
    return torch.stack(signals), torch.stack(risks), records


def fit(output):
    started = time.monotonic()
    base, state = load_parent(DEFAULT_PARENT)
    model = PersistentFfnModel(base).to('cuda').eval()
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    shots, train_ids = load_source_train()
    fitting, selection = split_shots(shots, 131)
    signal, risk, records = source_pairs(model, fitting, count=512, seed=131)
    held_signal, held_risk, held_records = source_pairs(model, selection, count=256, seed=132)
    keys, coefficients = fit_transport(signal.cuda(), risk.cuda(), ridge=.1)
    mapped = transport_gradient(held_signal.cuda(), keys, coefficients).cpu()
    alignment = F.cosine_similarity(mapped, held_risk, dim=-1)
    raw_alignment = F.cosine_similarity(held_signal, held_risk, dim=-1)
    protocol = dict(parent_sha256=PARENT_SHA256, source_only=True, no_east_reads=True,
        source_classifier_and_decoder_frozen=True, source_scaler_frozen=True,
        source_labels_unchanged='max(25ms,AdvancedTime)', source_train_ids=train_ids,
        fit_ids=[s['shot'] for s in fitting], selection_ids=[s['shot'] for s in selection],
        fit_pairs=512, selection_pairs=256, seed=131, selection_seed=132, ridge=.1,
        rule='unit signal gradients; ridge map to unit actual future source risk gradients; signal descent half-space',
        origin_truth='real next25', supervised_query='real origin+26 source point label; offline TRAIN only',
        parameter_count=model.temporary_dim, parameter_site=model.adaptation_site,
        step=.05, radius=1., stride=100, selection='fixed fit budget; no holdout or EAST tuning',
        scope='SOURCE AUXILIARY RULE DEVELOPMENT; not a new source classifier, physical law or online performance')
    output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), Path(__file__).parent/'persistent_ffn.py',
             Path(__file__).parent/'redesign_20261004/intra.py', Path(__file__).parent/'ffn_intra_20261005/model.py',
             Path(__file__).parent/'fast_direction.py', Path(__file__).parent/'fast_update.py',
             Path(__file__).parent/'fast_meta.py', Path(__file__).parent/'simple23.py',
             Path(__file__).parent/'simple_data.py', Path(__file__).parent/'io23.py',
             Path(__file__).parent/'redesign_20261004/prepare_prior.py', Path(__file__).parent/'ffn_intra_20261005/source.py']
    with zipfile.ZipFile(output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for file in files: archive.write(file, str(file.relative_to(Path(__file__).parent)))
    protocol['code_snapshot_sha256'] = _hash(output/'code_snapshot.zip')
    with (output/'operator.pt').open('xb') as handle:
        torch.save(dict(keys=keys.cpu(), coefficients=coefficients.cpu(), protocol=protocol), handle)
    _durable_json(output/'protocol.json', protocol)
    _durable_json(output/'pairs.json', dict(fit=records, selection=held_records))
    result = dict(raw_mean_cosine=float(raw_alignment.mean()), mapped_mean_cosine=float(alignment.mean()),
        raw_positive=int((raw_alignment > 0).sum()), mapped_positive=int((alignment > 0).sum()),
        held_pairs=256, source_parent_unchanged=True, elapsed_seconds=time.monotonic()-started,
        interpretation='direction diagnostic only; complete source/EAST risk not qualified')
    if (_hash(DEFAULT_PARENT) != PARENT_SHA256
            or any(not torch.equal(v.detach().cpu(), frozen[k]) for k, v in model.state_dict().items())):
        raise ValueError('source model or internal memory changed')
    _durable_json(output/'result.json', result)
    print(json.dumps(result), flush=True)


def validate(operator, output, *, role='val'):
    if role not in ('val', 'test'): raise ValueError('explicit source val/test development role required')
    asset = torch.load(operator, map_location='cpu', weights_only=True)
    protocol = asset['protocol']
    roles = simple_data.source_roles()
    if (protocol['parent_sha256'] != PARENT_SHA256 or not protocol['source_only'] or not protocol['no_east_reads']
            or set(protocol['source_train_ids']) != set(roles['train'])
            or set(protocol['fit_ids']) & set(protocol['selection_ids'])
            or set(protocol['fit_ids']) | set(protocol['selection_ids']) != set(roles['train'])):
        raise ValueError('matching source TRAIN operator required')
    base, state = load_parent(DEFAULT_PARENT)
    models = dict(static=PersistentFfnModel(base).to('cuda').eval(),
                  plain=PersistentFfnModel(base).to('cuda').eval(),
                  transported=TransportFfnModel(base, asset['keys'], asset['coefficients']).to('cuda').eval())
    snapshots = {arm: {k: v.detach().cpu().clone() for k,v in model.state_dict().items()} for arm,model in models.items()}
    output.mkdir(parents=True, exist_ok=False)
    for arm in models: (output/arm).mkdir()
    files = [Path(__file__), Path(__file__).parent/'persistent_ffn.py',
             Path(__file__).parent/'redesign_20261004/intra.py', Path(__file__).parent/'ffn_intra_20261005/model.py',
             Path(__file__).parent/'simple23.py', Path(__file__).parent/'simple_data.py',
             Path(__file__).parent/'simple_online.py', Path(__file__).parent/'simple_train.py', Path(__file__).parent/'io23.py']
    with zipfile.ZipFile(output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for file in files: archive.write(file, str(file.relative_to(Path(__file__).parent)))
    advanced = simple_data.advanced_times(); rows = {arm: [] for arm in models}; update_counts = dict(plain=0, transported=0)
    _durable_json(output/'protocol.json', dict(parent_sha256=PARENT_SHA256, operator_sha256=_hash(operator),
        source_evaluation_role=role, source_evaluation_ids=roles[role],
        code_snapshot_sha256=_hash(output/'code_snapshot.zip'),
        threshold=float(state['threshold']), no_east_reads=True,
        source_training=False, jtext_online=False, scope='OFFLINE WHOLE-SOURCE RULE GATE',
        step=.05, radius=1., stride=100, original_label='max(25ms,AdvancedTime)'))
    for index, sid in enumerate(roles[role], 1):
        shot = simple_data.attach_outcome(simple_data.read_shot('J-TEXT', sid), 'J-TEXT', advanced)
        for arm, model in models.items():
            scores, audit = predict_source_diagnostic(model, shot, intra=arm!='static', device='cuda')
            valid = shot['valid']
            seal_predictions(output/arm, index, sid, shot['times'][valid], scores[valid], arm=arm, threshold=state['threshold'])
            _durable_json(output/arm/f'{index:04d}_{sid}_audit.json', audit)
            if arm != 'static': update_counts[arm] += audit['accepted_updates']
            rows[arm].append(dict(shot=sid, is_disrupt=shot['is_disrupt'], down=shot['down'],
                                  times=shot['times'][valid], scores=scores[valid]))
        if index % 20 == 0: print(json.dumps(dict(source_role=role, completed=index, total=len(roles[role]))), flush=True)
    for arm, model in models.items():
        if any(not torch.equal(v.detach().cpu(), snapshots[arm][k]) for k,v in model.state_dict().items()):
            raise ValueError('source gate retained an update')
    measured = {arm: metrics(row, state['threshold'], lead=10.) for arm,row in rows.items()}
    result = dict(metrics=measured, accepted_updates=update_counts,
                  source_gate={arm: m['tpr']>=.9 and m['fpr']<.2 for arm,m in measured.items()},
                  scope='SOURCE DEVELOPMENT ONLY; no EAST or future independent validation')
    _durable_json(output/'result.json', result)
    if _hash(DEFAULT_PARENT) != PARENT_SHA256: raise ValueError('parent changed')
    print(json.dumps(dict(metrics={arm: {k:m[k] for k in ('tp','fp','tpr','fpr')} for arm,m in measured.items()},
                         accepted_updates=update_counts, source_gate=result['source_gate'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('fit','validate'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--operator', type=Path)
    parser.add_argument('--role', choices=('val','test'), default='val', help='explicit source development evaluation role')
    args = parser.parse_args()
    if args.output.exists(): parser.error('new output required')
    if (args.stage=='validate') != (args.operator is not None): parser.error('only validation requires operator')
    torch.set_num_threads(4); torch.manual_seed(131)
    if args.stage == 'fit': fit(args.output)
    else: validate(args.operator, args.output, role=args.role)


if __name__ == '__main__': main()
