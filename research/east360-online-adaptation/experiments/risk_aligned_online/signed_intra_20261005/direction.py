"""One source-only risk-sensitive FFN direction; not a physical/class axis.

No optimizer, target data, source-head change or candidate direction search.
The class-balanced leading singular vector measures sensitivity, not benefit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .. import simple_data
from ..simple_online import _durable_json, _hash
from ..redesign_20261004.prepare_prior import (
    DEFAULT_PARENT, PARENT_SHA256, load_parent, load_source_train,
    classifier_state, assert_classifier_unchanged,
)
from ..redesign_20261004.intra import _batch_windows
from ..ffn_intra_20261005.model import FfnModel
from ..ffn_intra_20261005.source import split_shots


def peak_candidates(shot, first_effect=6):
    if shot['device'] != 'J-TEXT':
        raise ValueError('source-only direction construction')
    mask = np.array(shot['valid'], dtype=bool, copy=True)
    if first_effect < 0:
        raise ValueError('nonnegative first effect required')
    mask[:first_effect] = False
    if shot['is_disrupt']:
        mask &= np.round((shot['down'] - shot['times']) * 1000., 9) > 30.
    return np.flatnonzero(mask)


def balanced_direction(gradients, labels):
    g = torch.as_tensor(gradients).detach().cpu().double()
    y = torch.as_tensor(labels, dtype=torch.long)
    if (g.ndim != 2 or g.shape[0] < 2 or g.shape[1] < 1 or y.shape != (len(g),)
            or not torch.isfinite(g).all() or not bool(((y == 0) | (y == 1)).all())):
        raise ValueError('finite row matrix and binary labels required')
    norms = g.norm(dim=1)
    counts = [int((y == k).sum()) for k in (0, 1)]
    if min(counts) == 0 or not bool((norms > 0).all()):
        raise ValueError('nonzero Jacobians from both classes required')
    normalized = g / norms[:, None]
    weights = torch.tensor([1. / (2 * counts[int(k)]) ** .5 for k in y], dtype=torch.float64)
    matrix = normalized * weights[:, None]
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix @ matrix.T)
    top, second = float(eigenvalues[-1]), max(0., float(eigenvalues[-2]))
    if top <= 0 or top - second <= 1e-8 * top:
        raise ValueError('leading direction is zero or numerically ambiguous')
    direction = matrix.T @ eigenvectors[:, -1]
    direction /= direction.norm()
    anchor = int(direction.abs().argmax())
    if direction[anchor] < 0:
        direction.neg_()
    projections = normalized @ direction
    audit = dict(class_counts=counts, first_eigenvalue=top, second_eigenvalue=second,
                 eigengap_relative=(top-second)/top, ambiguity_tolerance=1e-8,
                 first_energy_fraction=top / float(eigenvalues.sum()), sign_anchor=anchor,
                 normalized_projection_by_class={str(k): dict(
                     mean=float(projections[y == k].mean()),
                     min=float(projections[y == k].min()), max=float(projections[y == k].max()),
                     positive=int((projections[y == k] > 0).sum()), negative=int((projections[y == k] < 0).sum()))
                     for k in (0, 1)})
    return direction.float(), audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent substitute')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(131); started = time.monotonic()
    base, state = load_parent(DEFAULT_PARENT)
    base = base.to(args.device)
    model = FfnModel(base); before = classifier_state(base)
    auxiliary = {k: v.detach().cpu().clone() for k, v in base.forecast_head.state_dict().items()}
    shots, train_ids = load_source_train()
    fit, selection = split_shots(shots, 131)
    if len(fit) != 1083 or len(selection) != 270 or set(s['shot'] for s in fit) & set(s['shot'] for s in selection):
        raise ValueError('canonical 1083/270 source split required')
    root = Path(__file__).parents[1]
    files = [Path(__file__), root/'simple23.py', root/'simple_data.py', root/'simple_online.py',
             root/'io23.py', root/'ffn_intra_20261005/model.py', root/'ffn_intra_20261005/source.py',
             root/'redesign_20261004/prepare_prior.py', root/'redesign_20261004/intra.py']
    code = {str(f.resolve()): _hash(f) for f in files}
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, str(f.relative_to(root)))
    protocol = dict(parent_sha256=PARENT_SHA256, seed=131, source_only=True,
        source_train_ids=train_ids, fit_ids=[s['shot'] for s in fit],
        selection_ids=sorted(s['shot'] for s in selection), selection_used_for_fit=False,
        source_loader='reads TRAIN1353 only; 270 selection not used to construct direction',
        first_effect=6, positive_scope='strict lead>30ms without upper bound, source TRAIN offline only',
        static_classifier_frozen=True, threshold=float(state['threshold']),
        parameter_names=list(model.parameter_names), parameter_count=model.delta_count,
        direction='leading class-balanced normalized static-peak logit Jacobian; NOT a physical or contrastive axis',
        code_sha256=code, hardware=torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu',
        torch_version=str(torch.__version__), evidence='DEVELOPMENT_ONLY; no EAST inference or online policy')
    _durable_json(args.output/'protocol.json', protocol)
    rows, gradients, labels = [], [], []
    for index, shot in enumerate(fit):
        candidates = peak_candidates(shot)
        row = dict(shot=int(shot['shot']), label=int(shot['is_disrupt']), eligible=len(candidates))
        if not len(candidates):
            rows.append(dict(row, reason='no_eligible_source_endpoint')); continue
        digest = hashlib.sha256()
        for key in ('values', 'available', 'times', 'valid'):
            digest.update(np.ascontiguousarray(shot[key]).tobytes())
        digest.update(json.dumps([bool(shot['is_disrupt']), shot['down']], allow_nan=False).encode())
        signals = {k: shot[k] for k in ('values', 'available')}
        logits = []
        for start in range(0, len(candidates), 256):
            x, a = _batch_windows(simple_data, signals, candidates[start:start+256], model, args.device)
            out = model(x, a, model.initial_delta())['logit']
            logits.extend(out.detach().cpu().tolist())
        if not np.isfinite(logits).all():
            raise FloatingPointError('nonfinite static source risk')
        peak = int(candidates[int(np.argmax(logits))])
        x, a = _batch_windows(simple_data, signals, [peak], model, args.device)
        delta = model.initial_delta()
        logit = model(x, a, delta)['logit']
        if index == 0:
            direct = base(x, a, base.initial_fast(1).detach().requires_grad_(True))['logit']
            if not torch.equal(logit, direct):
                raise RuntimeError('zero FFN state differs from same-dispatch source')
        gradient, = torch.autograd.grad(logit.sum(), delta)
        norm = float(gradient.detach().double().norm())
        if not np.isfinite(norm):
            raise FloatingPointError('nonfinite source Jacobian')
        row.update(peak_index=peak, peak_logit=float(logit.detach()), gradient_norm=norm,
                   input_sha256=digest.hexdigest())
        if norm == 0:
            rows.append(dict(row, reason='zero_jacobian')); continue
        rows.append(dict(row, reason='included')); gradients.append(gradient.detach().cpu())
        labels.append(int(shot['is_disrupt']))
        if (index + 1) % 100 == 0:
            print(json.dumps(dict(stage='source_peak_jacobians', done=index+1, total=len(fit),
                                  seconds=time.monotonic()-started)), flush=True)
    matrix = torch.stack(gradients)
    direction, audit = balanced_direction(matrix, labels)
    assert_classifier_unchanged(base, before)
    if any(not torch.equal(auxiliary[k], v.detach().cpu()) for k, v in base.forecast_head.state_dict().items()):
        raise RuntimeError('auxiliary asset changed')
    if _hash(DEFAULT_PARENT) != PARENT_SHA256:
        raise RuntimeError('parent file changed')
    torch.save(dict(direction=direction, protocol=protocol, audit=audit), args.output/'direction.pt')
    np.savez_compressed(args.output/'jacobians.npz', gradients=matrix.numpy(), labels=np.array(labels))
    _durable_json(args.output/'source_peaks.json', rows)
    summary = dict(protocol=protocol, audit=audit, used=len(matrix), excluded=len(fit)-len(matrix),
                   direction_sha256=_hash(args.output/'direction.pt'), classifier_unchanged=True,
                   zero_probe_bit_exact=True, elapsed_seconds=time.monotonic()-started)
    _durable_json(args.output/'summary.json', summary)
    print(json.dumps(dict(stage='direction_complete', used=len(matrix), **audit,
                          seconds=summary['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
