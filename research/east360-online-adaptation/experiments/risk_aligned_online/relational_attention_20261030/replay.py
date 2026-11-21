"""Replay all sealed component outputs without refitting any reference."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23, simple_data
from ..attention_basis_20261029.learning import MatrixAttention
from ..causal_input import apply_input_contract
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import bounded_step
from .run import ARMS, chunks, evaluate, gradients, schedule, targets, windows


def execute(output):
    protocol = json.loads((output/'protocol.json').read_text())
    complete = json.loads((output/'complete.json').read_text())
    destination = output/'neural_replay.json'
    if destination.exists():
        raise ValueError('replay output already exists')
    for name, digest in complete['artifact_sha256'].items():
        if sha(output/name) != digest:
            raise ValueError('saved artifact changed: '+name)
    for name, digest in protocol['code_sha256'].items():
        if sha(io23.ROOT/name) != digest:
            raise ValueError('frozen dependency changed: '+name)
    reference = Path(protocol['reference'])
    for name, digest in protocol['reference_hashes'].items():
        if sha(reference/name) != digest:
            raise ValueError('parent dependency changed')
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for same-runtime replay')
    base, _ = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda())
    frozen = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    slow = torch.load(reference/'models/cold.pt', map_location='cuda', weights_only=True)['slow']
    mapping = json.loads((reference/'input_map.json').read_text())
    hashes = json.loads((reference/'input_hashes.json').read_text())
    checks, causal, opportunities = [], [], 0
    for fold in json.loads((output/'folds.json').read_text()):
        policy = torch.load(output/f"fold{fold['fold']}.pt", map_location='cuda', weights_only=True)
        ref, mean, mean_g = policy['reference'], policy['mean_target'], policy['mean_gradient']
        for sid in fold['held']:
            if sha(io23.path_for('EAST', sid)) != hashes[str(sid)]:
                raise ValueError('raw shot changed')
            raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
            sig = transform({k: raw[k] for k in ('values', 'available')}, mapping)
            sched = schedule(len(sig['values']))
            data = windows(sig, [m for _, m, _ in sched])
            later = windows(sig, [q for _, _, q in sched])
            saved = np.load(output/f'{sid}_predictions.npz')
            parts = {}
            for sl, d in chunks(data, 'cuda'):
                q = {k: v[sl].to('cuda') for k, v in later.items()}
                pred = evaluate(net, d, q, ref, mean, mean_g, slow)
                for k, v in pred.items():
                    parts.setdefault(k, []).append(v)
            if set(saved.files) != set(parts):
                raise AssertionError('replay arrays do not match saved arrays')
            for name, pieces in parts.items():
                replayed = np.concatenate(pieces)
                old = saved[name]
                exact = np.array_equal(replayed, old)
                checks.append(dict(shot=sid, array=name, exact=exact,
                    max_absolute_difference=float(np.max(np.abs(replayed.astype(float)-old.astype(float))))))
                if not exact:
                    raise AssertionError(f'nonexact replay {sid}/{name}')
            opportunities += len(sched)
            # One fixed middle opportunity per shot: changed FUTURE values and
            # masks must not alter its window, relationship target or actual step.
            _, maturity, _ = sched[len(sched)//2]
            changed = {k: v.copy() for k, v in sig.items()}
            changed['values'][maturity+1:] = 12345.
            changed['available'][maturity+1:] = False
            left = {k: v.cuda() for k, v in windows(sig, [maturity]).items()}
            right = {k: v.cuda() for k, v in windows(changed, [maturity]).items()}
            if not all(torch.equal(left[k], right[k]) for k in left):
                raise AssertionError('future alteration changed causal window')
            lt, le = targets(net, left, ref, mean)
            rt, re = targets(net, right, ref, mean)
            if not torch.equal(le, re) or not all(torch.equal(lt[k], rt[k]) for k in lt):
                raise AssertionError('future alteration changed target')
            steps = {}
            for a in ARMS[:4]:
                lg, _ = gradients(net, left, lt[a], le, slow)
                rg, _ = gradients(net, right, rt[a], re, slow)
                step = bounded_step(lg)
                if not torch.equal(step, bounded_step(rg)):
                    raise AssertionError('future alteration changed update')
                matrices = step.reshape(-1, 2, 64, 64)
                if torch.count_nonzero(matrices[:, :, 16:]):
                    raise AssertionError('non-head0 row updated')
                steps[a] = float(step.norm())
            causal.append(dict(shot=sid, maturity=maturity, exact_future_invariance=True,
                               head_zero_only=True, norms=steps))
    if any(not torch.equal(v.cpu(), frozen[k]) for k, v in net.state_dict().items()):
        raise AssertionError('model mutated during replay')
    result = dict(complete=True, observations_replayed=opportunities,
                  array_checks=len(checks), exact=all(c['exact'] for c in checks),
                  checks=checks, causal_tests=causal, refits=0, outcomes_read=0,
                  original_protocol_sha256=sha(output/'protocol.json'),
                  verifier_sha256=sha(Path(__file__)), source_unchanged=True)
    _durable_json(destination, result)
    print({k: v for k, v in result.items() if k not in ('checks', 'causal_tests')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    execute(parser.parse_args().output.absolute())
