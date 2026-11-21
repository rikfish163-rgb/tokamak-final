"""Exact saved-model replay of fit and held arrays; no fitting or raw outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23
from ..attention_basis_20261029.learning import MatrixAttention
from ..context_vector_20261030.learning import ContextTarget
from ..context_vector_20261030.scaling import transform_context
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import AnchoredTarget
from .run import KINDS, evaluate


def replay(output):
    protocol = json.loads((output/'protocol.json').read_text())
    completed = json.loads((output/'complete.json').read_text())
    for name, digest in completed['artifacts'].items():
        if sha(output/name) != digest:
            raise ValueError('artifact changed: '+name)
    for name, digest in protocol['code_sha256'].items():
        if sha(io23.ROOT/name) != digest:
            raise ValueError('code changed: '+name)
    previous, reference = Path(protocol['previous']), Path(protocol['parent'])
    for name, digest in protocol['input_hashes'].items():
        if sha(previous/name) != digest:
            raise ValueError('previous input changed: '+name)
    for name, digest in protocol['parent_hashes'].items():
        if sha(reference/name) != digest:
            raise ValueError('parent changed: '+name)
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    base, _ = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda())
    slow = torch.load(reference/'models/cold.pt', map_location='cuda', weights_only=True)['slow']
    data = torch.load(previous/'observations.pt', map_location='cpu', weights_only=True)
    records = json.loads((output/'records.json').read_text())
    folds = json.loads((output/'folds.json').read_text())
    checks = []
    for spec in folds:
        donors = torch.tensor(json.loads((output/f"fold{spec['fold']}_donors.json").read_text()), dtype=torch.int64)
        for kind in KINDS:
            path = (output/f"anchored_fold{spec['fold']}.pt" if kind == 'anchored'
                    else previous/f"representation_fold{spec['fold']}.pt")
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['spec'] != spec:
                raise ValueError('fold changed')
            controller = (AnchoredTarget() if kind == 'anchored' else ContextTarget()).cuda()
            controller.load_state_dict(saved['state_dict'], strict=True)
            controller.eval()
            scaled = transform_context(data['context'], saved['scale'])
            for sid in spec['fit']+spec['held']:
                index = np.array([i for i, r in enumerate(records) if r['shot'] == sid])
                pred = evaluate(net, slow, controller, kind, data, scaled, saved['mean'], donors, index)
                with np.load(output/f"{kind}_fold{spec['fold']}_{sid}.npz", allow_pickle=False) as original:
                    if set(original.files) != set(pred):
                        raise AssertionError('array keys changed')
                    for name, value in pred.items():
                        exact = np.array_equal(value, original[name])
                        checks.append(dict(fold=spec['fold'], kind=kind, shot=sid, array=name, exact=exact,
                            max_difference=float(np.max(np.abs(value.astype(float)-original[name].astype(float))))))
                        if not exact:
                            raise AssertionError(f'replay mismatch: {kind} {sid} {name}')
    if sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError('source changed')
    result = dict(complete=True, exact=True, checks=checks, array_checks=len(checks),
                  refits=0, raw_outcome_reads=0, source_unchanged=True,
                  protocol_sha256=sha(output/'protocol.json'))
    _durable_json(output/'neural_replay.json', result)
    print({k: v for k, v in result.items() if k != 'checks'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    replay(parser.parse_args().output)
