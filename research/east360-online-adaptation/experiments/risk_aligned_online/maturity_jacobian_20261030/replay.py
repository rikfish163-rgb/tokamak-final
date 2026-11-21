"""Recompute saved factorial arrays without fitting or new HDF5 access."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .. import io23
from ..anchored_feedback_20261030.learning import AnchoredTarget
from ..anchored_feedback_20261030.run import evaluate
from ..attention_basis_20261029.learning import MatrixAttention
from ..context_vector_20261030.scaling import transform_context
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from .run import CELLS, check_files, choice, configure, load_inputs, read


def replay(output):
    configure()
    protocol, result = read(output/'protocol.json'), read(output/'complete.json')
    check_files(output, result['artifacts'])
    check_files(io23.ROOT, protocol['code_sha256'])
    origin = Path(protocol['origin'])
    if sha(origin/'complete.json') != protocol['origin_complete_sha256']:
        raise ValueError('origin completion changed')
    old, _, data, mature, records, folds = load_inputs(origin)
    base, _ = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda())
    slow = torch.load(Path(old['parent'])/'models/cold.pt', map_location='cuda', weights_only=True)['slow']
    checks = []
    for spec in folds:
        controllers = {}
        for name, path in [('origin', origin/f"anchored_fold{spec['fold']}.pt"),
                           ('mature', output/f"mature_fold{spec['fold']}.pt")]:
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['spec'] != spec:
                raise ValueError('fold changed')
            controllers[name] = AnchoredTarget().cuda()
            controllers[name].load_state_dict(saved['state_dict'], strict=True)
            controllers[name].eval()
        mean, scale = saved['mean'], saved['scale']
        scaled = transform_context(data['context'], scale)
        donors = torch.tensor(read(origin/f"fold{spec['fold']}_donors.json"), dtype=torch.int64)
        for cell in CELLS:
            teacher, inputs = choice(cell, controllers, data, mature)
            for sid in spec['fit']+spec['held']:
                index = np.array([i for i, r in enumerate(records) if r['shot'] == sid])
                pred = evaluate(net, slow, teacher, 'anchored', inputs, scaled, mean, donors, index)
                with np.load(output/f"{cell}_fold{spec['fold']}_{sid}.npz", allow_pickle=False) as ref:
                    if set(pred) != set(ref.files):
                        raise AssertionError('array keys differ')
                    for name, value in pred.items():
                        if not np.array_equal(value, ref[name]):
                            raise AssertionError(f'mismatch: {cell}/{sid}/{name}')
                        checks.append(dict(cell=cell, fold=spec['fold'], shot=sid, array=name, exact=True))
    if sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError('source changed')
    result = dict(complete=True, exact=True, array_checks=len(checks), checks=checks,
                  refits=0, hdf5_reads=0, source_unchanged=True, protocol_sha256=sha(output/'protocol.json'))
    _durable_json(output/'neural_replay.json', result)
    print({k: v for k, v in result.items() if k != 'checks'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    replay(parser.parse_args().output)
