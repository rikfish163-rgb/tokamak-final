"""Real unlabeled suffix prefix: exact cold replay and causal perturbation."""

import numpy as np
import torch

from .. import io23, simple_data
from ..attention_basis_20261029.learning import MatrixAttention
from ..causal_input import apply_input_contract
from ..conditional_likelihood_20261029.run import read
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import ContextTarget
from .stream import VARIANTS, predict


def execute():
    output = io23.ROOT/'probe_outputs/context_vector_stream_smoke_v1'
    if output.exists():
        raise ValueError('do not overwrite prior smoke')
    previous = io23.ROOT/'probe_outputs/arrival_reference_east360_v1'
    spec = read(previous/'protocol.json')
    reference = io23.ROOT/spec['reference']
    sid = spec['order'][20]
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base, _ = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda())
    frozen = {k:v.clone() for k,v in net.state_dict().items()}
    slow = torch.load(reference/'models/cold.pt', map_location='cuda', weights_only=True)['slow']
    policies = {}
    for variant in VARIANTS:
        path = io23.ROOT/'probe_outputs/context_vector_factorial_v1'/variant/'foldall.pt'
        payload = torch.load(path, map_location='cpu', weights_only=True)
        controller = ContextTarget().cuda().eval().requires_grad_(False)
        controller.load_state_dict(payload['state'])
        policies[variant] = dict(controller=controller, scale=payload['scale'],
                                 mean_context=payload['mean_context'].cuda(), radius=payload['radius'])
    raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
    signal = transform({k:raw[k][:600].copy() for k in ('values','available')}, read(reference/'input_map.json'))
    curves, receipts = predict(net, signal, slow, policies)
    with np.load(previous/'shots'/f'0021_{sid}.npz', allow_pickle=False) as old:
        np.testing.assert_array_equal(curves['cold'], old['cold'][:600])
    changed = {k:v.copy() for k,v in signal.items()}
    changed['values'][400:] *= 3
    altered, _ = predict(net, changed, slow, policies)
    short, _ = predict(net, {k:v[:400] for k,v in signal.items()}, slow, policies)
    for arm in curves:
        np.testing.assert_array_equal(curves[arm][:400], altered[arm][:400])
        np.testing.assert_array_equal(curves[arm][:400], short[arm])
    assert all(torch.equal(v, frozen[k]) for k,v in net.state_dict().items())
    result = dict(scope='UNLABELED_REAL_PREFIX_STREAM_SMOKE_ONLY', shot=sid, points=600,
                  suffix_labels_read=0, full_stream_run=False, arms=len(curves),
                  exact_cold_archive=True, exact_future_perturbation=True, exact_prefix_truncation=True,
                  updates=len(receipts), max_norm=max(r['max_bound'] for r in receipts),
                  source_unchanged=True, stream_code_sha256=sha(__file__.replace('stream_smoke.py','stream.py')))
    output.mkdir()
    _durable_json(output/'check.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    execute()
