"""Real cold20-prefix CUDA differentiation check, not a performance experiment."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..attention_basis_20261029.learning import MatrixAttention
from ..causal_input import apply_input_contract
from ..conditional_likelihood_20261029.run import read
from ..delayed_recurrent_memory_20261025.learning import batch
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import ContextTarget, representation_step
from .observation import observation


def execute(output):
    if output.exists() or output.parent != io23.ROOT/'probe_outputs':
        raise ValueError('new worktree probe output required')
    protocol = read(io23.ROOT/'probe_outputs/operational_transport_east360_v1/protocol.json')
    reference = Path(protocol['reference'])
    for name, digest in protocol['reference_hashes'].items():
        assert sha(reference/name) == digest
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('rl_env CUDA required')
    base, _ = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda())
    frozen = {k: v.clone() for k, v in net.state_dict().items()}
    slow = torch.load(reference/'models/cold.pt', map_location='cuda', weights_only=True)['slow']
    slow_copy = slow.clone()
    hashes, mapping = read(reference/'input_hashes.json'), read(reference/'input_map.json')
    ids, observations, later_x, later_a, labels = [71304, 70404], [], [], [], []
    for sid in ids:
        assert sid in protocol['order'][:20]
        assert sha(io23.path_for('EAST', sid)) == hashes[str(sid)]
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        sig = transform({k: raw[k] for k in ('values', 'available')}, mapping)
        observations.append(observation(net, sig, 127))
        x, mask = batch(sig, [153], base.center.device)
        later_x.append(x)
        later_a.append(mask)
        # Both outcomes are from ended cold20, never from a future stream shot.
        flag, _ = io23.read_outcome('EAST', sid)
        labels.append(float(flag))
    data = {k: torch.cat([o[k] for o in observations]) for k in observations[0]}
    x, mask = torch.cat(later_x), torch.cat(later_a)
    controller = ContextTarget().cuda()
    source_state = x.new_zeros(2, 8192)
    with torch.no_grad():
        cold = net(x, mask, source_state, slow, 'qk', forecast=False)['logit']
    state = source_state.clone().requires_grad_()
    h, _, _ = net.first(data['x'], data['available'], state, 'qk')
    delta = controller(data['context'], data['support'])
    # Batched first-layer kernels may differ from per-shot collection by rounding;
    # use the identical current zero-state representation as the detached anchor.
    step = representation_step(h[:, -1], h[:, -1].detach(), delta, state, create_graph=True)
    assert torch.count_nonzero(step) == 0
    logit = net(x, mask, step, slow, 'qk', forecast=False)['logit']
    assert torch.equal(logit, cold)
    loss = F.binary_cross_entropy_with_logits(logit, x.new_tensor(labels))
    optimizer = torch.optim.SGD(controller.parameters(), lr=.001)
    optimizer.zero_grad()
    loss.backward()
    gradient_norm = sum(float(p.grad.square().sum()) for p in controller.parameters() if p.grad is not None)**.5
    assert gradient_norm > 0 and all(p.grad is None for p in net.parameters())
    optimizer.step()
    state2 = source_state.clone().requires_grad_()
    h2, _, _ = net.first(data['x'], data['available'], state2, 'qk')
    delta2 = controller(data['context'], data['support'])
    step2 = representation_step(h2[:, -1], h2[:, -1].detach(), delta2, state2)
    with torch.no_grad():
        after = net(x, mask, step2, slow, 'qk', forecast=False)['logit']
    assert torch.isfinite(after).all() and torch.count_nonzero(step2) > 0
    assert not torch.equal(after, cold)
    assert all(torch.equal(v, frozen[k]) for k, v in net.state_dict().items())
    assert torch.equal(slow, slow_copy) and sha(DEFAULT_PARENT) == PARENT_SHA256
    result = dict(scope='TWO_ENDED_COLD_PREFIXES_CUDA_AUTOGRAD_CHECK_ONLY',
                  shots=ids, origin=127, maturity=152, query=153,
                  zero_initial_step_exact=True, initial_risk_exact=True,
                  outer_gradient_norm=gradient_norm,
                  after_one_teaching_step_norm=step2.norm(dim=1).detach().cpu().tolist(),
                  cold_logit=cold.cpu().tolist(), after_logit=after.cpu().tolist(),
                  source_and_cold_unchanged=True, suffix_labels_read=0,
                  performance_claim=False, full_stream_run=False,
                  code_sha256={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')})
    output.mkdir()
    _durable_json(output/'integration_check.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    execute(parser.parse_args().output.absolute())
