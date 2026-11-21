"""Frozen-fit diagnostics on the two already-ended cold20 safe shots, CPU only."""
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.risk_aligned_online import simple_data
from experiments.risk_aligned_online.causal_input import apply_input_contract
from experiments.risk_aligned_online.context_memory_20261028.checkpoint import load_context_memory
from experiments.risk_aligned_online.delayed_recurrent_memory_20261025.learning import batch
from experiments.risk_aligned_online.lifecycle_stream_20261025.run import build_adapter
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from experiments.risk_aligned_online.warmstart_alignment_20261022.alignment import transform


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = Path(__file__).with_suffix('.json')
    if output.exists():
        raise FileExistsError(output)
    root = Path(__file__).parent
    old = root/'probe_outputs/current_order_context360_v2'
    new = root/'probe_outputs/short_history_context360_v1'
    protocol = read(new/'protocol.json')
    p = read(root/'SHORT_HISTORY_PRECHECK_20261031.json')
    assert p['old_fit_reproduced_bitwise']
    mapping = read(new/'input_map.json')
    assert mapping == read(old/'input_map.json')
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    base, _ = load_parent(DEFAULT_PARENT)
    model = build_adapter(base, readout='full', feedback='progressive', context=True)
    load_context_memory(model, Path(protocol['source_memory']['path']))
    states = dict(source_prior=model.adapted_state(),
                  original_cold=torch.load(old/'models/cold.pt', map_location='cpu', weights_only=True),
                  prefix_cold=torch.load(new/'models/cold.pt', map_location='cpu', weights_only=True))
    records = []
    for e in p['coverage']:
        if e['disruptive']:
            continue
        sid = e['shot']
        assert sid in mapping['calibration_ids']
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        flag, _ = simple_data.io23.read_outcome('EAST', sid)
        assert not flag
        signal = transform({k:raw[k] for k in ('values', 'available')}, mapping)
        for state_name, state in states.items():
            model.restore(state)
            delta, _ = model.memory_delta(signal)
            for region, ends in [('new_prefix', e['added_ends']), ('old_endpoints', e['old_ends'])]:
                with torch.no_grad():
                    x, m = batch(signal, ends, 'cpu')
                    h, valid = model.tokens(x, m)
                for mode in ('off', 'fast'):
                    d = delta[ends] if mode == 'fast' else torch.zeros_like(delta[ends])
                    z = model.risk(h, valid, d)
                    loss = F.binary_cross_entropy_with_logits(z, torch.zeros_like(z))
                    gradients = torch.autograd.grad(loss, (model.slow.weight, model.memory.output.weight))
                    probabilities = z.detach().sigmoid().numpy()
                    records.append(dict(shot=sid, state=state_name, region=region, mode=mode,
                        endpoints=ends, window_count=len(ends), mean_BCE=float(loss.detach()),
                        probability_mean=float(probabilities.mean()),
                        probability_max=float(probabilities.max()),
                        crossings=int((probabilities >= protocol['threshold']).sum()),
                        slow_gradient_norm=float(gradients[0].norm()),
                        output_gradient_norm=float(gradients[1].norm())))
    result = dict(scope='CPU_FROZEN_PRE_POST_FIT_TRAINING_DIAGNOSTIC_NOT_ONLINE_SCORE',
                  optimizer_steps=0, gpu_used=False, training_ids=[83646, 87484],
                  disputed_safe_training_shot=83646, uses_future_suffix=False,
                  threshold=protocol['threshold'], records=records,
                  initial_context_sha=protocol['source_memory']['sha256'],
                  old_cold_sha=sha(old/'models/cold.pt'), new_cold_sha=sha(new/'models/cold.pt'),
                  script_sha256=sha(__file__),
                  interpretation='Endpoint gradients are local sensitivities, not a causal '
                                 'leave-one-shot-out contribution or future benefit estimate.')
    with output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(records, indent=2), flush=True)


if __name__ == '__main__':
    main()
