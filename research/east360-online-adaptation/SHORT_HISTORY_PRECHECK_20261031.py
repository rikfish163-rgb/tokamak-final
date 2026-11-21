"""Reproduce the old cold20 fit and inspect added supervision before a new run."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.risk_aligned_online import simple_data
from experiments.risk_aligned_online.causal_input import apply_input_contract
from experiments.risk_aligned_online.context_memory_20261028.checkpoint import load_context_memory
from experiments.risk_aligned_online.lifecycle_stream_20261025.learning import (
    source_replay, target_example, update,
)
from experiments.risk_aligned_online.lifecycle_stream_20261025.protocol import sampled_endpoints
from experiments.risk_aligned_online.lifecycle_stream_20261025.run import build_adapter
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from experiments.risk_aligned_online.short_history_20261031.sampling import sampled_with_prefix
from experiments.risk_aligned_online.warmstart_alignment_20261022.alignment import transform


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = Path(__file__).with_suffix('.json')
    if output.exists():
        raise FileExistsError(output)
    root = Path(__file__).parent / 'probe_outputs/current_order_context360_v2'
    p = read(root/'protocol.json')
    assert digest(root/'completed.json') == 'cb9cf19197ab25fb6e825687ec002bcad6135834dca1cb7dc3371234eb7883a9'
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    base, _ = load_parent(DEFAULT_PARENT)
    model = build_adapter(base.cuda(), readout='full', feedback='progressive', context=True)
    load_context_memory(model, Path(p['source_memory']['path']))
    source, events = source_replay(model)
    assert events == read(root/'source_replay.json')
    mapping = read(root/'input_map.json')
    order = [r['shot'] for r in read(root/'order.json')]
    examples, coverage = [], []
    for i, sid in enumerate(order[:20], 1):
        path = root/'shots'/f'{i:04d}_{sid}.npz'
        assert digest(path) == read(root/'shots'/f'{i:04d}_{sid}_seal.json')['sha256']
        # These exact first20 have already ended and been sealed in the parent.
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        shot = simple_data.attach_outcome(raw, 'EAST', {})
        signal = transform({k:raw[k] for k in ('values', 'available')}, mapping)
        with np.load(path, allow_pickle=False) as a:
            scores = a['source'].copy()
            assert np.array_equal(raw['times'], a['times'])
        delta, _ = model.memory_delta(signal)
        examples.append(target_example(model, shot, scores, signal, delta))
        down = shot['down'] if shot['is_disrupt'] else None
        args = (shot['times'], shot['y'], shot['valid'], scores)
        old = sampled_endpoints(*args, down=down)
        new = sampled_with_prefix(*args, down=down)
        added = np.setdiff1d(new, old)
        retained = added[shot['y'][added] > 0] if shot['is_disrupt'] else added
        coverage.append(dict(shot=sid, disruptive=shot['is_disrupt'], old_ends=old.tolist(),
                             new_ends=new.tolist(), added_ends=added.tolist(),
                             added_endpoints_used_by_loss=retained.tolist()))
    receipt = update(model, examples, [], source, fast='mixed', steps=20,
                     guarded=False, policy='disruptive_positive_only')
    saved = torch.load(root/'models/cold.pt', map_location='cuda', weights_only=True)
    state = model.adapted_state()
    differences = {k:float((state[k]-saved[k]).abs().max()) for k in saved}
    assert all(torch.equal(state[k], saved[k]) for k in saved), differences
    assert digest(DEFAULT_PARENT) == p['source_sha256']
    result = dict(scope='DEFAULT_COLD20_REPRODUCTION_AND_PREFIX_COVERAGE_NOT_NEW_GAIN',
                  only_ended_east_ids=order[:20], source_event_replay_exact=True,
                  old_fit_reproduced_bitwise=True, parameter_max_abs_difference=differences,
                  old_training_receipt=receipt, coverage=coverage,
                  new_operator_trained=False, source_sha256=digest(DEFAULT_PARENT),
                  script_sha256=digest(__file__), torch=torch.__version__,
                  gpu=torch.cuda.get_device_name())
    with output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(dict(old_fit_bitwise=True, differences=differences,
               added_selected=sum(len(r['added_ends']) for r in coverage),
               added_used=sum(len(r['added_endpoints_used_by_loss']) for r in coverage)), flush=True)


if __name__ == '__main__':
    main()
