"""Could past-only paired fast/off loss have flagged update65067? Posthoc audit."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .. import simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import load_context_memory
from ..east_context_operator_20261031.run import configure, read
from ..lifecycle_stream_20261025.learning import source_replay, target_example, guard_losses
from ..lifecycle_stream_20261025.run import build_adapter
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .run import REFERENCE


@torch.no_grad()
def main(root):
    output = root/'update65067_available_history.json'
    if output.exists():
        raise FileExistsError(output)
    configure()
    p = read(REFERENCE/'protocol.json')
    order = read(REFERENCE/'order.json')
    ids = [r['shot'] for r in order]
    index = ids.index(65067)
    receipt = read(REFERENCE/'receipts.json')[index]['updates']['both']
    assert receipt['accepted'] and index == 221
    expected_ids = ids[index-8:index]
    assert receipt['history_ids'] == expected_ids
    states = read(root/'state_evolution.json')
    selected = {}
    for phase, position in [('before', index), ('after', index+1)]:
        path = root/'states'/f'{position+1:04d}_{ids[position]}.pt'
        assert sha(path) == states[position-20]['state_sha256']
        selected[phase] = torch.load(path, map_location='cuda', weights_only=True)['both']
    base, _ = load_parent(DEFAULT_PARENT)
    model = build_adapter(base.cuda(), readout='full', feedback='progressive', context=True)
    load_context_memory(model, Path(p['source_memory']['path']))
    source, events = source_replay(model)
    assert events == read(REFERENCE/'source_replay.json')
    mapping = read(root/'input_map.json')
    replay, manifest = [], []
    for sid in expected_ids:
        j = ids.index(sid)
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        shot = simple_data.attach_outcome(raw, 'EAST', {})
        signal = transform({k:raw[k] for k in ('values', 'available')}, mapping)
        path = REFERENCE/'shots'/f'{j+1:04d}_{sid}.npz'
        assert sha(path) == read(REFERENCE/'shots'/f'{j+1:04d}_{sid}_seal.json')['sha256']
        with np.load(path, allow_pickle=False) as z:
            scores = z['both'].copy()
        delta, _ = model.memory_delta(signal)
        example = target_example(model, shot, scores, signal, delta)
        replay.append(example)
        manifest.append(dict(shot=sid, rank=j+1, flag=example['flag'], ends=example['ends']))
    losses = {}
    for phase, state in selected.items():
        model.restore(state)
        losses[phase] = {}
        for name, fast in [('on', True), ('off', False)]:
            losses[phase][name] = guard_losses(model, replay, source, fast, policy=p['target_loss'])
        assert losses[phase]['on'] == receipt['guard_'+phase]
    result = {}
    for key in losses['before']['on']:
        before = losses['before']['on'][key]-losses['before']['off'][key]
        after = losses['after']['on'][key]-losses['after']['off'][key]
        result[key] = dict(fast_increment_before=before, fast_increment_after=after,
                           increment_change=after-before, paired_gap_increased=after>before+1e-7)
    recorded = dict(scope='POSTHOC_AVAILABLE_PAST_ONLY_GUARD_DIAGNOSTIC', update_shot=65067,
                    update_rank=index+1, original_accepted=True, original_guard_losses_exact=True,
                    all_history_precedes_update=True, manifest=manifest, losses=losses, paired_gap=result,
                    candidate_gap_rule_rejects=any(v['paired_gap_increased'] for v in result.values()),
                    no_new_rule_deployed=True, no_future_case_inputs_used=True,
                    selection_was_posthoc=True, no_new_policy_metrics=True,
                    source_checkpoint_sha256=sha(DEFAULT_PARENT),
                    completed_sha256=sha(root/'completed.json'), script_sha256=sha(Path(__file__)))
    _durable_json(output, recorded)
    print(recorded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
