"""Past-only loss-gap census over every accepted update in a fixed history."""
import argparse
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import load_context_memory
from ..east_context_operator_20261031.run import configure, read
from ..lifecycle_stream_20261025.learning import source_replay, target_example, guard_losses
from ..lifecycle_stream_20261025.run import build_adapter
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..slow_fast_attribution_20261031.run import REFERENCE
from ..warmstart_alignment_20261022.alignment import transform
from .rules import evaluate_guards

STATES = io23.ROOT/'probe_outputs/slow_fast_attribution_v1'
STATES_SHA = 'bbbea164486e5f515942e4e078052a504016d67244db4f3e1e15bb0504700f79'
PLAN = io23.ROOT/'PAIRED_GAP_CENSUS_PLAN_20261031.md'


@torch.no_grad()
def main(root):
    if root.exists():
        raise FileExistsError(root)
    configure()
    start = time.monotonic()
    assert sha(STATES/'completed.json') == STATES_SHA
    p = read(REFERENCE/'protocol.json')
    order = read(REFERENCE/'order.json')
    ids = [r['shot'] for r in order]
    receipts = read(REFERENCE/'receipts.json')
    raw_hashes = read(REFERENCE/'input_hashes.json')
    state_manifest = read(STATES/'state_evolution.json')
    selected = [i for i in range(20, 360) if receipts[i]['updates']['both']['accepted']]
    assert len(selected) == 57
    base, _ = load_parent(DEFAULT_PARENT)
    model = build_adapter(base.cuda(), readout='full', feedback='progressive', context=True)
    assert load_context_memory(model, Path(p['source_memory']['path'])) == p['source_memory']
    source, events = source_replay(model)
    assert events == read(REFERENCE/'source_replay.json')
    mapping = read(REFERENCE/'input_map.json')
    root.mkdir(parents=True)
    (root/'rows').mkdir()
    files = set(Path(__file__).parent.glob('*.py')) | {PLAN}
    for name, module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module, '__file__', None):
            files.add(Path(module.__file__))
    _durable_json(root/'protocol.json', dict(scope='ORIGINAL_ACCEPTED_UPDATES_ONLY_NOT_NEW_POLICY',
        source_sha256=PARENT_SHA256, state_parent=str(STATES), state_completed_sha256=STATES_SHA,
        original_parent=str(REFERENCE), original_completed_sha256=sha(REFERENCE/'completed.json'),
        selected_ranks=[i+1 for i in selected], selected_update_shots=[ids[i] for i in selected],
        original_accepted=57, original_rejected_not_recomputed=283, tolerance=1e-7,
        all_groups_past_only=True, plan_sha256=sha(PLAN), independent_confirmation=False,
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files}))
    with zipfile.ZipFile(root/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))
    cache, inputs = {}, {}

    def example_at(position, update_position):
        if not 0 <= position < update_position:
            raise ValueError('guard may only access earlier ended shots')
        if position not in cache:
            sid = ids[position]
            assert sha(io23.path_for('EAST', sid)) == raw_hashes[str(sid)]
            raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
            shot = simple_data.attach_outcome(raw, 'EAST', {})
            signal = transform({k:raw[k] for k in ('values', 'available')}, mapping)
            prefix = f'{position+1:04d}_{sid}'
            path = REFERENCE/'shots'/f'{prefix}.npz'
            assert sha(path) == read(REFERENCE/'shots'/f'{prefix}_seal.json')['sha256']
            with np.load(path, allow_pickle=False) as data:
                scores = data['both'].copy()
            delta, _ = model.memory_delta(signal)
            cache[position] = target_example(model, shot, scores, signal, delta)
            e = cache[position]
            inputs[str(sid)] = dict(rank=position+1, raw_sha256=raw_hashes[str(sid)],
                                    flag=e['flag'], ends=e['ends'])
        return cache[position]

    def state_at(position):
        if position == 360:
            return torch.load(REFERENCE/'models/both.pt', map_location='cuda', weights_only=True)
        path = STATES/'states'/f'{position+1:04d}_{ids[position]}.pt'
        assert sha(path) == state_manifest[position-20]['state_sha256']
        return torch.load(path, map_location='cuda', weights_only=True)['both']

    rows = []
    for count, index in enumerate(selected, 1):
        original = receipts[index]['updates']['both']
        history = ids[index-8:index]
        assert original['history_ids'] == history
        examples = [example_at(i, index) for i in range(index-8, index)]
        losses = {}
        for phase, position in [('before', index), ('after', index+1)]:
            model.restore(state_at(position))
            losses[phase] = {name:guard_losses(model, examples, source, fast, policy=p['target_loss'])
                             for name, fast in [('on', True), ('off', False)]}
            assert losses[phase]['on'] == original['guard_'+phase], (ids[index], phase)
        rules = evaluate_guards(losses['before']['on'], losses['before']['off'],
                                losses['after']['on'], losses['after']['off'])
        assert rules['original_pass']
        row = dict(shot=ids[index], rank=index+1, history_ids=history, losses=losses,
                   has_later_scored_shot=index < 359, rules=rules,
                   original_guard_exact=True, original_accepted=True)
        if ids[index] == 65067:
            prior = read(STATES/'update65067_available_history.json')
            assert losses == prior['losses']
        _durable_json(root/'rows'/f'{index+1:04d}_{ids[index]}.json', row)
        rows.append(row)
        if count % 10 == 0 or count == len(selected):
            print(dict(accepted_updates_audited=count, total=len(selected),
                       cached_history_shots=len(cache), seconds=time.monotonic()-start), flush=True)
    summary = dict(total_original_accepted=len(rows),
        original_pass=sum(r['rules']['original_pass'] for r in rows),
        dual_absolute_pass=sum(r['rules']['dual_absolute_pass'] for r in rows),
        paired_pass=sum(r['rules']['paired_pass'] for r in rows),
        additional_gap_rejections=sum(r['rules']['dual_absolute_pass'] and not r['rules']['paired_pass'] for r in rows),
        violations={kind:{group:sum(group in r['rules'][kind] for r in rows)
                          for group in ('historical_S', 'historical_D', 'source_event')}
                    for kind in ('on_violations', 'off_violations', 'gap_violations')},
        passed_ids={kind:[r['shot'] for r in rows if r['rules'][kind]]
                    for kind in ('dual_absolute_pass', 'paired_pass')})
    _durable_json(root/'inputs.json', inputs)
    _durable_json(root/'rows.json', rows)
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    completed = dict(status='COMPLETED', summary=summary, seconds=time.monotonic()-start,
                     original_guard_losses_exact=True, case65067_exact=True,
                     source_unchanged=True, no_model_training=True, no_new_policy_metrics=True,
                     scope='ORIGINAL_HISTORY_SCREENING_NOT_CAUSAL_POLICY_REPLAY',
                     rows_sha256=sha(root/'rows.json'), independent_confirmation=False)
    _durable_json(root/'completed.json', completed)
    print(completed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
