"""Apply earlier saved states to fixed later-case inputs, strictly post hoc.

Later inputs were NOT available when those states were trained. This analysis
does not retroactively veto updates or alter an online policy.
"""
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from .. import simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import load_context_memory
from ..delayed_recurrent_memory_20261025.learning import batch
from ..delayed_recurrent_memory_20261025.model import RecurrentAdapter
from ..east_context_operator_20261031.run import configure, read
from ..lifecycle_stream_20261025.run import build_adapter
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .run import REFERENCE, CASES


def local_geometry(model, h, valid, delta, state, position, case):
    """Derivative with respect to the post-slow last token, not a trainable update."""
    gradients, result = {}, {}
    actions = {name:F.linear(delta, state[name]['fast_output'])*valid[:, -2, None]
               for name in ('cold', 'both')}
    zero = torch.zeros_like(delta)
    for name in ('cold', 'both'):
        before = h.clone()
        before[:, -2] = h[:, -2]+F.linear(h[:, -2], state[name]['slow'])*valid[:, -2, None]
        with torch.enable_grad():
            leaf = before.detach().requires_grad_(True)
            z = RecurrentAdapter.risk(model, leaf, valid, zero)
            gradient, = torch.autograd.grad(z[position], leaf)
        expected_off = 'cold_off' if name == 'cold' else 'both_off'
        assert float(z[position]) == case['at_same_endpoint'][expected_off]['logit']
        g = gradient[position, -2].detach()
        gradients[name] = g
        variance = before[position, -2].var(unbiased=False)
        den = (variance+model.base.encoder.layers[1].norm1.eps).sqrt()
        item = dict(baseline_logit=float(z[position]), gradient_norm=float(g.norm()),
                    post_slow_token_std=float(variance.sqrt()), layernorm1_denominator=float(den),
                    readouts={})
        for output_name, action in actions.items():
            samples = []
            for alpha in (0., .25, .5, .75, 1.):
                changed = before.clone()
                changed[:, -2] = changed[:, -2]+alpha*action
                prediction = RecurrentAdapter.risk(model, changed, valid, zero)
                samples.append(dict(alpha=alpha, logit=float(prediction[position]),
                                    probability=float(prediction.sigmoid()[position])))
            norm = float(action[position].norm())
            linear = float(g@action[position])
            true = samples[-1]['logit']-item['baseline_logit']
            arm = {('cold','cold'):'cold_on', ('cold','both'):'cold_slow_both_readout',
                   ('both','cold'):'both_slow_cold_readout', ('both','both'):'both_on'}[name,output_name]
            assert samples[-1]['logit'] == case['at_same_endpoint'][arm]['logit']
            item['readouts'][output_name] = dict(action_norm=norm, first_order_logit_change=linear,
                                                actual_logit_change=true, interpolation=samples)
        result[name] = item
    den = gradients['cold'].norm()*gradients['both'].norm()
    result['gradient_direction_cosine'] = float(gradients['cold']@gradients['both']/den) if den > 0 else None
    return result


@torch.no_grad()
def main(root):
    output = root/'case_state_transitions.json'
    if output.exists():
        raise FileExistsError(output)
    assert read(root/'completed.json')['status'] == 'COMPLETED'
    configure()
    source_p = read(REFERENCE/'protocol.json')
    base, _ = load_parent(DEFAULT_PARENT)
    model = build_adapter(base.cuda(), readout='full', feedback='progressive', context=True)
    load_context_memory(model, Path(source_p['source_memory']['path']))
    mapping = read(root/'input_map.json')
    threshold = source_p['threshold']
    order = read(REFERENCE/'order.json')
    receipts = read(REFERENCE/'receipts.json')
    states = read(root/'state_evolution.json')
    cases = {row['shot']:row for row in read(root/'cases.json')}
    version_indices = [0]+[j for j in range(1, len(states))
                           if receipts[states[j]['index']-2]['updates']['both']['accepted']]
    all_rows = []
    for sid in CASES:
        case = cases[sid]
        end, rank = case['reference_endpoint'], case['index']
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        signal = transform({k:raw[k] for k in ('values', 'available')}, mapping)
        delta, _ = model.memory_delta(signal)
        lo, hi = end//32*32, min(end//32*32+32, len(signal['values']))
        x, available = batch(signal, range(lo, hi), 'cuda')
        h, valid = model.tokens(x, available)
        action = delta[lo:hi]
        previous = None
        trajectory = []
        for j in version_indices:
            row = states[j]
            if row['index'] > rank:
                continue
            path = root/'states'/f"{row['index']:04d}_{row['shot']}.pt"
            assert sha(path) == row['state_sha256']
            state = torch.load(path, map_location='cuda', weights_only=True)['both']
            combinations = dict(full=(state, state))
            if previous is not None:
                combinations.update(slow_only=(state, previous), readout_only=(previous, state),
                                    previous=(previous, previous))
            scores = {}
            for name, (s, o) in combinations.items():
                model.restore(dict(slow=s['slow'], fast_output=o['fast_output']))
                z = model.risk(h, valid, action)
                scores[name] = dict(logit=float(z[end-lo]), probability=float(z.sigmoid()[end-lo]))
            transition = dict(state_before_rank=row['index'], state_before_shot=row['shot'], scores=scores)
            if previous is not None:
                old = scores['previous']['probability']
                new = scores['full']['probability']
                original = receipts[row['index']-2]
                update = original['updates']['both']
                assert update['accepted']
                assert original['shot'] == order[row['index']-2]['shot']
                transition.update(revealed_update_rank=row['index']-1,
                    revealed_update_shot=original['shot'],
                    upward_threshold_crossing=old < threshold <= new,
                    downward_threshold_crossing=new < threshold <= old,
                    full_probability_change=new-old,
                    guard_before=update['guard_before'], guard_after=update['guard_after'],
                    parameter_delta=update['retained_delta'])
            trajectory.append(transition)
            previous = state
        assert trajectory and previous is not None
        assert trajectory[-1]['scores']['full']['probability'] == case['at_same_endpoint']['both_on']['probability']
        state_path = root/'states'/f'{rank:04d}_{sid}.pt'
        actual_state = torch.load(state_path, map_location='cuda', weights_only=True)
        geometry = local_geometry(model, h, valid, action, actual_state, end-lo, case)
        up = [r for r in trajectory if r.get('upward_threshold_crossing')]
        down = [r for r in trajectory if r.get('downward_threshold_crossing')]
        changes = [r for r in trajectory if 'full_probability_change' in r]
        largest = max(changes, key=lambda r:r['full_probability_change']) if changes else None
        all_rows.append(dict(shot=sid, actual_rank=rank, fixed_endpoint=end,
            warning='same later prefix applied posthoc; not available at historical update time',
            evaluated_versions=len(trajectory), trajectory=trajectory,
            upward_update_shots=[r['revealed_update_shot'] for r in up],
            downward_update_shots=[r['revealed_update_shot'] for r in down],
            largest_increase_update=largest, actual_case_local_geometry=geometry))
    result = dict(scope='POSTHOC_FIXED_LATER_INPUT_WEIGHT_VERSION_DIAGNOSTIC',
                  no_policy_change=True, source_checkpoint_sha256=sha(DEFAULT_PARENT),
                  completed_sha256=sha(root/'completed.json'), script_sha256=sha(Path(__file__)),
                  threshold=threshold, cases=all_rows, independent_confirmation=False)
    _durable_json(output, result)
    print([dict(shot=r['shot'], versions=r['evaluated_versions'], up=r['upward_update_shots'],
                down=r['downward_update_shots'],
                largest=None if r['largest_increase_update'] is None else
                {k:r['largest_increase_update'][k] for k in ('revealed_update_shot', 'full_probability_change', 'scores')})
           for r in all_rows])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
