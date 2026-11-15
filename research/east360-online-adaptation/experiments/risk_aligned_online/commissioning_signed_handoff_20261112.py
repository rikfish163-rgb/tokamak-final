"""History-taught signed diagnostics with fixed, causal EAST handoff."""

import argparse
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_group_token_20261112 as group
from . import io23, simple_data
from .history_gradient_metric_20261102.run import read, sha
from .mask_separated_intra_20261103.run import state_digest


lag, layer = group.lag, group.layer
FIT_STEPS, LR, SCALE, REG, BOX = 160, .05, .10, .25, .75


def predicted_changes(gram, slope, radius, weights):
    w = weights.flatten()
    squared = torch.einsum('i,...ij,j->...', w, gram, w)
    return radius*(slope @ w)/squared.clamp_min(group.EPS**2).sqrt()


def fit_weights(gram, slope, radius):
    """Choose eight bounded weights; optimize risk movement at equal hidden-token dose."""
    param = torch.nn.Parameter(gram.new_zeros(2, 4))
    opt = torch.optim.Adam([param], lr=LR)
    for _ in range(FIT_STEPS):
        opt.zero_grad(set_to_none=True)
        change = predicted_changes(gram, slope, radius, 1.+BOX*param.tanh())
        pos, safe = change.unbind(1)
        loss = (F.softplus(-pos/SCALE)+F.softplus(safe/SCALE)).mean()
        (loss+REG*param.square().mean()).backward()
        opt.step()
    return 1.+BOX*param.detach().tanh()


def historical_case(spec, role, args, decoder, path, lock, advanced, allowed):
    sid, origin, wanted, positive = (spec[k] for k in ('shot', 'origin', 'action', 'positive'))
    source = role != 'cold'
    device = 'J-TEXT' if source else 'EAST'
    if sid not in allowed or wanted < origin+lag.H+1:
        raise ValueError('invalid historical identity or timing')
    if not source and sha(io23.path_for(device, sid)) != lock['hashes'][str(sid)]:
        raise ValueError('cold-shot signal changed')
    raw = simple_data.read_shot(device, sid, rate_contract='strict' if source else 'nominal')
    truth = simple_data.attach_outcome(raw, device, advanced)
    if truth['is_disrupt'] != positive or not truth['valid'][wanted]:
        raise ValueError('history disagrees with HDF5')
    base = args[0]
    t = max(wanted, origin+lag.H+1+base.cfg.patch)
    record = dict(spec, action=t)
    if (t >= len(raw['times']) or not truth['valid'][t]
            or (positive and (not truth['y'][t] or round((truth['down']-raw['times'][t])*1000., 9)
                              <= (10. if source else 30.)))):
        return None, dict(record, exclusion='action would miss timely region after one patch')
    sig = raw if source else lag.prepare_actual(raw, lag.STARTUP+1, lock['mapping'])
    q, _, _, beta, prior, phase, correction = group.parent_track(sig, args, decoder, path, source=source)
    s = int(lag.delayed_slots(q, np.array([t]))[0][0])
    if s == 0 or q[s-1] != origin:
        return None, dict(record, exclusion='recorded origin not the latest legal feedback')
    eligible, fields, waveform = group.feedback_waveform(sig, q, args[2], beta.device)
    i = np.array([s-1])
    if not eligible[i][0]:
        return None, dict(record, exclusion='insufficient observed reference')
    issued, current, state, wave, center, spread = group.signed_window(
        base, sig, q, i, phase, correction, prior, fields, decoder)
    rp, rf, sp, sf = group.contrast_forces(wave, waveform[i], center, spread)
    weights = torch.eye(8, device=wave.device).reshape(8, 2, 4)
    spec = [(rp, rf), *((sp, group.shape_force(sf, w)) for w in weights),
            *((sp, group.shape_force(sf.flip(1), w)) for w in weights)]
    updates = group.signed_directions(issued, current, state, spec)
    x, m = lag._windows(base, sig, np.array([t]))
    transform = layer.patcher(phase[s:s+1], correction[s:s+1])
    basis, dose = [], []
    def capture(raw):
        updated = transform(raw)
        basis.append(group.group_edges(updated, base.embedding.weight))
        dose.append(F.linear(updated-raw, base.embedding.weight).norm())
        return updated
    sites, z = layer.hidden_pass(base, x, m, patcher=capture, differentiable=True, beta=beta)
    activation, = torch.autograd.grad(z.sum(), sites[0])
    with torch.no_grad():
        g = activation[:, :-1].flatten()
        moves = group.exchange(updates, basis[0]).flatten(1)
        initial = torch.stack((moves[0], moves[1:9].sum(0), moves[9:].sum(0)))
        if dose[0] <= group.EPS or (initial.norm(dim=1) <= group.EPS).any():
            return None, dict(record, exclusion='no common internal write')
        directions = moves[1:9]
        gram, slope = directions@directions.T, directions@g
    return (gram, slope, dose[0], updates, x, m, transform, beta), record


@torch.no_grad()
def finite_comparison(base, item, weights):
    _, _, _, updates, x, mask, transform, beta = item
    w = weights.flatten()[:, None, None, None]
    normal, reversed_ = updates[1:9], updates[9:]
    writes = torch.stack((updates[0], normal.sum(0), reversed_.sum(0),
                          (w*normal).sum(0), (w*reversed_).sum(0)))
    view, dose, error = group.forward_hidden(base, x, mask, transform, writes=writes)
    if float(error) > 1e-5:
        raise ValueError('history comparison did not match write norm')
    return float(dose.item()), ((view[1:, 0]-view[0, 0])@beta).cpu().tolist()


def teach():
    lag.configure()
    lock = lag.load_inputs()
    p, _, args, decoder, path = group.load_reference(lock)
    base = state_digest(args[0])
    archive = group.REFERENCE/'historical.json'
    if sha(archive) != p['history_sha256']:
        raise ValueError('historical reference changed')
    manifests = read(archive)['orientation']
    allowed = {**{k: set(v) for k, v in io23.source_roles().items()}, 'cold': set(lock['ids'][:lag.STARTUP])}
    advanced = io23.advanced_times()
    usable, receipts = {}, {}
    for role in ('train', 'val', 'cold'):
        manifest = manifests[role]['manifest']
        examples = [historical_case(r, role, args, decoder, path, lock, advanced, allowed[role])
                    for r in manifest]
        records, valid = [], []
        for a, b in zip(examples[::2], examples[1::2], strict=True):
            if not a[1]['positive'] or b[1]['positive']:
                raise ValueError('expected positive-negative paired history')
            records.append((a[1], b[1]))
            if a[0] is not None and b[0] is not None:
                valid.append((a[0], b[0]))
        usable[role] = valid
        receipts[role] = dict(registered_pairs=len(records), comparable_pairs=len(valid), pairs=records)
        print(dict(stage='signed_history', role=role, compared=len(valid), registered=len(records)), flush=True)
    if not usable['train']:
        raise ValueError('no common source TRAIN pairs for signed feedback teaching')
    gram, slope, radius = (torch.stack([torch.stack((a[k], b[k])) for a, b in usable['train']])
                           for k in range(3))
    weights = fit_weights(gram, slope, radius)
    if not torch.isfinite(weights).all() or base != state_digest(args[0]):
        raise ValueError('invalid signed weights or changed frozen source')
    for role, episodes in usable.items():
        rows = [(finite_comparison(args[0], a, weights), finite_comparison(args[0], b, weights))
                for a, b in episodes]
        present = [(p, s) for (d, p), (e, s) in rows if min(d, e) > group.EPS]
        scores = {}
        for k, name in enumerate(group.SIGNED):
            pos = [a[k] for a, _ in present]
            safe = [b[k] for _, b in present]
            scores[name] = dict(both_correct=sum(a>0.>b for a, b in zip(pos, safe, strict=True)),
                positive_higher=sum(a>b for a, b in zip(pos, safe, strict=True)),
                positive_mean=float(np.mean(pos)) if pos else None,
                safe_mean=float(np.mean(safe)) if safe else None)
        receipts[role].update(finite_pairs=len(present), scores=scores, finite_changes=rows)
    record = dict(scope='JTEXT_TRAIN_ONLY_SIGNED_FORCE_TEACHER', source_sha256=p['source_sha256'],
        history_sha256=sha(archive), fit_code_sha256=sha(Path(__file__)),
        weights=weights.cpu().tolist(), rows=('persistent_error', 'zero_mean_time_variation'),
        groups=[[io23.RAW23[c] for c in g] for g in group.TRUSTED],
        fit=dict(steps=FIT_STEPS, lr=LR, slope_scale=SCALE, penalty=REG, box=BOX), evidence=receipts)
    print(dict(stage='signed_teacher_frozen', weights=record['weights'],
               held={k: (v['finite_pairs'], v['scores']) for k, v in receipts.items()}), flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument('--components-from', type=Path, help='reuse the frozen teacher for a component ablation')
    comparison.add_argument('--innovations-from', type=Path, help='reuse the frozen teacher for a history ablation')
    a = parser.parse_args()
    dest = a.out.resolve()
    from_path = a.components_from or a.innovations_from
    if from_path is None:
        teacher, executed = teach(), Path(__file__)
    else:
        parent = from_path.resolve()
        p = read(parent/'protocol.json')
        teacher, executed = read(parent/'teaching.json'), parent/'executed_teacher.py'
        if (p['scope'] != 'EAST360_SIGNED_HANDOFF_DEVELOPMENT'
                or p['teaching_sha256'] != sha(parent/'teaching.json')
                or teacher['fit_code_sha256'] != sha(executed)
                or teacher['source_sha256'] != p['source_sha256']):
            raise ValueError('invalid signed teacher provenance')
        # The replay reads the frozen training artifact, not the parent’s scored outcomes.
    group.run(dest, teaching=teacher, split=a.components_from is not None, innovation=a.innovations_from is not None)
    copyfile(executed, dest/'executed_teacher.py')
    group.score(dest)
