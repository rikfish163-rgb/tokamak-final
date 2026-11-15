"""Match source two-update episodes without changing the fitted EAST head."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_transition_data_20261110 as data
from . import commissioning_transition_net_20261110 as neural
from . import io23, simple_data
from .commissioning_event_roughness_20261110 import CUE
from .commissioning_intra_20261110 import CAPS, STARTUP, load_frozen
from .commissioning_transfer_20261108 import capacity, features
from .commissioning_transition_east_20261110 import scores
from .commissioning_transition_source_20261110 import shot_coordinates
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, prepare_actual
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .source_anticipation_20261104.static_stream import timing_decomposition


ARMS = ('real', 'reverse')


@torch.no_grad()
def recent_two(net, x, q, y, mask):
    """At each release, replay just the last two valid observations from zero."""
    valid = (q + neural.HORIZON < len(x)) & mask.any(axis=(1, 2))
    q, y, mask = q[valid], y[valid], mask[valid]
    pre, key = net.design(torch.from_numpy(x))
    fixed = pre.tanh().numpy()
    if not len(q):
        return {k: fixed.copy() for k in ARMS}, 0
    a, k = pre[q], key[q]
    m = torch.from_numpy(mask)
    zero = net.initial(len(q))
    releases = q + neural.HORIZON + 1
    ends = np.r_[releases[1:], len(x)]
    views = {}
    for arm, target in zip(ARMS, (y, data.reverse_valid(y, mask)), strict=True):
        target = torch.from_numpy(target)
        first, _ = net.update(a, k, zero, target, m)
        history = torch.cat((zero[:1], first[:-1]))
        second, _ = net.update(a, k, history, target, m)
        h = fixed.copy()
        for j, (s, e) in enumerate(zip(releases, ends, strict=True)):
            h[s:e] = net.project(pre[s:e], key[s:e], second[j].expand(e-s, -1, -1)).numpy()
        views[arm] = h
    return views, len(q)


def run(ref, out):
    started = time.monotonic()
    configure()
    inputs, _, original, _, _ = load_frozen()
    protocol, seal, cal = (read(ref/k) for k in ('protocol.json', 'seal.json', 'calibration.json'))
    source = Path(protocol['source_transition_fit'])
    source_protocol = read(source/'protocol.json')
    if (protocol['order'] != inputs['ids'] or protocol['source_sha256'] != PARENT_SHA256
            or protocol['original'] != original
            or any(sha(source/k) != v for k,v in protocol['source_artifacts_sha256'].items())
            or any(sha(ref/k) != v for k,v in seal['artifacts_sha256'].items())
            or source_protocol['model_sha256'] != sha(Path(neural.__file__))
            or source_protocol['data_sha256'] != sha(Path(data.__file__))):
        raise ValueError('reference must use the same source, input and fitted head')
    thresholds = cal['cutoffs']
    base, _ = load_parent(DEFAULT_PARENT)
    center, scale = (v.detach().numpy().copy() for v in (base.center, base.scale))
    base.cuda().requires_grad_(False)
    frozen_base = state_digest(base)
    if frozen_base != inputs['baseline_protocol']['source_state_sha256']:
        raise ValueError('qualified source changed')
    net = neural.TransitionNet(len(CUE)).eval().requires_grad_(False)
    with np.load(source/'model.npz', allow_pickle=False) as m:
        net.load_state_dict({k:torch.from_numpy(m[k]) for k in m.files})
    frozen_net = state_digest(net)
    with np.load(ref/'readout.npz', allow_pickle=False) as m:
        coef = torch.from_numpy(m['coef']).cuda()
    out.mkdir(parents=True)
    (out/'shots').mkdir()
    write(out/'protocol.json', dict(
        scope='EAST360_TWO_RECENT_VALID_UPDATES_SAME_HEAD_DEVELOPMENT', order=inputs['ids'],
        reference=str(ref), reference_seal_sha256=sha(ref/'seal.json'), producer_sha256=sha(Path(__file__)),
        threshold=thresholds, original=original, source_sha256=PARENT_SHA256, startup=STARTUP,
        arms=ARMS, persistent_updates=0))
    seals, prefix, observed = {}, {}, {}

    def representation(signal):
        z, m, c, cm = shot_coordinates(signal, center, scale)
        q = np.arange(neural.FIRST, len(z)-neural.HORIZON, neural.STRIDE)
        y, mask = data.targets(z, m, q)
        x = data.inputs(z, m, c, cm, np.arange(len(z)))
        rep, n = recent_two(net, x, q, y, mask)
        for value in rep.values():
            value[:neural.FIRST] = 0.
        q = q[mask.any(axis=(1, 2))]
        third = int(q[2]+neural.HORIZON+1) if len(q)>=3 else len(z)
        return rep, n, third

    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('unsealed EAST outcome access')):
        for i, sid in enumerate(inputs['ids'], 1):
            name = f'{i:04d}_{sid}.npz'
            file = ref/'shots'/name
            if sha(file) != seal['curve_sha256'][name]:
                raise ValueError('reference curve changed')
            with np.load(file, allow_pickle=False) as f:
                times, original_curves = f['times'], {k:f[k] for k in ARMS}
            if i <= STARTUP:
                curves = original_curves
            else:
                if sha(io23.path_for('EAST', sid)) != inputs['hashes'][str(sid)]:
                    raise ValueError(f'input changed: {sid}')
                raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                np.testing.assert_array_equal(times, raw['times'])
                shot = prepare_actual(raw, i, inputs['mapping'])
                h, _ = features(base, shot, 'cuda')
                v, observed[str(sid)], first_difference = representation(shot)
                curves = scores(h, v, coef, ARMS)
                for k, y in curves.items():
                    n = first_difference
                    np.testing.assert_allclose(y[:n], original_curves[k][:n], atol=2e-6, rtol=0)
                    for t in thresholds.values():
                        np.testing.assert_array_equal(y[:n]>=t, original_curves[k][:n]>=t)
                if i in (21, 120, 360):
                    sub = {k:shot[k][:243] for k in ('values','available')}
                    hs, _ = features(base, sub, 'cuda')
                    ps = scores(hs, representation(sub)[0], coef, ARMS)
                    prefix[str(sid)] = {}
                    for k, y in ps.items():
                        z = curves[k][:len(y)]
                        np.testing.assert_allclose(y, z, atol=2e-6, rtol=0)
                        for t in thresholds.values():
                            np.testing.assert_array_equal(y>=t,z>=t)
                        prefix[str(sid)][k] = float(np.max(abs(y-z)))
            path = out/'shots'/name
            with path.open('xb') as f:
                np.savez_compressed(f, times=times, **curves)
            seals[name] = sha(path)
            if i % 60 == 0:
                print(dict(stage='sealed', shots=i, seconds=time.monotonic()-started), flush=True)
    if frozen_net != state_digest(net) or frozen_base != state_digest(base):
        raise ValueError('persistent state changed')
    write(out/'seal.json', dict(curve_sha256=seals, prefix_witnesses=prefix))
    per = {c:{k:[] for k in ARMS} for c in CAPS}
    maxima, labels = {k:[] for k in ARMS}, []
    for i, sid in enumerate(inputs['ids'], 1):
        path = out/'shots'/f'{i:04d}_{sid}.npz'
        if sha(path) != seals[path.name]:
            raise ValueError('unsealed curve')
        d, down = io23.read_outcome('EAST', sid)
        labels.append(d)
        with np.load(path, allow_pickle=False) as f:
            t = f['times']
            mask = np.round((down-t)*1000, 9)>30 if d else np.ones(len(t), bool)
            for k in ARMS:
                y = f[k]
                maxima[k].append(float(y[mask].max(initial=-np.inf)))
                for c, g in per.items():
                    r = score_curve(t, y, d, down, original if i<=STARTUP else thresholds[c])
                    g[k].append(dict(shot=sid, disruptive=d, down_s=down, **r))
    prior = read(ref/'per_shot.json')
    totals = read(ref/'result.json')['dynamics']['observed_blocks']
    if sum(observed.values()) != totals:
        raise ValueError('feedback budget changed')
    metrics = {c:{k:timing_decomposition(v) for k,v in g.items()} for c,g in per.items()}
    vs_fixed = {c:{k:flips(prior[c]['fixed'],g[k]) for k in ARMS} for c,g in per.items()}
    vs_long = {c:{k:flips(prior[c][k],g[k]) for k in ARMS} for c,g in per.items()}
    write(out/'per_shot.json', per)
    write(out/'result.json', dict(
        status='COMPLETED_RECENT_TWO_EAST360_DEVELOPMENT', metrics=metrics,
        paired_vs_fixed=vs_fixed, paired_vs_cumulative=vs_long,
        paired_real_vs_reverse={c:flips(g['reverse'],g['real']) for c,g in per.items()},
        observed_blocks=totals, posthoc_capacity_not_deployable={
            k:capacity(v, labels, original, startup=STARTUP) for k,v in maxima.items()},
        seconds=time.monotonic()-started))
    print({c:{k:(v['tp'],v['fp']) for k,v in g.items()} for c,g in metrics.items()}, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    run(args.reference.resolve(), args.output.resolve())
