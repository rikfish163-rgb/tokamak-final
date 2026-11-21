"""Paired EAST replay of the source-trained neural transition update."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_transition_data_20261110 as data
from . import commissioning_transition_net_20261110 as neural
from . import commissioning_transition_source_20261110 as source
from . import io23, simple_data
from .commissioning_calibration_20261108 import held_safe_cutoffs, prefix_mapping
from .commissioning_event_roughness_20261110 import CUE
from .commissioning_intra_20261110 import CALIBRATION, CAPS, PARENT_ARM, PARENT_RUN, STARTUP, load_frozen
from .commissioning_latent_rate_20261110 import read_parent_curve
from .commissioning_transfer_20261108 import alarm_loss, capacity, combine, features, make_rows, shot_weights
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, prepare_actual
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .source_anticipation_20261104.static_stream import timing_decomposition


NAMES = ('parent', *neural.MODES)
FIT_MODES = ('fixed', 'real')
STEPS = 256
FIRST_EFFECT = neural.FIRST + neural.HORIZON + 1


def encode(base, net, signal, center, scale):
    raw, _ = features(base, signal, 'cuda')
    z, m, context, available = source.shot_coordinates(signal, center, scale)
    q = np.arange(neural.FIRST, len(z)-neural.HORIZON, neural.STRIDE)
    y, observed = data.targets(z, m, q)
    x = data.inputs(z, m, context, available, np.arange(len(z)))
    reps, audit = neural.replay(net, x, q, y, observed)
    for v in reps.values():
        v[:neural.FIRST] = 0.
    return raw, reps, audit


def fit(base, net, raws, outcomes, ids, mapping, center, scale, original, *, encode_fn=encode):
    by_mode = {k: [] for k in FIT_MODES}
    for g, sid in enumerate(ids):
        raw, views, _ = encode_fn(base, net, prepare_actual(raws[sid], STARTUP+1, mapping), center, scale)
        for k in FIT_MODES:
            x = np.concatenate((raw, views[k]), axis=1)
            by_mode[k].append(make_rows(x, raws[sid]['times'], *outcomes[sid], 125., 30., g))
    full, adapted = map(combine, by_mode.values())
    target = torch.from_numpy(full.y.astype(np.float32)).cuda()
    weights = torch.from_numpy(shot_weights(full.shot, full.y).astype(np.float32)).cuda()
    x = torch.from_numpy(np.stack((full.x, adapted.x))).cuda()
    origin = torch.cat((base.risk_head.weight.flatten(), x.new_zeros(x.shape[-1]-base.cfg.dim), base.risk_head.bias))
    coef = torch.nn.Parameter(origin.clone())
    opt = torch.optim.Adam([coef], lr=.02)
    boundary = float(np.log(original/(1.-original)))

    def groups(eligible):
        return [torch.from_numpy(np.flatnonzero(eligible & (full.shot == s))).cuda()
                for s in np.unique(full.shot[eligible])]

    pos, neg = groups(full.timely), groups(full.safe)
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        logit = x @ coef[:-1] + coef[-1]
        point = (F.binary_cross_entropy_with_logits(logit, target.expand_as(logit), reduction='none') @ weights).mean()
        warning = torch.stack([alarm_loss(z, pos, neg, boundary) for z in logit]).mean()
        (point + warning + .05*(coef-origin).square().sum()).backward()
        opt.step()
    if not torch.isfinite(coef).all():
        raise FloatingPointError('nonfinite EAST risk coefficients')
    return coef.detach(), dict(point=float(point.detach()), warning=float(warning.detach()), steps=STEPS)


@torch.no_grad()
def scores(raw, views, coef, modes=neural.MODES):
    return {k: (torch.from_numpy(np.concatenate((raw, views[k]), axis=1)).cuda() @ coef[:-1]
                + coef[-1]).sigmoid().cpu().numpy() for k in modes}


def calibrate(base, net, raws, outcomes, source_stats, center, scale, original, started, *, encode_fn=encode):
    held = []
    for sid, (d, _) in outcomes.items():
        if d:
            continue
        ids = [s for s in raws if s != sid]
        mapping = prefix_mapping(source_stats, [raws[s] for s in ids])
        coef, _ = fit(base, net, raws, outcomes, ids, mapping, center, scale, original, encode_fn=encode_fn)
        raw, views, _ = encode_fn(base, net, prepare_actual(raws[sid], STARTUP+1, mapping), center, scale)
        peak = float(scores(raw, views, coef, ('fixed',))['fixed'].max())
        held.append(dict(held_safe=sid, fit_ids=ids, fixed_max=peak))
        if len(held) % 3 == 0:
            print(dict(stage='calibration', held=len(held), seconds=time.monotonic()-started), flush=True)
    return dict(folds=held, cutoffs=held_safe_cutoffs([f['fixed_max'] for f in held]))


def run(source_dir, out):
    started = time.monotonic()
    configure()
    inputs, parent_coef, original, parent_cutoffs, bound = load_frozen()
    base, ckpt = load_parent(DEFAULT_PARENT)
    center, scale = (v.detach().numpy().copy() for v in (base.center, base.scale))
    base.cuda().requires_grad_(False)
    frozen = state_digest(base)
    if ckpt['threshold'] != original or frozen != inputs['baseline_protocol']['source_state_sha256']:
        raise ValueError('qualified source changed')
    protocol, result = (read(source_dir/k) for k in ('protocol.json', 'result.json'))
    assets = {k:sha(source_dir/k) for k in ('protocol.json', 'result.json', 'model.npz')}
    if (protocol['parent_sha256'] != PARENT_SHA256 or result['model_sha256'] != assets['model.npz']
            or protocol['model_sha256'] != sha(Path(neural.__file__))
            or protocol['data_sha256'] != sha(Path(data.__file__))):
        raise ValueError('source prior and running code must match their fit')
    net = neural.TransitionNet(len(CUE))
    with np.load(source_dir/'model.npz', allow_pickle=False) as f:
        net.load_state_dict({k: torch.from_numpy(f[k]) for k in f.files})
    net.eval().requires_grad_(False)
    net_hash = state_digest(net)
    ids = inputs['ids']
    parent_seals = read(PARENT_RUN/'result.json')['curve_sha256']
    out.mkdir(parents=True)
    (out/'shots').mkdir()
    write(out/'protocol.json', dict(
        scope='EAST360_COLD20_NEURAL_TRANSITION_INTRA_ONLY_DEVELOPMENT', order=ids, startup=STARTUP,
        source_sha256=PARENT_SHA256, **bound, original=original, parent_thresholds=parent_cutoffs,
        arms=NAMES, cold_fit_views=FIT_MODES, source_transition_fit=str(source_dir),
        source_artifacts_sha256=assets, producer_sha256=sha(Path(__file__)),
        first_update_effect=FIRST_EFFECT, horizon=neural.HORIZON, stride=neural.STRIDE, persistent_updates=0))

    def load_raw(sid, t):
        if sha(io23.path_for('EAST', sid)) != inputs['hashes'][str(sid)]:
            raise ValueError(f'EAST input changed: {sid}')
        s = simple_data.read_shot('EAST', sid, rate_contract='nominal')
        np.testing.assert_array_equal(t, s['times'])
        return s

    raws, outcomes = {}, {}
    for i, sid in enumerate(ids[:STARTUP], 1):
        t, _ = read_parent_curve(f'{i:04d}_{sid}.npz', parent_seals)
        raws[sid] = load_raw(sid, t)
        outcomes[sid] = io23.read_outcome('EAST', sid)
    if sum(d for d, _ in outcomes.values()) != 8:
        raise ValueError('expected 8 disruptive outcomes in the ended cold prefix')
    cal = calibrate(base, net, raws, outcomes, inputs['mapping']['source_statistics'], center, scale, original, started)
    write(out/'calibration.json', cal)
    cutoffs = cal['cutoffs']
    coef, fit_audit = fit(base, net, raws, outcomes, ids[:STARTUP], inputs['mapping'], center, scale, original)
    with (out/'readout.npz').open('xb') as f:
        np.savez(f, coef=coef.cpu().numpy())
    binding = {k:sha(out/k) for k in ('calibration.json', 'readout.npz')}
    print(dict(stage='fit20_complete', seconds=time.monotonic()-started), flush=True)
    seals, audit, prefix = {}, {}, {}
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('unsealed EAST outcome access')):
        for i, sid in enumerate(ids, 1):
            name = f'{i:04d}_{sid}.npz'
            t, parent = read_parent_curve(name, parent_seals)
            if i <= STARTUP:
                curves = dict.fromkeys(NAMES, parent)
            else:
                s = prepare_actual(load_raw(sid, t), i, inputs['mapping'])
                raw, h, audit[str(sid)] = encode(base, net, s, center, scale)
                same_parent = (torch.from_numpy(raw).cuda() @ parent_coef[:-1] + parent_coef[-1]).sigmoid().cpu().numpy()
                np.testing.assert_array_equal(parent, same_parent)
                curves = dict(parent=parent, **scores(raw, h, coef))
                for k in neural.MODES[1:]:
                    np.testing.assert_array_equal(curves[k][:FIRST_EFFECT], curves['fixed'][:FIRST_EFFECT])
                if i in (21, 120, 360):
                    short = {k:s[k][:243] for k in ('values', 'available')}
                    x, h, _ = encode(base, net, short, center, scale)
                    prefix[str(sid)] = {}
                    for k, v in scores(x, h, coef).items():
                        full = curves[k][:len(v)]
                        np.testing.assert_allclose(v, full, rtol=0., atol=1e-5)
                        for cutoff in cutoffs.values():
                            np.testing.assert_array_equal(v >= cutoff, full >= cutoff)
                        prefix[str(sid)][k] = float(np.max(abs(v-full)))
            path = out/'shots'/name
            with path.open('xb') as f:
                np.savez_compressed(f, times=t, **curves)
            seals[name] = sha(path)
            if i % 60 == 0:
                print(dict(stage='sealed', shots=i, seconds=time.monotonic()-started), flush=True)
    if (state_digest(base) != frozen or state_digest(net) != net_hash
            or any(sha(out/k) != v for k,v in binding.items())):
        raise ValueError('a persistent state changed while replaying the suffix')
    write(out/'seal.json', dict(curve_sha256=seals, artifacts_sha256=binding, prefix_witnesses=prefix))
    write(out/'transition_by_shot.json', audit)
    per = {c:{k:[] for k in NAMES} for c in CAPS}
    maxima, labels = {k:[] for k in NAMES}, []
    for i, sid in enumerate(ids, 1):
        path = out/'shots'/f'{i:04d}_{sid}.npz'
        if sha(path) != seals[path.name]:
            raise ValueError('sealed prediction changed')
        d, down = io23.read_outcome('EAST', sid)
        labels.append(d)
        with np.load(path, allow_pickle=False) as f:
            t = f['times']
            eligible = np.round((down-t)*1000, 9)>30 if d else np.ones(len(t), bool)
            for k in NAMES:
                p = f[k]
                maxima[k].append(float(p[eligible].max(initial=-np.inf)))
                for cap, rows in per.items():
                    threshold = original if i <= STARTUP else (parent_cutoffs if k=='parent' else cutoffs)[cap]
                    rows[k].append(dict(shot=sid, disruptive=d, down_s=down, **score_curve(t, p, d, down, threshold)))
    history = read(CALIBRATION/'per_shot.json')
    if any(g['parent'] != history[c][PARENT_ARM] for c,g in per.items()):
        raise ValueError('archived parent score changed')
    metrics = {c:{k:timing_decomposition(v) for k,v in g.items()} for c,g in per.items()}
    paired = {c:{k:flips(g['fixed'],g[k]) for k in neural.MODES[1:]} for c,g in per.items()}
    direct = {c:flips(g['reverse'],g['real']) for c,g in per.items()}
    tallies = {k:sum(v[k] for v in audit.values()) for k in
               ('observed_blocks','transition_labels','reversed_labels_different','reversed_blocks_different')}
    nll = {k:sum(v['nll_sum'][k] for v in audit.values()) / tallies['transition_labels'] for k in neural.MODES}
    tallies['nonzero_steps'] = {k:sum(v['nonzero_steps'][k] for v in audit.values()) for k in neural.MODES[1:]}
    write(out/'per_shot.json', per)
    write(out/'result.json', dict(status='COMPLETED_NEURAL_TRANSITION_EAST360_DEVELOPMENT',
        metrics=metrics, paired_vs_fixed=paired, paired_real_vs_reverse=direct, fit=fit_audit,
        dynamics=tallies, nll_on_actual_transition_labels=nll,
        posthoc_capacity_not_deployable={k:capacity(v, labels, original, startup=STARTUP) for k,v in maxima.items()},
        seconds=time.monotonic()-started))
    print({c:{k:(v['tp'],v['fp']) for k,v in g.items()} for c,g in metrics.items()}, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', required=True, type=Path)
    cli.add_argument('--output', required=True, type=Path)
    args = cli.parse_args()
    run(args.source.resolve(), args.output.resolve())
