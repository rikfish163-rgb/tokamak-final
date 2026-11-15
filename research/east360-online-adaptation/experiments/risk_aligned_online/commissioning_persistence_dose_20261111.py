"""Fixed-fit hidden-state dose diagnostic; no model or threshold selection."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_persistence_20261111 as persistent
from . import commissioning_persistence_learning_20261111 as train
from . import io23, simple_data
from .commissioning_intra_20261110 import STARTUP
from .commissioning_lag_state_20261110 import LagState
from .commissioning_transfer_20261108 import features
from .commissioning_transition_east_20261110 import scores
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, load_inputs, prepare_actual
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .source_anticipation_20261104.static_stream import timing_decomposition


DOSES = dict(inherited=0., sustained=1., real2=2., opposite2=-2., real4=4., opposite4=-4.)
ADDITIONAL = dict(current4='current_equal_norm', spaced4='spaced', lagged4='lagged')
ARMS = (*DOSES, *ADDITIONAL)


def dose_views(pre, update):
    return {k: (pre+scale*update).tanh() if scale else pre.tanh() for k, scale in DOSES.items()}


@torch.no_grad()
def encode(base, net, signal, params, old, quiet, mapper, coef):
    x, _ = features(base, signal, 'cuda')
    tr, pre, ready = train.observed_track(net, signal, np.arange(len(x)), params, old)
    state, _ = persistent.replay(*ready, quiet)
    k = torch.from_numpy(tr['key'])
    d = torch.einsum('...k,...f,kfw->...w', k, torch.from_numpy(state['sustained']), mapper)
    views = dose_views(pre, d)
    for name, variant in ADDITIONAL.items():
        views[name] = train.view(pre, k, 4.*torch.from_numpy(state[variant]), mapper)
    return scores(x, {k: v.numpy() for k, v in views.items()}, coef, ARMS)


def run(experiment, out):
    configure()
    start = time.monotonic()
    lock = load_inputs()
    ids = lock['ids']
    p, s = (read(experiment/k) for k in ('protocol.json', 'seal.json'))
    if (p['scope'] != 'EAST360_PERSISTENT_SIGNED_DEVIATION_FIXED_HEAD_DEVELOPMENT'
            or p['order'] != ids or p['startup'] != STARTUP or p['source_sha256'] != PARENT_SHA256
            or len(s['curve_sha256']) != 360 or any(sha(experiment/k) != v for k, v in s['fitted'].items())
            or any(sha(Path(__file__).parent/k) != v for k, v in p['code_sha256'].items())):
        raise ValueError('sealed, matching persistence run required')
    previous = Path(p['previous'])
    if any(sha(previous/k) != v for k, v in p['previous_assets'].items()):
        raise ValueError('prior feedback map changed')
    prior = read(previous/'protocol.json')
    ref = Path(prior['reference'])
    if any(sha(ref/k) != v for k, v in prior['reference_assets'].items()):
        raise ValueError('fixed warning head or calibration changed')
    source = Path(read(ref/'protocol.json')['source_prior'])
    if any(sha(source/k) != v for k, v in prior['source_assets'].items()):
        raise ValueError('source representation changed')
    archive = Path(prior['forecast_reference'])/'fit.npz'
    if sha(archive) != prior['forecast_source_fit_sha256']:
        raise ValueError('source forecast changed')
    with np.load(archive, allow_pickle=False) as f:
        decoder, noise = (f['decoder_weight'], f['decoder_bias']), f['scale']
    with np.load(previous/'mapper.npz', allow_pickle=False) as f:
        old, context_scale = torch.from_numpy(f['mapper']), f['context_scale']
    with np.load(experiment/'state.npz', allow_pickle=False) as f:
        mapper, quiet = torch.from_numpy(f['mapper']), (f['location'], f['scale'])
    info = read(source/'result.json')
    net = LagState(info['inputs'], info['edges']).eval().requires_grad_(False)
    with np.load(source/'model.npz', allow_pickle=False) as f:
        net.load_state_dict({k: torch.from_numpy(f[k]) for k in f.files})
    base, ckpt = load_parent(DEFAULT_PARENT)
    params = (base.center.detach().numpy().copy(), base.scale.detach().numpy().copy(),
              decoder, noise, context_scale)
    base.cuda().requires_grad_(False)
    frozen, net_frozen = state_digest(base), state_digest(net)
    if frozen != lock['baseline_protocol']['source_state_sha256'] or ckpt['threshold'] != p['original']:
        raise ValueError('source checkpoint changed')
    with np.load(ref/'readout.npz', allow_pickle=False) as f:
        coef = torch.from_numpy(f['coef']).cuda()
    thresholds, original = p['calibration'], p['original']
    out.mkdir(parents=True)
    (out/'shots').mkdir()
    write(out/'protocol.json', dict(
        scope='EAST360_FROZEN_HIDDEN_DOSE_DIAGNOSTIC_NOT_MODEL_SELECTION',
        order=ids, startup=STARTUP, calibration=thresholds, original=original,
        source_sha256=PARENT_SHA256, experiment=str(experiment),
        parent_assets={k: sha(experiment/k) for k in ('protocol.json','seal.json','state.npz','per_shot.json')},
        code_sha256={**p['code_sha256'], Path(__file__).name: sha(Path(__file__))},
        doses=DOSES, additional_fourfold_controls=ADDITIONAL, arms=ARMS,
        new_fit=False, first_effect=persistent.FIRST_EFFECT,
        intervention='scale the same causal preactivation update, not the score or alarm cutoff'))
    hashes, witnesses = {}, {}
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('unsealed EAST outcome access')):
        for rank, sid in enumerate(ids, 1):
            name = f'{rank:04d}_{sid}.npz'
            f = experiment/'shots'/name
            if sha(f) != s['curve_sha256'][name]:
                raise ValueError('parent evidence changed')
            with np.load(f, allow_pickle=False) as z:
                t, old_risk, actual = z['times'], z['inherited'], z['sustained']
            if rank <= STARTUP:
                curves = dict.fromkeys(ARMS, old_risk)
            else:
                if sha(io23.path_for('EAST', sid)) != lock['hashes'][str(sid)]:
                    raise ValueError('EAST signal changed')
                raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                np.testing.assert_array_equal(t, raw['times'])
                signal = prepare_actual(raw, rank, lock['mapping'])
                args = params, old, quiet, mapper, coef
                curves = encode(base, net, signal, *args)
                np.testing.assert_array_equal(curves['inherited'], old_risk)
                np.testing.assert_array_equal(curves['sustained'], actual)
                for v in curves.values():
                    np.testing.assert_array_equal(v[:persistent.FIRST_EFFECT], old_risk[:persistent.FIRST_EFFECT])
                first_lagged = persistent.FIRST_EFFECT+persistent.STRIDE
                np.testing.assert_array_equal(curves['lagged4'][:first_lagged], old_risk[:first_lagged])
                if rank in (21, 120, 360):
                    length = persistent.FIRST_EFFECT+2*persistent.STRIDE+3
                    clip = {k: signal[k][:length] for k in ('values','available')}
                    for k, v in encode(base, net, clip, *args).items():
                        raw_prefix = curves[k][:len(v)]
                        np.testing.assert_allclose(v, raw_prefix, atol=2e-6, rtol=0.)
                        for tau in thresholds.values():
                            np.testing.assert_array_equal(v>=tau, raw_prefix>=tau)
                    witnesses[str(sid)] = length
            path = out/'shots'/name
            with path.open('xb') as f:
                np.savez_compressed(f, times=t, **curves)
            hashes[name] = sha(path)
            if rank%120 == 0:
                print(dict(stage='dose_seal', shots=rank, seconds=time.monotonic()-start), flush=True)
    if state_digest(base) != frozen or state_digest(net) != net_frozen:
        raise ValueError('model changed during frozen replay')
    write(out/'seal.json', dict(curve_sha256=hashes, prefix_witnesses=witnesses))
    per = {c: {k: [] for k in ARMS} for c in thresholds}
    for rank, sid in enumerate(ids, 1):
        name = f'{rank:04d}_{sid}.npz'
        path = out/'shots'/name
        if sha(path) != hashes[name]:
            raise ValueError('predictions changed after sealing')
        d, down = io23.read_outcome('EAST', sid)
        with np.load(path, allow_pickle=False) as f:
            for c, v in per.items():
                for k in ARMS:
                    a = score_curve(f['times'], f[k], d, down, original if rank<=STARTUP else thresholds[c])
                    v[k].append(dict(shot=sid, disruptive=d, down_s=down, **a))
    old_per = read(experiment/'per_shot.json')
    if any(g[k] != old_per[c][k] for c, g in per.items() for k in ('inherited','sustained')):
        raise ValueError('inherited first alarms changed')
    metrics = {c: {k: timing_decomposition(v) for k, v in g.items()} for c, g in per.items()}
    write(out/'per_shot.json', per)
    write(out/'result.json', dict(status='COMPLETED_FROZEN_DOSE_DIAGNOSTIC_EAST360',
        metrics=metrics, paired_vs_inherited={c: {k: flips(g['inherited'],g[k]) for k in ARMS[1:]}
                                             for c, g in per.items()},
        seconds=time.monotonic()-start))
    print({c: {k: (v['tp'],v['fp']) for k, v in g.items()} for c, g in metrics.items()}, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    run(a.experiment.resolve(), a.output.resolve())
