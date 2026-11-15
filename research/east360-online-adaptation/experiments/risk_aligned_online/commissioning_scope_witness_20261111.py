"""Post-seal comparison of two action subspaces, not an online policy."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_coincident_support_audit_20261111 as hdf
from . import commissioning_complementary_feedback_20261111 as comp
from . import commissioning_eos_replay_20261111 as eos
from . import commissioning_lag_clock_20261111 as clock
from . import commissioning_lag_guard_20261111 as guard
from . import commissioning_lag_mixer_20261111 as lag
from . import io23, simple_data
from .history_gradient_metric_20261102.run import read, sha, write
from .mask_separated_intra_20261103.run import state_digest


ARCHIVE = io23.ROOT/'probe_outputs/commissioning_complementary_feedback_east360_20261111_v1'
ARMS = ('cross_family', 'within_family')


def witness_points(times, probs, support, disrupted, down):
    if disrupted:
        lead = np.round((down-times)*1000., 9)
        return np.flatnonzero(support & (lead > 30.) & (lead <= 125.))
    return np.asarray([a[np.argmax(probs[a])] for a in
                       np.array_split(np.flatnonzero(support), 4) if len(a)], dtype=np.int64)


def ascent_candidate(raw, g, mask):
    n = len(lag.CHANNELS)
    theta = raw.new_zeros((len(raw), n, n), requires_grad=True)
    with patch.object(lag, 'CROSS', mask):
        d = lag.patch_input(raw, theta)
        q, = torch.autograd.grad((d*g).sum(), theta)
        return lag.patch_input(raw, guard._unit(q.detach()))-raw


def hidden_forward(base, x, m, transform):
    hidden = []
    h = base.embedding.register_forward_pre_hook(lambda _l, v: (transform(v[0]),))
    v = base.risk_head.register_forward_pre_hook(lambda _l, a: hidden.append(a[0]))
    try:
        base(x, m)
    finally:
        v.remove()
        h.remove()
    return hidden[0]


def action_witness(base, x, m, phase, beta):
    saved = []

    def original(raw):
        anchor = lag.patch_input(raw, phase).detach().requires_grad_()
        saved.extend((raw.detach(), anchor))
        return anchor

    h0 = hidden_forward(base, x, m, original)
    raw, anchor = saved
    g, = torch.autograd.grad((h0*beta).sum(), anchor)
    anchor = anchor.detach()
    old = anchor-raw
    first, second, used = comp.equal_angle(
        old, ascent_candidate(raw, g, lag.CROSS), ascent_candidate(raw, g, ~lag.CROSS))
    mask = used.reshape((-1,)+(1,)*(raw.ndim-1))
    with torch.no_grad():
        paths = [torch.where(mask, raw+candidate, anchor) for candidate in (first, second)]
        dose = torch.stack([(v-raw).flatten(1).square().mean(1).sqrt()
                            for v in (anchor, *paths)])
        error = float((dose[1:]-dose[0]).abs().max())
        if error > comp.TOL:
            raise ValueError('the compared realized input doses differ')
        hs = hidden_forward(base, torch.cat((x, x)), torch.cat((m, m)),
                            lambda _raw: torch.cat(paths)).chunk(2)
        delta = torch.stack([torch.where(used, ((h-h0.detach())*beta).sum(-1), 0.) for h in hs])
    if not bool(torch.isfinite(delta).all()):
        raise FloatingPointError('nonfinite change in the frozen risk readout')
    return delta.cpu().numpy(), used.cpu().numpy(), error


def phase_at(signal, args, decoder, params, releases, endpoints):
    base, net, (center, scale, forecast, noise, _), *_ = args
    slots = np.searchsorted(releases, endpoints, side='right')-1
    if np.any(slots < 0):
        raise ValueError('a witness endpoint has no prior completed forecast')
    origins, inverse = np.unique(releases[slots]-lag.H-1, return_inverse=True)
    tr = lag.direct.track(net, signal, center, scale, origins)
    i = np.searchsorted(tr['q'], origins)
    np.testing.assert_array_equal(tr['q'][i], origins)
    if not tr['observed'][i].all():
        raise ValueError('archived forecast support is not reproduced')
    prior = lag.direct.base_forecast(tr, forecast)[i]
    target = lag.direct.normalized_target(tr, noise)[i]
    real, _ = guard._bases(base, signal, origins, prior, target, decoder)
    key = torch.as_tensor(tr['key'], device=base.center.device)
    return -lag.RADIUS*guard.signed_direction(*real, key, params)[inverse]


def logit(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-7, 1.-1e-7)
    return np.log(x)-np.log1p(-x)


def summarize(rows):
    supported = [r for r in rows if r['compared'] > 0]
    crossing = {k:[r['shot'] for r in supported if r['crossing'][k]] for k in ARMS}
    return dict(shots=len(rows), endpoints=sum(r['selected'] for r in rows),
                compared_shots=len(supported), compared_endpoints=sum(r['compared'] for r in rows),
                crossing=crossing, exclusive={k:sorted(set(crossing[k])-set(crossing[other]))
                     for k,other in (ARMS, ARMS[::-1])},
                median_max_gain={k:float(np.median([r['max_gain'][k] for r in supported]))
                                  if supported else None for k in ARMS})


def run(archive, out):
    lag.configure()
    started = time.monotonic()
    p, s = (read(archive/k) for k in ('protocol.json','seal.json'))
    ledger = read(archive/'per_shot.json')['0.2'][clock.OLD]
    ref, lock = Path(p['reference']), lag.load_inputs()
    gp, _, _, args, decoder = eos._load(ref, lock)
    if (p['scope'] != 'EAST360_EQUAL_DOSE_COMPLEMENTARY_FEEDBACK_DEVELOPMENT'
            or p['order'] != lock['ids'] or len(ledger) != 360 or len(s['curve_sha256']) != 360
            or p['startup'] != gp['startup'] or p['calibration'] != gp['calibration']
            or p['reference_seal'] != sha(ref/'seal.json')
            or gp['teacher_sha256'] != sha(ref/'teacher.json')):
        raise ValueError('invalid sealed curve or direction reference')
    base, net = args[:2]
    params = torch.tensor(read(ref/'teacher.json')['path'], device='cuda', dtype=torch.float32)
    beta = args[6][:base.cfg.dim]
    frozen = [state_digest(a) for a in (base, net)]
    tau = p['calibration']['0.2']
    boundary = float(logit(tau))
    out.mkdir()
    copyfile(Path(__file__), out/'executed_probe.py')
    rows = []
    max_error = 0.
    for rank, (sid, old) in enumerate(zip(p['order'], ledger, strict=True), 1):
        name = f'{rank:04d}_{sid}.npz'
        path = archive/'shots'/name
        if sha(path) != s['curve_sha256'][name]:
            raise ValueError('the sealed score trace changed')
        d, down, t = hdf._raw_truth(sid)
        if (sid,d,down) != (old['shot'],old['disruptive'],old['down_s']):
            raise ValueError('the recorded outcome differs from raw HDF5')
        if rank <= p['startup'] or (d and old['timely']) or (not d and old['alarm']):
            continue
        with np.load(path, allow_pickle=False) as f:
            np.testing.assert_array_equal(f['times'], t)
            prob, support, release = f[clock.OLD], f['rotated'], f['phase_release']
        i = witness_points(t, prob, support, d, down)
        if not len(i):
            continue
        if sha(io23.path_for('EAST',sid)) != lock['hashes'][str(sid)]:
            raise ValueError('the input to the state witness changed')
        raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
        np.testing.assert_array_equal(t, raw['times'])
        signal = lag.prepare_actual(raw, rank, lock['mapping'])
        phase = phase_at(signal, args, decoder, params, release, i)
        x, m = lag._windows(base, signal, i)
        delta, keep, error = action_witness(base, x, m, phase, beta)
        max_error = max(max_error, error)
        margin = logit(prob[i[keep]])-boundary
        rows.append(dict(shot=sid, rank=rank, disruptive=d, selected=len(i), compared=int(keep.sum()),
            baseline_selected_crossing=bool(np.any(prob[i] >= tau)),
            crossing={k:bool(np.any(margin+v[keep] >= 0.)) for k,v in zip(ARMS,delta,strict=True)},
            max_gain={k:float(v[keep].max()) if keep.any() else None
                      for k,v in zip(ARMS,delta,strict=True)}))
        if len(rows)%40==0:
            print(dict(stage='scope_witness', selected=len(rows), rank=rank,
                       seconds=round(time.monotonic()-started,2)),flush=True)
    if [state_digest(a) for a in (base,net)] != frozen:
        raise ValueError('a frozen model changed')
    groups = {name:summarize([r for r in rows if r['disruptive']==d])
              for name,d in (('old_late_or_miss', True), ('old_unalarmed_safe', False))}
    write(out/'result.json',dict(
        status='POSTSEAL_EQUAL_DOSE_ACTION_SUBSPACE_WITNESS_NOT_AN_ONLINE_RESULT',
        archive=str(archive), seal_sha256=sha(archive/'seal.json'), executed_sha256=sha(Path(__file__)),
        selection='all prior late/miss supported (30,125] ms endpoints, up to four safe support quartile peaks',
        direction='one frozen risk ascent gradient per parameter mask, rotated from the old input direction',
        threshold=tau, angle_ratio=comp.RATIO, max_dose_rms_error=max_error,
        raw_outcomes_rechecked=len(ledger), counts=groups, rows=rows, seconds=time.monotonic()-started))
    print(groups, flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    a = parser.parse_args()
    run(a.archive.resolve(), a.out)
