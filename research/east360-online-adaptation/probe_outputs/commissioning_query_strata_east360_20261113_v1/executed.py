"""Compare balanced, delayed EAST replay with one safety donor changed."""

import argparse
import os
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import expit, logit
import torch

from joint_lora_hindsight_capacity_20261104 import write_new
from layer0_local_memory_20261105 import EPISODE_RADIUS, LR, STEPS
from layer0_query_protocol_20261105 import reveal_block

from . import commissioning_field_scale_20261113 as field
from . import commissioning_write_delay_20261112 as delay
from . import io23, simple_data, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest


ROOT = io23.ROOT / 'probe_outputs'
FIXED = ROOT / 'commissioning_write_delay_east360_20261112_v2'
QUERY = ROOT / 'layer0_direct_transfer360_20261112_v2'
FIELD = ROOT / 'commissioning_field_scale_east2953_20261113_v1' / 'protocol.json'
group, lag = delay.group, delay.lag
BASES = ('bias_only', 'current_5')
POLICIES = ('recent', 'recent_supported', 'matched_supported', 'high_supported')
PARENTS = {f'{b}_{p}':b for b in BASES for p in POLICIES}
ARMS = ('fixed', *BASES, *PARENTS)
N, CIII, POINTS, RIDGE = field.N, io23.RAW23.index('CIII'), 8, .10


def field_bin(signal, cut):
    x, m = signal['values'][:N], signal['available'][:N]
    if (len(x) < N or not m[:, (field.BT, CIII)].all()
            or not np.isfinite(x[:, CIII]).all()
            or np.isclose(x[:, CIII], -10., atol=.01, rtol=0.).any()):
        return 0
    return field.direction(x[:, field.BT], cut)


def stratum(signal, cut):
    return field_bin(signal, cut) == 1


def select(records, next_rank):
    seen = [i for i, r in enumerate(records) if r['available_from'] <= next_rank and r['points'] is not None]
    d = next((i for i in reversed(seen) if records[i]['disruptive']), None)
    s = [i for i in reversed(seen) if not records[i]['disruptive']]
    return d, next(iter(s), None), next((i for i in s if records[i]['bin'] == 1), None)


def teaching_points(t, baseline, disruptive, down):
    if disruptive:
        remaining = np.round((down-t)*1000., 9)
        valid = np.flatnonzero((remaining > 30.) & (remaining <= 125.) & (np.arange(len(t)) >= N-1))
        if len(valid) < POINTS:
            return None
        return valid[np.linspace(0, len(valid)-1, POINTS, dtype=int)]
    if len(t)-N+1 < POINTS:
        return None
    return np.array([j[np.argmax(baseline[j])] for j in np.array_split(np.arange(N-1, len(t)), POINTS)])


def fit(d, s):
    # One shot per class, eight distinct endpoints per shot; both parent views
    # receive the same supervision. A single residual is shared between views.
    x = np.concatenate([r['hidden'][:, r['points']] for r in (d, s)], axis=1)
    z = np.concatenate([logit(np.clip(r['base'][:, r['points']], 1e-6, 1.-1e-6)) for r in (d, s)], axis=1)
    y = np.repeat([1., 0.], POINTS)
    w = np.zeros(x.shape[-1], dtype=np.float64)
    for _ in range(STEPS):
        w -= LR * (((expit(z+x@w)-y)[..., None]*x).mean((0, 1)) + RIDGE*w)
        w *= min(1., EPISODE_RADIUS / max(1e-12, np.linalg.norm(w)))
    return w


def features(signal, args, decoder, path, weights, cold):
    if cold:
        _, h, _, _ = group.guard._track(signal, args)
        return np.stack((h, h)), None
    base = args[0]
    q, h, z, beta, prior, phase, correction = group.parent_track(signal, args, decoder, path)
    avail, fields, wave = group.feedback_waveform(signal, q, args[2], z.device)
    states, eligible = group._estimate_innovation(
        base, signal, q, avail, phase, correction, prior, fields, wave, decoder, weights)
    parent_h, current_h = h.clone(), h.clone()

    def capture(j, k, plain, views, enabled):
        if j == 0:
            parent_h[k] = plain
        current_h[k] = torch.where(enabled[:, None], views[0], plain)

    current, _, dose, _, _, parent = delay.render(
        base, signal, q, h, z, beta, phase, correction, states, eligible,
        waits=(0, 5), logits=True, hidden_trace=capture)
    return torch.stack((parent_h, current_h)).detach().cpu().numpy(), (
        expit(np.stack((parent, current[-1]))), q+lag.H+1, eligible, dose[-1])


def predict(base, hidden, bin_id, weights, cold=False):
    values = {}
    for k, w in weights.items():
        v = base
        if w is not None and not cold and (k == 'recent' or bin_id == 1) and base.shape[1] >= N:
            v = base.copy()
            v[:, N-1:] = expit(logit(np.clip(base[:, N-1:], 1e-6, 1.-1e-6)) + hidden[:, N-1:]@w)
        values.update({f'{name}_{k}': v[i] for i, name in enumerate(BASES)})
    return values


def violations(records, candidate, policy, levels):
    issues = []
    for r in records:
        y, down, t = r['disruptive'], r['down'], r['times']
        shadow = predict(r['base'], r['hidden'], r['bin'], {policy: candidate})
        for i, name in enumerate(BASES):
            for tau in levels:
                a, b = (score_curve(t, v, y, down, tau)
                        for v in (r['base'][i], shadow[f'{name}_{policy}']))
                early_a, early_b = ((v['lead_ms'] or 0.) > lead.EARLY_MS for v in (a,b))
                if ((not y and b['alarm'] and not a['alarm']) or (a['timely'] and not b['timely'])
                        or (not early_a and early_b)):
                    issues.append(dict(shot=r['shot'], base=name, threshold=tau, before=a, after=b))
    return issues


def bound(path, digest):
    if sha(path) != digest:
        raise ValueError(f'changed sealed artifact: {path}')
    return path


def save(path, arrays):
    with path.open('xb') as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    return sha(path)


def run(out):
    start = time.monotonic()
    lag.configure()
    lock = lag.load_inputs()
    p, ref, args, decoder, path = group.load_reference(lock)
    fixed, frozen = (read(FIXED/f'{k}.json') for k in ('protocol', 'seal'))
    plan, budget = (read(QUERY/f'{k}.json') for k in ('protocol', 'seal'))
    bins = read(FIELD)
    cut = bins['cutoff_from_cold']
    bound(QUERY/'protocol.json', budget['protocol_sha256'])
    assert p['order'] == fixed['order'] == plan['order']
    assert p['startup'] == fixed['startup'] == plan['cold'] == plan['block'] == 20
    assert plan['query'] == 16 and len(budget['blocks']) == 18 and len(frozen['curve_sha256']) == 360
    assert p['calibration'] == fixed['calibration'] and p['original'] == fixed['original']
    assert p['order'][:20] == bins['preparation'] and bins['first'] == N-1
    assert fixed['reference_seal'] == sha(group.REFERENCE/'seal.json')
    assert fixed['history_seal'] == sha(delay.HISTORY/'seal.json')
    teaching = delay.HISTORY/'teaching.json'
    bound(teaching, read(delay.HISTORY/'protocol.json')['teaching_sha256'])
    wg = torch.as_tensor(read(teaching)['weights'], device='cuda', dtype=torch.float32)
    initial = [state_digest(a) for a in args[:2]]
    levels = sorted(set(p['calibration'].values()))
    for name in ('shots', 'blocks'):
        (out/name).mkdir(parents=True)
    copyfile(Path(__file__), out/'executed.py')
    write_new(out/'protocol.json', dict(
        scope='EAST360_BALANCED_DELAYED_QUERY_REPLAY', order=p['order'], startup=p['startup'],
        original=p['original'], calibration=p['calibration'], arms=ARMS, first=N-1, field_cut=cut,
        fixed_seal=sha(FIXED/'seal.json'), query_seal=sha(QUERY/'seal.json'), field_protocol=sha(FIELD),
        executed_sha256=sha(out/'executed.py'), points_per_shot=POINTS, optimizer_steps=STEPS,
        learning_rate=LR, radius=EPISODE_RADIUS, l2=RIDGE,
        fitting='one recent disruptive and one safe shot, balanced by class and points; one shared residual',
        guard='all revealed labels, both parent views and the fixed calibration levels; reject to zero if '
              'new false alarm, lost timely hit or newly >3000 ms; no current/next label',
        gates='none for recent; for supported arms require prior matched safe at fit time and a low-Bt, '
              'CIII-present 100-point prefix at inference; compare recent, low-Bt, and high-Bt safety donors',
        schedule='frozen prior query IDs and reveal boundaries; read fit eligibility only after each new reveal'))
    oracle = io23.read_outcome
    records, hashes, receipts, witness = [], {}, [], {}
    state = dict.fromkeys(POLICIES)
    error = 0.
    block_size = plan['block']
    fitted = {k:0 for k in POLICIES}
    queries = 0
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('uncommitted outcome access')):
        for b, old in enumerate(budget['blocks'], 1):
            name = f'{b:02d}'
            begin, end = (b-1)*block_size, b*block_size
            ids = p['order'][begin:end]
            commit = read(bound(QUERY/'blocks'/f'{name}_commit.json', old['commit_sha256']))
            selected = commit['selected']
            assert commit['rank_end'] == end and commit['block'] == b
            assert len(set(selected)) == len(selected) and set(selected) <= set(ids)
            assert len(selected) == (20 if b == 1 else int(b < len(budget['blocks'])))
            queries += len(selected) if b > 1 else 0
            pending, block_hashes = {}, {}
            for rank, sid in enumerate(ids, begin+1):
                fn = f'{rank:04d}_{sid}.npz'
                with np.load(bound(FIXED/'shots'/fn, frozen['curve_sha256'][fn]), allow_pickle=False) as a, \
                        np.load(bound(group.REFERENCE/'shots'/fn, ref['curve_sha256'][fn]), allow_pickle=False) as f:
                    curves = {k:a[k] for k in ('fixed', *BASES)}
                    base, t = np.stack([curves[k] for k in BASES]), f['times']
                    release, eligible, dose = a['release'], a['eligible'], a['dose'][-1]
                bound(io23.path_for('EAST', sid), lock['hashes'][str(sid)])
                raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                np.testing.assert_array_equal(raw['times'], t)
                sig = lag.prepare_actual(raw, rank, lock['mapping'])
                g = field_bin(sig, cut)
                active = any(w is not None and (k == 'recent' or g == 1) for k,w in state.items())
                cold = b == 1
                h = None
                if active or sid in selected or (b >= 10 and g == 1 and len(witness) < 2 and len(t) > 550):
                    h, check = features(sig, args, decoder, path, wg, cold)
                    h = np.concatenate((h, np.ones_like(h[..., :1])), axis=-1)
                    if check is not None:
                        pred, r, e, d = check
                        np.testing.assert_array_equal(r, release)
                        np.testing.assert_array_equal(e, eligible)
                        np.testing.assert_allclose(d, dose, atol=3e-6, rtol=1e-6)
                        np.testing.assert_allclose(pred, base, atol=3e-6, rtol=0.)
                        for tau in levels:
                            np.testing.assert_array_equal(pred >= tau, base >= tau)
                        error = max(error, float(np.abs(pred-base).max()))
                curves.update(predict(base, h, g, state, cold=cold))
                for k, parent in PARENTS.items():
                    np.testing.assert_array_equal(curves[k][:N-1], curves[parent][:N-1])
                if b >= 10 and g == 1 and len(witness) < 2 and len(t) > 550:
                    point = 550
                    short = {**sig, **{k:sig[k][:point] for k in ('values', 'available')}}
                    assert field_bin(short, cut) == g
                    sh, (sp, sr, se, sd) = features(short, args, decoder, path, wg, False)
                    keep = release < point
                    np.testing.assert_array_equal(sr, release[keep])
                    np.testing.assert_array_equal(se, eligible[keep])
                    np.testing.assert_allclose(sd, dose[:point], atol=3e-6, rtol=1e-6)
                    sh = np.concatenate((sh, np.ones_like(sh[..., :1])), axis=-1)
                    arms = dict(zip(BASES, sp, strict=True)) | predict(sp, sh, g, state)
                    witness[str(sid)] = dict(points=point, active=active,
                                             max_error=max(float(np.abs(v-curves[k][:point]).max())
                                                           for k,v in arms.items()))
                    for k,v in arms.items():
                        np.testing.assert_allclose(v, curves[k][:point], atol=3e-6, rtol=0.)
                        for tau in levels:
                            np.testing.assert_array_equal(v >= tau, curves[k][:point] >= tau)
                if sid in selected:
                    pending[sid] = dict(rank=rank, bin=g, times=t, hidden=h, base=base)
                block_hashes[fn] = save(out/'shots'/fn, dict(times=t, field_bin=g, **curves))
            here = out/'blocks'/f'{name}_commit.json'
            write_new(here, dict(block=b, rank_end=end, selected=selected, shots=block_hashes,
                                 state={k:v.tolist() if v is not None else None for k,v in state.items()},
                                 committed_ns=time.monotonic_ns()))
            revealed = reveal_block(here, selected, oracle)
            rv = out/'blocks'/f'{name}_reveal.json'
            write_new(rv, revealed)
            for r in revealed['outcomes']:
                view = pending[r['shot']]
                pts = teaching_points(view['times'], view['base'].max(0), r['disruptive'], r['down'])
                records.append(dict(r, **view, available_from=end+1, points=pts))
            old_update = read(bound(QUERY/'blocks'/f'{name}_update.json', old['update_sha256']))
            log = {}
            if old_update.get('fit') is not None:
                assert old_update['fit']['steps'] == STEPS and b < len(budget['blocks'])
                d, recent, matched = select(records, end+1)
                high = next((i for i,r in reversed(list(enumerate(records))) if not r['disruptive']
                             and r['points'] is not None and r['bin'] == -1), None)
                assert d is not None and recent is not None
                for policy, donor in zip(POLICIES, (recent, recent, matched, high), strict=True):
                    supported = policy == 'recent' or (matched is not None and donor is not None)
                    s = recent if donor is None else donor
                    w = fit(records[d], records[s])
                    fitted[policy] += STEPS
                    issues = violations(records, w, policy, levels) if supported else []
                    state[policy] = w if supported and not issues else None
                    log[policy] = dict(disrupted=records[d]['shot'], safe=records[s]['shot'],
                                       applicable=supported, accepted=state[policy] is not None,
                                       violations=issues, coefficient=w.tolist())
            up = out/'blocks'/f'{name}_update.json'
            write_new(up, dict(next_rank=end+1, guard_ids=[r['shot'] for r in records],
                               reveal_sha256=sha(rv), fits=log, updated_ns=time.monotonic_ns()))
            receipts.append(dict(commit=sha(here), reveal=sha(rv), update=sha(up)))
            hashes.update(block_hashes)
            print(dict(block=b, rank=end, accepted=[k for k,v in log.items() if v['accepted']],
                       seconds=round(time.monotonic()-start, 2)), flush=True)
    assert initial == [state_digest(a) for a in args[:2]]
    assert queries == plan['query'] and set(fitted.values()) == {160} and len(witness) == 2
    write_new(out/'seal.json', dict(protocol_sha256=sha(out/'protocol.json'), shots=hashes, blocks=receipts,
                                   queries=queries, optimizer_steps=fitted, parent_max_error=error,
                                   truncated=witness, seconds=time.monotonic()-start))


def score(out):
    s = read(out/'seal.json')
    p = read(bound(out/'protocol.json', s['protocol_sha256']))
    bound(out/'executed.py', p['executed_sha256'])
    assert len(s['shots']) == len(p['order']) == 360 and len(s['blocks']) == 18
    revealed = set()
    previous = 0
    expected = dict.fromkeys(POLICIES)
    for b, hashes in enumerate(s['blocks'], 1):
        commit, reveal, update = (read(bound(out/'blocks'/f'{b:02d}_{k}.json', h))
                                  for k,h in hashes.items())
        assert update['reveal_sha256'] == hashes['reveal'] and reveal['commit_sha256'] == hashes['commit']
        assert commit['rank_end'] == b*20 and update['next_rank'] == b*20+1
        assert previous <= commit['committed_ns'] <= update['updated_ns']
        assert commit['state'] == expected
        assert all(s['shots'][k] == v for k,v in commit['shots'].items())
        selected = [r['shot'] for r in reveal['outcomes']]
        assert commit['selected'] == selected and not (set(selected) & revealed)
        for r in reveal['outcomes']:
            assert commit['committed_ns'] <= r['read_start_ns'] <= r['read_end_ns'] <= update['updated_ns']
        revealed.update(selected)
        assert set(update['guard_ids']) == revealed
        for policy, fit_record in update['fits'].items():
            assert {fit_record[k] for k in ('disrupted', 'safe')} <= revealed
            expected[policy] = fit_record['coefficient'] if fit_record['accepted'] else None
        previous = update['updated_ns']
    rows = {c:{k:[] for k in ARMS} for c in p['calibration']}
    bins, modified = [], {k:[] for k in PARENTS}
    for rank,sid in enumerate(p['order'], 1):
        y, down, times = group.hdf._raw_truth(sid)
        fn = f'{rank:04d}_{sid}.npz'
        with np.load(bound(out/'shots'/fn, s['shots'][fn]), allow_pickle=False) as a:
            np.testing.assert_array_equal(times, a['times'])
            bins.append(int(a['field_bin']))
            for k, parent in PARENTS.items():
                if not np.array_equal(a[k], a[parent]):
                    modified[k].append(sid)
            for c, tau in p['calibration'].items():
                level = p['original'] if rank <= p['startup'] else tau
                for k in ARMS:
                    rows[c][k].append(dict(shot=sid, disruptive=y, down_s=down,
                                          **score_curve(times, a[k], y, down, level)))
    old = read(FIXED/'per_shot.json')
    groups = {}
    for c, rr in rows.items():
        assert all(rr[k] == old[c][k] for k in ('fixed', *BASES))
        subsets = {'full360': list(range(360)), 'postcold340': list(range(20,360)),
                   'target_cell': [i for i,g in enumerate(bins) if g == 1],
                   'target_cell_after_first_reveal': [i for i,g in enumerate(bins) if i >= 180 and g == 1]}
        groups[c] = {}
        for group_name, ix in subsets.items():
            paired = {k:[rs[i] for i in ix] for k,rs in rr.items()}
            comparisons = [(b, k) for k,b in PARENTS.items()]
            comparisons += [(f'{b}_recent_supported', f'{b}_{k}') for b in BASES
                            for k in ('matched_supported', 'high_supported')]
            groups[c][group_name] = dict(
                metrics={k:lead.summarize(rs) for k,rs in paired.items()},
                paired={f'{a}__{b}':dict(flips=flips(paired[a], paired[b]), lead=lead.paired(paired[a], paired[b]))
                        for a,b in comparisons})
    write_new(out/'per_shot.json', rows)
    write_new(out/'result.json', dict(scope='SEALED_360_DEVELOPMENT', groups=groups, modified_shots=modified,
                                     original_hdf5_scored=360*len(ARMS)*len(rows), revealed_ids=sorted(revealed)))
    for c, v in groups.items():
        print(c, {k:(m['by_deadline']['30']['tp'], m['fp']) for k,m in v['full360']['metrics'].items()},
              flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    opts = parser.parse_args()
    run(opts.out.resolve())
    score(opts.out.resolve())
