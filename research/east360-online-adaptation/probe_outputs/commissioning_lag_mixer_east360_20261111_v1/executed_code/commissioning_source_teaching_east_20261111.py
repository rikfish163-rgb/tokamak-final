"""Source teaching ablation with identical causal adaptation on EAST360."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_persistence_20261111 as persistent
from . import commissioning_persistence_learning_20261111 as training
from . import commissioning_source_teaching_20261111 as teaching
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
from .warning_lead_metrics import report


EXTRA = {'legacy': ('legacy', 'sustained', 1.), 'legacy4': ('legacy', 'sustained', 4.),
         'legacy_current4': ('legacy', 'current_equal_norm', 4.),
         'coverage': ('coverage', 'sustained', 1.), 'coverage4': ('coverage', 'sustained', 4.),
         'coverage_opp4': ('coverage', 'opposite', 4.),
         'native': ('native', 'sustained', 1.), 'native4': ('native', 'sustained', 4.),
         'native_opp4': ('native', 'opposite', 4.)}
ARMS = ('fixed', 'inherited', *EXTRA)
MATCHED = ('coverage_matched4', 'native_matched4')


def match_magnitude(candidate, reference):
    """Preserve the candidate direction at each sample, with the reference norm."""
    a, b = (x.norm(dim=-1, keepdim=True) for x in (candidate, reference))
    if torch.any((a == 0) & (b != 0)):
        raise ValueError('no candidate direction at a nonzero reference action')
    den = torch.where(a > 0., a, 1.)
    return candidate*(b/den)


@torch.no_grad()
def encode(base, net, signal, params, old, quiet, weights, coef, *, norm_control=False):
    x, _ = features(base, signal, 'cuda')
    tr, pre, ready = training.observed_track(net, signal, np.arange(len(x)), params, old)
    views, _ = persistent.replay(*ready, quiet)
    key = torch.from_numpy(tr['key'])
    hidden = {'fixed': torch.from_numpy(tr['pre']).tanh().numpy(),
              'inherited': pre.tanh().numpy()}
    for name, (mode, variant, dose) in EXTRA.items():
        hidden[name] = training.view(pre, key, dose*torch.from_numpy(views[variant]), weights[mode]).numpy()
    if norm_control:
        measured = 4.*torch.from_numpy(views['sustained'])
        d = {k:torch.einsum('...k,...f,kfw->...w',key,measured,w) for k,w in weights.items()}
        for mode in ('coverage','native'):
            hidden[f'{mode}_matched4'] = (pre+match_magnitude(d[mode],d['legacy'])).tanh().numpy()
    return scores(x, hidden, coef, (*ARMS,*MATCHED) if norm_control else ARMS)


def fit(net, base, coef, params, old, quiet, legacy, lock, roles, scale, cutoff):
    advanced = io23.advanced_times()
    raw_train = teaching.collect(net, roles['train'], advanced, params, old, 'source_train')
    train_packs = {k: training.pack(r, quiet, scale) for k,r in raw_train.items()}
    del raw_train
    train_pairs = {k: teaching.receipt(v) for k,v in train_packs.items()}
    old_rows = {tuple(r.values()) for r in train_pairs['legacy']}
    coverage_rows = {tuple(r.values()) for r in train_pairs['coverage']}
    if not old_rows < coverage_rows or not all(r['inside_official'] for r in train_pairs['native']):
        raise ValueError('coverage must preserve the exact old pairs; native needs the original positive mask')
    cold = []
    for sid in lock['ids'][:STARTUP]:
        if sha(io23.path_for('EAST', sid)) != lock['hashes'][str(sid)]:
            raise ValueError('cold evidence changed')
        raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
        sig = prepare_actual(raw, STARTUP+1, lock['mapping'])
        cold.append(training.labelled(net, raw, sig, None, params, old, base=base, coef=coef))
    cold = training.pack(cold, quiet, scale)
    weights, beta = {}, coef[-25:-1]
    for k, rows in train_packs.items():
        print(dict(stage='teach', policy=k, pairs=len(rows[1][0]), cold=len(cold[1][0])), flush=True)
        weights[k] = training.fit(rows, cold, beta, cutoff)
    torch.testing.assert_close(weights['legacy'], legacy, atol=0., rtol=0.)
    held = teaching.collect(net, roles['val'], advanced, params, old, 'source_val')
    held = {k: training.pack(r, quiet, scale) for k,r in held.items()}
    pairs = dict(train=train_pairs, held={k:teaching.receipt(v) for k,v in held.items()})
    info = dict(
        legacy_reproduced_exactly=True, preserved_pairs=len(old_rows), added_pairs=len(coverage_rows-old_rows),
        source_train={k:training.concordance(train_packs[k],beta.cpu(),w) for k,w in weights.items()},
        source_held={pool:{k:training.concordance(rows,beta.cpu(),w) for k,w in weights.items()}
                     for pool,rows in held.items()},
        cold_training={k:training.concordance(cold,beta.cpu(),w) for k,w in weights.items()},
        map_norm={k:float(w.norm()) for k,w in weights.items()})
    return weights, pairs, info


def run(experiment, dose, out, *, reuse=None):
    configure()
    start = time.monotonic()
    lock = load_inputs()
    ids, roles = lock['ids'], io23.source_roles()
    p, s = (read(experiment/k) for k in ('protocol.json', 'seal.json'))
    dp, ds = (read(dose/k) for k in ('protocol.json', 'seal.json'))
    if (p['scope'] != 'EAST360_PERSISTENT_SIGNED_DEVIATION_FIXED_HEAD_DEVELOPMENT'
            or p['order'] != ids or p['startup'] != STARTUP or p['source_sha256'] != PARENT_SHA256
            or p['source_fit_ids'] != roles['train'] or p['source_held_ids'] != roles['val']
            or any(sha(experiment/k) != v for k,v in s['fitted'].items())
            or any(sha(Path(__file__).parent/k) != v for k,v in p['code_sha256'].items())):
        raise ValueError('matching sealed source and persistence calibration required')
    if (dp['scope'] != 'EAST360_FROZEN_HIDDEN_DOSE_DIAGNOSTIC_NOT_MODEL_SELECTION'
            or Path(dp['experiment']) != experiment or dp['order'] != ids
            or dp['calibration'] != p['calibration'] or len(ds['curve_sha256']) != len(ids)
            or any(sha(experiment/k) != v for k,v in dp['parent_assets'].items())):
        raise ValueError('same-dose sealed comparison required')
    previous = Path(p['previous'])
    if any(sha(previous/k) != v for k,v in p['previous_assets'].items()):
        raise ValueError('inherited fast state changed')
    prior = read(previous/'protocol.json')
    ref = Path(prior['reference'])
    if any(sha(ref/k) != v for k,v in prior['reference_assets'].items()):
        raise ValueError('fixed readout or calibration changed')
    source = Path(read(ref/'protocol.json')['source_prior'])
    if any(sha(source/k) != v for k,v in prior['source_assets'].items()):
        raise ValueError('source representation changed')
    forecast = Path(prior['forecast_reference'])/'fit.npz'
    if sha(forecast) != prior['forecast_source_fit_sha256']:
        raise ValueError('fixed direct decoder changed')
    with np.load(forecast, allow_pickle=False) as f:
        decoder, noise = (f['decoder_weight'], f['decoder_bias']), f['scale']
    with np.load(previous/'mapper.npz', allow_pickle=False) as f:
        old, context_scale = torch.from_numpy(f['mapper']), f['context_scale']
    with np.load(experiment/'state.npz', allow_pickle=False) as f:
        legacy, quiet = torch.from_numpy(f['mapper']), (f['location'], f['scale'])
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
    cutoffs, original = p['calibration'], p['original']
    files = [experiment/k for k in ('protocol.json','seal.json','state.npz')]
    files.extend(dose/k for k in ('protocol.json','seal.json'))
    arms = ARMS
    if reuse is not None:
        rp, rs = (read(reuse/k) for k in ('protocol.json','seal.json'))
        if (rp['scope'] != 'EAST360_RELEASE_ALIGNED_SOURCE_TEACHING_DEVELOPMENT'
                or Path(rp['inherited_reference'])!=experiment or Path(rp['dose_reference'])!=dose
                or rp['order']!=ids or rp['calibration']!=cutoffs or tuple(rp['arms'])!=ARMS
                or any(sha(reuse/k)!=v for k,v in rs['fit_sha256'].items())):
            raise ValueError('norm control requires the exact previous teaching fit')
        files.extend(reuse/k for k in ('protocol.json','seal.json','mapper.npz'))
        arms = (*arms,*MATCHED)
    out.mkdir(parents=True)
    (out/'shots').mkdir()
    write(out/'protocol.json', dict(
        scope=('EAST360_FROZEN_SOURCE_TEACHING_MAGNITUDE_CONTROL' if reuse is not None else
               'EAST360_RELEASE_ALIGNED_SOURCE_TEACHING_DEVELOPMENT'), order=ids,
        startup=STARTUP, source_sha256=PARENT_SHA256, calibration=cutoffs, original=original,
        arms=arms, views=EXTRA, source_fit_ids=roles['train'], source_held_ids=roles['val'],
        cold_fit_ids=ids[:STARTUP], no_suffix_fit=True, no_cross_shot_update=True,
        inherited_reference=str(experiment), dose_reference=str(dose),
        data_bindings=lock['bound'], source_label_sha256=sha(io23.JTEXT_ADVANCED_TIME_PATH),
        reference_sha256={str(path):sha(path) for path in files},
        reused_fit=str(reuse) if reuse is not None else None,
        magnitude_control=('match candidate preactivation L2 to legacy4 point by point'
                           if reuse is not None else None),
        code_sha256={**p['code_sha256'],
                     Path(teaching.__file__).name:sha(Path(teaching.__file__)),
                     Path(__file__).name:sha(Path(__file__))},
        first_effect=persistent.FIRST_EFFECT,
        teaching=dict(coverage='fill missing fixed lead bins; earlier context allowed at >=127',
                      native='original J-TEXT positive mask and strict >10 ms; thirds of the native window',
                      shared='same old safe rows, same cold20 and EAST teaching, forecaster, objective, optimizer, seed'),
        doses='unit size and prior diagnostic fourfold size; no gain or threshold selection on the suffix'))
    if reuse is None:
        weights, pairs, fit_info = fit(net,base,coef,params,old,quiet,legacy,lock,roles,
                                       np.asarray(info['match_scale']),cutoffs['0.2'])
    else:
        with np.load(reuse/'mapper.npz', allow_pickle=False) as f:
            weights = {k:torch.from_numpy(f[k]) for k in teaching.POLICIES}
        torch.testing.assert_close(weights['legacy'],legacy,atol=0.,rtol=0.)
        pairs, fit_info = (read(reuse/k) for k in ('source_pairs.json','fit.json'))
    write(out/'source_pairs.json',pairs)
    write(out/'fit.json',fit_info)
    with (out/'mapper.npz').open('xb') as f:
        np.savez(f, **{k:w.numpy() for k,w in weights.items()})
    fitted = {k:sha(out/k) for k in ('fit.json','source_pairs.json','mapper.npz')}
    hashes, witnesses = {}, {}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed EAST outcome access')):
        for rank, sid in enumerate(ids,1):
            name = f'{rank:04d}_{sid}.npz'
            ep, dt = experiment/'shots'/name, dose/'shots'/name
            if sha(ep)!=s['curve_sha256'][name] or sha(dt)!=ds['curve_sha256'][name]:
                raise ValueError('sealed comparison changed')
            with np.load(ep,allow_pickle=False) as f, np.load(dt,allow_pickle=False) as g:
                t = f['times']
                np.testing.assert_array_equal(t,g['times'])
                old_curves = dict(fixed=f['fixed'], inherited=f['inherited'], legacy=f['sustained'],
                                  legacy4=g['real4'], legacy_current4=g['current4'])
            if reuse is not None:
                original_curves = reuse/'shots'/name
                if sha(original_curves)!=rs['curve_sha256'][name]:
                    raise ValueError('previous teaching curve changed')
                with np.load(original_curves,allow_pickle=False) as f:
                    np.testing.assert_array_equal(t,f['times'])
                    for k in old_curves:
                        np.testing.assert_array_equal(old_curves[k],f[k])
                    old_curves = {k:f[k] for k in ARMS}
            if rank<=STARTUP:
                curves = dict.fromkeys(arms,old_curves['fixed'])
            else:
                if sha(io23.path_for('EAST',sid)) != lock['hashes'][str(sid)]:
                    raise ValueError('raw signals changed')
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                np.testing.assert_array_equal(t,raw['times'])
                signal = prepare_actual(raw,rank,lock['mapping'])
                args = params,old,quiet,weights,coef
                curves = encode(base,net,signal,*args,norm_control=reuse is not None)
                for k,v in old_curves.items():
                    np.testing.assert_array_equal(curves[k],v)
                for k in arms[2:]:
                    n = persistent.FIRST_EFFECT
                    np.testing.assert_array_equal(curves[k][:n],curves['inherited'][:n])
                if rank in (21,120,360):
                    n = persistent.FIRST_EFFECT+2*persistent.STRIDE+3
                    prefix = {k:signal[k][:n] for k in ('values','available')}
                    for k,v in encode(base,net,prefix,*args,norm_control=reuse is not None).items():
                        np.testing.assert_allclose(v,curves[k][:len(v)],atol=2e-6,rtol=0.)
                        for cutoff in cutoffs.values():
                            np.testing.assert_array_equal(v>=cutoff,curves[k][:len(v)]>=cutoff)
                    witnesses[str(sid)] = n
            path = out/'shots'/name
            with path.open('xb') as f:
                np.savez_compressed(f,times=t,**curves)
            hashes[name] = sha(path)
            if rank%60 == 0:
                print(dict(stage='sealed',shots=rank,seconds=time.monotonic()-start),flush=True)
    if (state_digest(base)!=frozen or state_digest(net)!=net_frozen
            or any(sha(out/k)!=v for k,v in fitted.items())):
        raise ValueError('frozen parameters changed during replay')
    write(out/'seal.json', dict(curve_sha256=hashes,prefix_witnesses=witnesses,fit_sha256=fitted))
    per = {c:{k:[] for k in arms} for c in cutoffs}
    for rank,sid in enumerate(ids,1):
        name = f'{rank:04d}_{sid}.npz'
        path = out/'shots'/name
        if sha(path)!=hashes[name]:
            raise ValueError('post-seal prediction changed')
        d,down = io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as f:
            for c,g in per.items():
                for k in arms:
                    a = score_curve(f['times'],f[k],d,down,original if rank<=STARTUP else cutoffs[c])
                    g[k].append(dict(shot=sid,disruptive=d,down_s=down,**a))
    write(out/'per_shot.json',per)
    metrics = {c:{k:timing_decomposition(v) for k,v in g.items()} for c,g in per.items()}
    write(out/'result.json',dict(status='COMPLETED_EAST360_SOURCE_TEACHING_DEVELOPMENT',
        metrics=metrics,
        fixed_dose_flips={c:{k:flips(g[a],g[k]) for a,k in (
            ('legacy','coverage'),('coverage','native'),
            ('legacy4','coverage4'),('coverage4','native4'),
            ('coverage_opp4','coverage4'),('native_opp4','native4'))} for c,g in per.items()},
        equal_magnitude_flips={c:{k:flips(g['legacy4'],g[k]) for k in MATCHED}
                               for c,g in per.items()} if reuse is not None else {},
        seconds=time.monotonic()-start))
    write(out/'lead_sensitivity.json',report([out/'per_shot.json'],('fixed','inherited','legacy','legacy4','coverage4')))
    print({c:{k:(v['tp'],v['fp']) for k,v in g.items()} for c,g in metrics.items()},flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment',type=Path,required=True)
    ap.add_argument('--dose',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--reuse-for-norm-control',type=Path)
    a = ap.parse_args()
    run(a.experiment.resolve(),a.dose.resolve(),a.output.resolve(),
        reuse=a.reuse_for_norm_control.resolve() if a.reuse_for_norm_control else None)
