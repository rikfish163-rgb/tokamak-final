"""Equal-norm EAST360 direction routing from observed diagnostic agreement."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_direction_evidence_20261111 as policy
from . import commissioning_direction_fit_20261111 as teaching
from . import commissioning_persistence_20261111 as persistent
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


ENDPOINTS = ('fixed','inherited','legacy4','legacy_current4','native_matched4')
GATES = ('midpoint','constant','neutral','agreement','learned','lagged','inverted')
ARMS = (*ENDPOINTS,*GATES)


@torch.no_grad()
def encode(base, net, signal, params, old_map, quiet, maps, coef, gate):
    x,_ = features(base,signal,'cuda')
    t = np.arange(len(x))
    row = teaching.views(net,signal,t,params,old_map,quiet,maps)
    p,a,b = (row[k] for k in ('pre','old','native'))
    mix = policy.weights(row['evidence'],gate)
    hidden = dict(fixed=row['bare'].tanh().numpy(),inherited=p.tanh().numpy(),
                  legacy4=(p+a).tanh().numpy(),native_matched4=(p+b).tanh().numpy(),
                  legacy_current4=row['current'].numpy())
    for k,w in mix.items():
        hidden[k] = (p+policy.mix_direction(a,b,w)).tanh().numpy()
    q = np.arange(persistent.FIRST_EFFECT,len(t),persistent.STRIDE)
    traces = dict(route_index=q,agreement_evidence=row['evidence'][q],
                  routing_weights=np.column_stack([mix[k][q].numpy() for k in GATES]))
    return scores(x,hidden,coef,ARMS),traces,row['stats']


def run(ref, out):
    configure()
    start = time.monotonic()
    lock = load_inputs()
    p,s = (read(ref/k) for k in ('protocol.json','seal.json'))
    ids,roles = lock['ids'],io23.source_roles()
    if (p['scope']!='EAST360_FROZEN_SOURCE_TEACHING_MAGNITUDE_CONTROL'
            or p['order']!=ids or p['source_sha256']!=PARENT_SHA256 or p['startup']!=STARTUP
            or p['source_fit_ids']!=roles['train'] or p['source_held_ids']!=roles['val']
            or any(sha(ref/k)!=v for k,v in s['fit_sha256'].items())
            or any(sha(Path(k))!=v for k,v in p['reference_sha256'].items())
            or any(sha(Path(__file__).parent/k)!=v for k,v in p['code_sha256'].items())):
        raise ValueError('exact sealed directional endpoints required')
    p0 = read(Path(p['inherited_reference'])/'protocol.json')
    earlier = Path(p0['previous'])
    if any(sha(earlier/k)!=v for k,v in p0['previous_assets'].items()):
        raise ValueError('frozen inherited update changed')
    pre = read(earlier/'protocol.json')
    risk = Path(pre['reference'])
    if any(sha(risk/k)!=v for k,v in pre['reference_assets'].items()):
        raise ValueError('frozen hazard head or threshold changed')
    source = Path(read(risk/'protocol.json')['source_prior'])
    if any(sha(source/k)!=v for k,v in pre['source_assets'].items()):
        raise ValueError('frozen representation changed')
    forecast = Path(pre['forecast_reference'])/'fit.npz'
    if sha(forecast)!=pre['forecast_source_fit_sha256']:
        raise ValueError('diagnostic decoder changed')
    with np.load(forecast,allow_pickle=False) as f:
        decoder,noise = (f['decoder_weight'],f['decoder_bias']),f['scale']
    with np.load(earlier/'mapper.npz',allow_pickle=False) as f:
        old_map,context_scale = torch.from_numpy(f['mapper']),f['context_scale']
    with np.load(Path(p['inherited_reference'])/'state.npz',allow_pickle=False) as f:
        quiet = f['location'],f['scale']
    with np.load(ref/'mapper.npz',allow_pickle=False) as f:
        maps = {k:torch.from_numpy(f[k]) for k in ('legacy','native')}
    info = read(source/'result.json')
    net = LagState(info['inputs'],info['edges']).eval().requires_grad_(False)
    with np.load(source/'model.npz',allow_pickle=False) as f:
        net.load_state_dict({k:torch.from_numpy(f[k]) for k in f.files})
    base,ckpt = load_parent(DEFAULT_PARENT)
    params = (base.center.detach().numpy().copy(),base.scale.detach().numpy().copy(),
              decoder,noise,context_scale)
    base.cuda().requires_grad_(False)
    frozen,net_frozen = state_digest(base),state_digest(net)
    if frozen!=lock['baseline_protocol']['source_state_sha256'] or ckpt['threshold']!=p['original']:
        raise ValueError('frozen J-TEXT parent changed')
    with np.load(risk/'readout.npz',allow_pickle=False) as f:
        coef = torch.from_numpy(f['coef']).cuda()
    cutoffs,original = p['calibration'],p['original']
    out.mkdir(parents=True)
    (out/'shots').mkdir()
    code = (Path(__file__),Path(policy.__file__),Path(teaching.__file__))
    write(out/'protocol.json',dict(
        scope='EAST360_PREQUENTIAL_DIAGNOSTIC_DIRECTION_GATE_DEVELOPMENT',order=ids,
        startup=STARTUP,source_sha256=PARENT_SHA256,arms=ARMS,gates=GATES,calibration=cutoffs,original=original,
        reference=str(ref),reference_sha256={k:sha(ref/k) for k in ('protocol.json','seal.json','mapper.npz','source_pairs.json')},
        code_sha256={**p['code_sha256'],**{f.name:sha(f) for f in code}},
        gate_fit_source=roles['train'],gate_source_held=roles['val'],
        gate_held_cold=ids[:STARTUP],gate_fit_on_east=False,cold_used_for_inherited_components=True,
        data_bindings=lock['bound'],first_action=persistent.FIRST_EFFECT,
        first_forecast_evidence=teaching.FIRST_EVIDENCE,stride=persistent.STRIDE,
        steps=teaching.STEPS,magnitude='all new gates and native_matched4 use legacy4 pointwise internal L2 norm',
        agreement='old minus native Huber loss in the measured next block, released q+26; normalized globally and by four diagnostic groups',
        no_suffix_fit=True,no_cross_shot_update=True))
    manifest = read(ref/'source_pairs.json')
    pack_args = params,old_map,quiet,maps
    train,weight,r0 = teaching.pack(net,manifest['train'],roles['train'],*pack_args,'gate_source_train')
    gate = teaching.fit(train,weight,coef[-25:-1].cpu())
    held,_,r1 = teaching.pack(net,manifest['held'],roles['val'],*pack_args,'gate_source_held')
    fit_info = dict(train=dict(r0,**teaching.diagnostics(train,coef[-25:-1].cpu(),gate)),
                    source_held=dict(r1,**teaching.diagnostics(held,coef[-25:-1].cpu(),gate)),
                    intercept=float(gate[0]),coefficients=gate[1].tolist(),constant=float(gate[2]))
    write(out/'fit.json',fit_info)
    print(dict(stage='held_pair_check',pairs=r1['pairs'],evidence=r1['pairs_with_evidence'],
               forecast_sign=fit_info['source_held']['forecast_agreement_on_measured_windows']),flush=True)
    args = params,old_map,quiet,maps,coef,gate
    cold = {c:{k:[] for k in ARMS} for c in cutoffs}
    for sid in ids[:STARTUP]:
        if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
            raise ValueError('gate-held cold evidence changed')
        raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
        x = prepare_actual(raw,STARTUP+1,lock['mapping'])
        curves,_,_ = encode(base,net,x,*args)
        d,down = io23.read_outcome('EAST',sid)
        for c,g in cold.items():
            for k,v in curves.items():
                g[k].append(dict(shot=sid,disruptive=d,down_s=down,
                                 **score_curve(raw['times'],v,d,down,cutoffs[c])))
    write(out/'cold_gate_held_probe.json',dict(
        scope='gate-only held, inherited components have seen these 20 shots; not their locked stream predictions',
        metrics={c:{k:timing_decomposition(v) for k,v in g.items()} for c,g in cold.items()}))
    hashes,witnesses,comparisons = {},{},{}
    fit_hash = sha(out/'fit.json')
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed EAST outcome access')):
        for rank,sid in enumerate(ids,1):
            name = f'{rank:04d}_{sid}.npz'
            baseline = ref/'shots'/name
            if sha(baseline)!=s['curve_sha256'][name]:
                raise ValueError('sealed reference curve changed')
            with np.load(baseline,allow_pickle=False) as f:
                t,old = f['times'],{k:f[k] for k in ENDPOINTS}
            extra = {}
            if rank<=STARTUP:
                curves = dict.fromkeys(ARMS,old['fixed'])
            else:
                if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('raw signal changed')
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                np.testing.assert_array_equal(t,raw['times'])
                x = prepare_actual(raw,rank,lock['mapping'])
                curves,extra,comparisons[str(sid)] = encode(base,net,x,*args)
                for k,v in old.items():
                    np.testing.assert_array_equal(curves[k],v)
                n = persistent.FIRST_EFFECT
                for k in GATES:
                    np.testing.assert_array_equal(curves[k][:n],curves['inherited'][:n])
                n = teaching.FIRST_EVIDENCE
                for k,anchor,stop in (('learned','neutral',n),('agreement','midpoint',n),
                                      ('lagged','neutral',n+persistent.STRIDE),('inverted','neutral',n)):
                    np.testing.assert_array_equal(curves[k][:stop],curves[anchor][:stop])
                if rank in (21,120,360):
                    n += persistent.STRIDE+3
                    prefix = {k:x[k][:n] for k in ('values','available')}
                    v,_,_ = encode(base,net,prefix,*args)
                    for k,row in v.items():
                        np.testing.assert_allclose(row,curves[k][:len(row)],rtol=0.,atol=2e-6)
                        for tau in cutoffs.values():
                            np.testing.assert_array_equal(row>=tau,curves[k][:len(row)]>=tau)
                    witnesses[str(sid)] = n
            path = out/'shots'/name
            with path.open('xb') as f:
                np.savez_compressed(f,times=t,**curves,**extra)
            hashes[name] = sha(path)
            if rank%60==0:
                print(dict(stage='sealed',shots=rank,seconds=time.monotonic()-start),flush=True)
    if state_digest(base)!=frozen or state_digest(net)!=net_frozen or sha(out/'fit.json')!=fit_hash:
        raise ValueError('frozen predictors changed during replay')
    write(out/'seal.json',dict(curve_sha256=hashes,prefix_witnesses=witnesses,fit_sha256=fit_hash))
    write(out/'forecast_comparisons.json',comparisons)
    per = {c:{k:[] for k in ARMS} for c in cutoffs}
    for rank,sid in enumerate(ids,1):
        name = f'{rank:04d}_{sid}.npz'
        path = out/'shots'/name
        if sha(path)!=hashes[name]:
            raise ValueError('prediction changed after sealing')
        d,down = io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as f:
            for c,g in per.items():
                for k in ARMS:
                    a = score_curve(f['times'],f[k],d,down,original if rank<=STARTUP else cutoffs[c])
                    g[k].append(dict(shot=sid,disruptive=d,down_s=down,**a))
    write(out/'per_shot.json',per)
    metrics = {c:{k:timing_decomposition(v) for k,v in g.items()} for c,g in per.items()}
    write(out/'result.json',dict(status='COMPLETED_EAST360_PREQUENTIAL_DIRECTION_GATE_DEVELOPMENT',
        metrics=metrics,forecast_comparisons={k:sum(r[k] for r in comparisons.values())
                                             for k in next(iter(comparisons.values()))},
        relative_to_legacy={c:{k:flips(g['legacy4'],g[k]) for k in ARMS[4:]} for c,g in per.items()},
        relative_to_neutral={c:{k:flips(g['neutral'],g[k]) for k in ('learned','lagged','inverted')}
                             for c,g in per.items()},seconds=time.monotonic()-start))
    write(out/'lead_sensitivity.json',report([out/'per_shot.json'],('legacy4','legacy_current4','neutral')))
    print({c:{k:(v['tp'],v['fp']) for k,v in g.items()} for c,g in metrics.items()},flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a = ap.parse_args()
    run(a.reference.resolve(),a.output.resolve())
