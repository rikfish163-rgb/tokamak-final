from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online import commissioning_lag_guard_20261111 as a
from experiments.risk_aligned_online import commissioning_lag_mixer_20261111 as lag
from experiments.risk_aligned_online import commissioning_phase_innovation_20261111 as phase
from experiments.risk_aligned_online.hdf5_score_recheck_20261103 import score_curve
from experiments.risk_aligned_online.history_gradient_metric_20261102.run import read, sha, write

out = Path('probe_outputs/commissioning_phase_innovation_east360_20261111_v2')
p, seal, summary = (read(out/k) for k in ('protocol.json','seal.json','result.json'))
guard = Path(p['previous_guard'])
gp, gs, fitted = (read(guard/k) for k in ('protocol.json','seal.json','teacher.json'))
previous = Path(gp['previous_teacher'])
pp = read(previous/'protocol.json')
lp = read(Path(pp['previous_lag'])/'protocol.json')
coords = next(Path(f).parent for f in p['assets'] if Path(f).parent.name.startswith('commissioning_local_cross_response_'))
lag.configure()
lock = lag.load_inputs()
parent, _, args, assets = lag.parent._frozen(Path(lp['reference']), coords, lock)
assert p['assets'] == assets and sha(guard/'teacher.json') == gp['teacher_sha256']
assert sha(guard/'seal.json') == p['previous_seal'] and len(seal['curve_sha256']) == 360
decoder = lag._bind_forecast(Path(lp['source_decoder']).parent, parent)
base = args[0]
tau = p['calibration']['0.2']
pairs = ((a.PATH, phase.PHASE), (phase.SHIFT, phase.PHASE))
labelsets = summary['paired']['0.2']
affected = set().union(*(set(s) for pair in pairs for s in labelsets['__to__'.join(pair)].values()))
fitted_path = torch.as_tensor(fitted['path'], device='cuda', dtype=base.center.dtype)
original_patch = lag.patch_input
cols = list(lag.CHANNELS)
results = []
for sid in sorted(affected):
    rank = p['order'].index(sid)+1
    assert rank > p['startup'] and sha(io23.path_for('EAST', sid)) == lock['hashes'][str(sid)]
    name = f'{rank:04d}_{sid}.npz'
    f, g = (v/'shots'/name for v in (out, guard))
    assert sha(f) == seal['curve_sha256'][name] and sha(g) == gs['curve_sha256'][name]
    with np.load(f, allow_pickle=False) as npz, np.load(g, allow_pickle=False) as old:
        saved = {k:npz[k] for k in (phase.PHASE, phase.SHIFT)} | {a.PATH:old[a.PATH]}
        q_saved = npz['release']
    raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
    signal = lag.prepare_actual(raw, rank, lock['mapping'])
    released = phase._release(raw, signal, args, decoder)
    q, h, z, beta = (released[k] for k in ('q','h','z','beta'))
    np.testing.assert_array_equal(q+lag.H+1, q_saved)
    matrices = phase._states(base, signal, released, decoder, fitted_path, np.arange(len(q)))
    tr, _, _, _ = a._track(signal, args)
    q0, _, prior, target = a.teacher._mature(tr, args[2][2], args[2][3])
    np.testing.assert_array_equal(q0, q)
    prior_basis, _ = a._bases(base, signal, q, prior, target, decoder)
    steps = {a.PATH:-lag.RADIUS*a.signed_direction(*prior_basis, released['key'], fitted_path),
             phase.PHASE:matrices[1], phase.SHIFT:matrices[2]}
    d,down=io23.read_outcome('EAST',sid)
    slot,_=lag.delayed_slots(q,np.arange(len(z)))
    use=np.flatnonzero(slot>0)
    for lhs,rhs in pairs:
        dose=[]
        raw_dz=np.zeros((2,len(z)),np.float32)
        fair_dz=np.zeros_like(raw_dz)
        with torch.no_grad():
            for sel in np.array_split(use,(len(use)+255)//256):
                x,m=lag._windows(base,signal,sel)
                th=[steps[k][slot[sel]-1] for k in (lhs,rhs)]
                ht=torch.as_tensor(h[sel],device='cuda')

                def equalize(patches,_theta,idx):
                    delta=torch.stack([original_patch(patches,t)-patches for t in th])
                    v=delta.reshape(2,len(sel),lag.W,46)[...,cols]
                    rms=v.square().mean((2,3)).sqrt()
                    common=rms.min(0).values
                    scale=common/rms[idx].clamp_min(1e-12)
                    if idx==0:
                        dose.append(torch.stack((*rms,common)).cpu().numpy())
                    effective=patches+scale.reshape(len(sel),1,1)*delta[idx]
                    actual=(effective-patches).reshape(len(sel),lag.W,46)[...,cols]
                    torch.testing.assert_close(actual.square().mean((1,2)).sqrt(),common,
                                               atol=2e-6,rtol=.002)
                    return effective

                for i,t in enumerate(th):
                    feat=lag._features(base,x,m,t)
                    raw_dz[i,sel]=((feat-ht)*beta).sum(-1).cpu().numpy()
                    with patch.object(lag,'patch_input',lambda p,w,i=i:equalize(p,w,i)):
                        feat=lag._features(base,x,m,t)
                    fair_dz[i,sel]=((feat-ht)*beta).sum(-1).cpu().numpy()
        def probability(dz):
            return torch.as_tensor(z[None]+dz,device='cuda').sigmoid().cpu().numpy()
        raw_prob,fair_prob=map(probability,(raw_dz,fair_dz))
        for j,k in enumerate((lhs,rhs)):
            np.testing.assert_allclose(raw_prob[j],saved[k],atol=3e-6,rtol=0.)
            np.testing.assert_array_equal(raw_prob[j]>=tau,saved[k]>=tau)
        results.append(dict(shot=sid,pair=[lhs,rhs],disruptive=d,
            original=[score_curve(raw['times'],v,d,down,tau) for v in raw_prob],
            equal_rms=[score_curve(raw['times'],v,d,down,tau) for v in fair_prob],
            median_rms=np.median(np.concatenate(dose,axis=1),axis=1).tolist()))
write(out/'case_equal_input_rms_postseal.json',dict(
    scope='POSTSEAL_PHASE_CASE_ATTRIBUTION_NOT_FULL_STREAM', seal_sha256=sha(out/'seal.json'),
    cases=results))
for x in results:
    print(x,flush=True)
