"""Run matched chronological and group-phase feedback on the same EAST head."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_lag_data_20261110 as lag
from . import commissioning_lag_source_20261110 as source
from . import commissioning_lag_state_20261110 as neural
from . import commissioning_transition_east_20261110 as risk
from . import io23, simple_data
from .commissioning_intra_20261110 import CALIBRATION, CAPS, PARENT_ARM, PARENT_RUN, STARTUP, load_frozen
from .commissioning_latent_rate_20261110 import read_parent_curve
from .commissioning_transfer_20261108 import capacity, features
from .commissioning_transition_net_20261110 import FIRST, HORIZON, STRIDE
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, prepare_actual
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .source_anticipation_20261104.static_stream import timing_decomposition


ARMS=('parent',*neural.MODES)


def encode(base,net,signal,center,scale):
    h,_=features(base,signal,'cuda')
    z,m,c,cm=source.shot_coordinates(signal,center,scale)
    q=np.arange(FIRST,len(z)-HORIZON,STRIDE)
    x=source.relation_inputs(z,m,c,cm,np.arange(len(z)))
    r,s,ok=lag.feedback(z,m,q+HORIZON)
    v,audit=neural.replay(net,x,q,r,s,ok)
    for feature in v.values():
        feature[:FIRST]=0.
    return h,v,audit


def run(prior,out,*,graph_memory=False):
    started=time.monotonic()
    configure()
    inputs,parent_head,original,parent_cutoffs,bound=load_frozen()
    base,ckpt=load_parent(DEFAULT_PARENT)
    center,scale=(v.detach().numpy().copy() for v in (base.center,base.scale))
    base.cuda().requires_grad_(False)
    frozen=state_digest(base)
    if ckpt['threshold']!=original or frozen!=inputs['baseline_protocol']['source_state_sha256']:
        raise ValueError('qualified parent changed')
    src,trained=(read(prior/k) for k in ('protocol.json','result.json'))
    bindings={k:sha(prior/k) for k in ('protocol.json','model.npz','result.json')}
    if (src['parent_sha256']!=PARENT_SHA256 or trained['model_sha256']!=bindings['model.npz']
            or src['graph_data_sha256']!=sha(Path(lag.__file__))
            or src['model_code_sha256']!=sha(Path(neural.__file__))
            or src['univariate_code_sha256']!=sha(Path(source.univariate.__file__))
            or src['update_code_sha256']!=sha(Path(source.update_rule.__file__))
            or src['episodes_code_sha256']!=sha(Path(source.episode.__code__.co_filename))
            or src['producer_sha256']!=sha(Path(source.__file__))):
        raise ValueError('trained lag and inference implementation must match')
    memory_sha=None
    if graph_memory:
        from . import commissioning_lag_memory_20261111 as memory

        net=memory.GraphMemory(memory.train_mean(center,scale,trained))
        encode_fn=memory.encode
        memory_sha=sha(Path(memory.__file__))
    else:
        net=neural.LagState(trained['inputs'],trained['edges'])
        with np.load(prior/'model.npz',allow_pickle=False) as f:
            net.load_state_dict({k:torch.from_numpy(f[k]) for k in f.files})
        encode_fn=encode
    net.eval().requires_grad_(False)
    saved_net=state_digest(net)
    ids=inputs['ids']
    original_seals=read(PARENT_RUN/'result.json')['curve_sha256']
    out.mkdir(parents=True)
    (out/'shots').mkdir()
    write(out/'protocol.json',dict(
        scope='EAST360_PAIRED_CROSS_DIAGNOSTIC_PHASE_FEEDBACK_DEVELOPMENT',order=ids,
        startup=STARTUP,source_sha256=PARENT_SHA256,**bound,original=original,parent_thresholds=parent_cutoffs,
        arms=ARMS,first_effect=FIRST+HORIZON+1,source_prior=str(prior),source_assets=bindings,
        memory_sha256=memory_sha,
        fitter_sha256=sha(Path(risk.__file__)),producer_sha256=sha(Path(__file__)),persistent_updates=0))
    if graph_memory:
        with (out/'source_mean.npz').open('xb') as f:
            np.savez(f,mean=net.source_mean.numpy())

    def raw(sid,t):
        if sha(io23.path_for('EAST',sid))!=inputs['hashes'][str(sid)]:
            raise ValueError(f'raw signal changed: {sid}')
        x=simple_data.read_shot('EAST',sid,rate_contract='nominal')
        np.testing.assert_array_equal(x['times'],t)
        return x

    raws,known={},{}
    for rank,sid in enumerate(ids[:STARTUP],1):
        t,_=read_parent_curve(f'{rank:04d}_{sid}.npz',original_seals)
        raws[sid]=raw(sid,t)
        known[sid]=io23.read_outcome('EAST',sid)
    if sum(d for d,_ in known.values())!=8:
        raise ValueError('ended cold prefix must contain 8 disruptive shots')
    cal=risk.calibrate(base,net,raws,known,inputs['mapping']['source_statistics'],center,scale,
                       original,started,encode_fn=encode_fn)
    write(out/'calibration.json',cal)
    coef,fit=risk.fit(base,net,raws,known,ids[:STARTUP],inputs['mapping'],center,scale,
                      original,encode_fn=encode_fn)
    with (out/'readout.npz').open('xb') as f:
        np.savez(f,coef=coef.cpu().numpy())
    assets=['calibration.json','readout.npz']
    if graph_memory:
        assets.append('source_mean.npz')
    local_bind={k:sha(out/k) for k in assets}
    thresholds=cal['cutoffs']
    print(dict(stage='fit20_complete',seconds=time.monotonic()-started),flush=True)
    hashes,checks,dynamics={},{},{}

    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed EAST outcome access')):
        for i,sid in enumerate(ids,1):
            name=f'{i:04d}_{sid}.npz'
            t,parent=read_parent_curve(name,original_seals)
            if i<=STARTUP:
                curves=dict.fromkeys(ARMS,parent)
            else:
                signal=prepare_actual(raw(sid,t),i,inputs['mapping'])
                h,v,dynamics[str(sid)]=encode_fn(base,net,signal,center,scale)
                current=(torch.from_numpy(h).cuda()@parent_head[:-1]+parent_head[-1]).sigmoid().cpu().numpy()
                np.testing.assert_array_equal(current,parent)
                curves=dict(parent=parent,**risk.scores(h,v,coef,neural.MODES))
                n=FIRST+HORIZON+1
                for k in neural.MODES[1:]:
                    np.testing.assert_array_equal(curves[k][:n],curves['fixed'][:n])
                if i in (21,120,360):
                    sub={k:signal[k][:243] for k in ('values','available')}
                    x,y,_=encode_fn(base,net,sub,center,scale)
                    checks[str(sid)]={}
                    for k,p in risk.scores(x,y,coef,neural.MODES).items():
                        full=curves[k][:len(p)]
                        np.testing.assert_allclose(p,full,atol=2e-6,rtol=0)
                        for tau in thresholds.values():
                            np.testing.assert_array_equal(p>=tau,full>=tau)
                        checks[str(sid)][k]=float(np.max(abs(p-full)))
            path=out/'shots'/name
            with path.open('xb') as f:
                np.savez_compressed(f,times=t,**curves)
            hashes[name]=sha(path)
            if i%60==0:
                print(dict(stage='sealed',shots=i,seconds=time.monotonic()-started),flush=True)
    if (state_digest(net)!=saved_net or state_digest(base)!=frozen
            or any(sha(out/k)!=v for k,v in local_bind.items())):
        raise ValueError('persistent fitted state changed within suffix')
    write(out/'seal.json',dict(curve_sha256=hashes,prefix_witnesses=checks,artifacts=local_bind))
    write(out/'dynamics_by_shot.json',dynamics)
    per={c:{k:[] for k in ARMS} for c in CAPS}
    maxima,dflags={k:[] for k in ARMS},[]
    for i,sid in enumerate(ids,1):
        path=out/'shots'/f'{i:04d}_{sid}.npz'
        if sha(path)!=hashes[path.name]:
            raise ValueError('post-seal curve changed')
        d,down=io23.read_outcome('EAST',sid)
        dflags.append(d)
        with np.load(path,allow_pickle=False) as f:
            t=f['times']
            mask=np.round((down-t)*1000.,9)>30 if d else np.ones(len(t),bool)
            for k in ARMS:
                y=f[k]
                maxima[k].append(float(y[mask].max(initial=-np.inf)))
                for c,group in per.items():
                    tau=original if i<=STARTUP else (parent_cutoffs if k=='parent' else thresholds)[c]
                    group[k].append(dict(shot=sid,disruptive=d,down_s=down,**score_curve(t,y,d,down,tau)))
    old=read(CALIBRATION/'per_shot.json')
    if any(group['parent']!=old[c][PARENT_ARM] for c,group in per.items()):
        raise ValueError('archived parent outcomes not reproduced')
    metrics={c:{k:timing_decomposition(v) for k,v in group.items()} for c,group in per.items()}
    n=sum(v['observed_blocks'] for v in dynamics.values())
    write(out/'per_shot.json',per)
    write(out/'result.json',dict(
        status=('COMPLETED_CROSS_DIAGNOSTIC_MEAN_EAST360_DEVELOPMENT' if graph_memory else
                'COMPLETED_CROSS_DIAGNOSTIC_PHASE_EAST360_DEVELOPMENT'),metrics=metrics,
        paired_vs_fixed={c:{k:flips(g['fixed'],g[k]) for k in neural.MODES[1:]} for c,g in per.items()},
        real_vs_shifted={c:flips(g['shifted'],g['real']) for c,g in per.items()},fit=fit,
        observed_blocks=n,updates={k:sum(v['updates'][k] for v in dynamics.values()) for k in neural.MODES[1:]},
        true_alignment_accuracy={k:sum(v['true_order_wins'][k] for v in dynamics.values())/n for k in neural.MODES},
        true_alignment_nll={k:sum(v['true_order_loss_sum'][k] for v in dynamics.values())/n for k in neural.MODES},
        posthoc_capacity_not_deployable={k:capacity(v,dflags,original,startup=STARTUP) for k,v in maxima.items()},
        seconds=time.monotonic()-started))
    print({c:{k:(v['tp'],v['fp']) for k,v in group.items()} for c,group in metrics.items()},flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',required=True,type=Path)
    cli.add_argument('--output',required=True,type=Path)
    cli.add_argument('--graph-memory',action='store_true')
    args=cli.parse_args()
    run(args.source.resolve(),args.output.resolve(),graph_memory=args.graph_memory)
