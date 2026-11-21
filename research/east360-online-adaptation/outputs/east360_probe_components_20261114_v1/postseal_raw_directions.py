"""Attribute the frozen raw-trajectory readout on the exact difficult pairs."""

from pathlib import Path

import numpy as np

from experiments.risk_aligned_online import commissioning_weak_signal_probe_20261114 as probe
from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.commissioning_roughness_20261110 import coordinates
from experiments.risk_aligned_online.history_gradient_metric_20261102.run import read, sha, write
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import PREPARATION
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent


WORK = Path(__file__).resolve().parent
OBSERVED = WORK.parent/'east360_weak_signal_probe_20261114_v2'


def main():
    p, seal, result = (read(OBSERVED/f'{k}.json') for k in ('protocol','seal','result'))
    model = read(OBSERVED/'fit.json')['probes']['raw_trajectory']
    base, _ = load_parent(DEFAULT_PARENT)
    center, scale = (x.numpy()[list(probe.COLUMNS)] for x in (base.center,base.scale))
    w = np.array(model['coef'][:-1])
    width = probe.WIDTH//probe.BIN
    mapping = read(PREPARATION/'input_map.json')
    col = {c:i for i,c in enumerate(probe.COLUMNS)}
    family = {
        name: [col[c] for c in group]
        for name,group in zip(('magnetic','motion','radiation','loop_voltage'),
                              probe.group.TRUSTED,strict=True)
    }
    ranks = {sid:rank for rank,sid in enumerate(p['order'],1)}

    def contribution(shot, end):
        rank = ranks[shot]
        raw = simple_data.read_shot('EAST',shot,rate_contract='nominal')
        signal = probe.lag.prepare_actual(raw,rank,mapping)
        t = np.array([end])
        raw, valid = coordinates(signal['values'][:,probe.COLUMNS],
                                 signal['available'][:,probe.COLUMNS],center,scale)
        x, available = probe.trajectory(raw,valid,t)
        op, enough = probe.operating(signal,t)
        if not (available & enough).all():
            raise ValueError('matched endpoint lost observations')
        design = np.c_[op.astype(np.float32),x][0]
        z = np.clip((design-model['center'])/model['spread'],-probe.CLIP,probe.CLIP)
        parts = w*z
        key = f'{rank:04d}_{shot}.npz'
        if sha(OBSERVED/'shots'/key) != seal['curves'][key]:
            raise ValueError('original pair observation changed')
        with np.load(OBSERVED/'shots'/key,allow_pickle=False) as f:
            i = int(np.searchsorted(f['ends'],end))
            if f['ends'][i] != end or not f['valid'][i] or f['dose'][i] <= 0.:
                raise ValueError('need the same original eligible decision')
            np.testing.assert_allclose(parts.sum()+model['coef'][-1],f['score'][i,1],atol=1e-6,rtol=1e-6)
        return parts

    definitions = {}
    for row in result['comparisons']['difficult']['pair_rows']:
        pos, neg = (row[k] for k in ('precursor_shot','safe_shot'))
        d = contribution(pos,row['precursor_index'])-contribution(neg,row['safe_index'])
        v = d[3:].reshape(len(probe.COLUMNS),width)
        op, level, shape = float(d[:3].sum()), float(v[:,0].sum()), float(v[:,1:].sum())
        definitions[f'{pos}/{neg}'] = dict(
            lead_ms=row['lead_ms'], margin=float(d.sum()), context=op, level=level, shape=shape,
            family={k:float(v[i].sum()) for k,i in family.items()},
            per_channel=dict(zip((io23.RAW23[c] for c in probe.COLUMNS),v.sum(1).tolist(),strict=True)))
    def count(f):
        return sum(f(d)>0. for d in definitions.values())

    summary = dict(full=count(lambda d:d['margin']),
        remove_family={k:count(lambda d:d['margin']-d['family'][k]) for k in family},
        keep_only_family_and_context={k:count(lambda d:d['context']+d['family'][k]) for k in family},
        remove_level=count(lambda d:d['margin']-d['level']),
        remove_shape=count(lambda d:d['margin']-d['shape']))
    write(WORK/'postseal_raw_directions.json', dict(
        scope='decomposition of fitted score; no refit, no alarm or physical causal intervention',
        reference=str(OBSERVED), pair_results=definitions, **summary))
    print(summary)


if __name__=='__main__':
    main()
