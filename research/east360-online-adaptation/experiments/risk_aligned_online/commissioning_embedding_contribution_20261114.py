"""Factor the archived, supervised embedding into measured and mask paths."""

import argparse
from copy import deepcopy
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from . import commissioning_target_separability_20261114 as target, io23, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, write
from .lifecycle_stream_20261025.audit import flips
from .operating_point_audit import best_at_count, peak_for_shot, timely_auc


INPUT = io23.ROOT/'outputs/east360_target_separability_20261114_v3_location'
ARMS = ('head_only', 'measured', 'mask_and_bias', 'full')


def mix_embedding(w, b, target_w, target_b, arm):
    if arm not in ARMS:
        raise ValueError('unknown embedding component')
    if w.shape!=(64, 368) or target_w.shape!=w.shape or b.shape!=(64,) or target_b.shape!=b.shape:
        raise ValueError('requires 8 data-and-mask rows per patch')
    if arm=='head_only':
        return w, b
    if arm=='full':
        return target_w, target_b
    use = torch.arange(368, device=w.device)%46 < 23
    if arm=='measured':
        return torch.where(use, target_w, w), b
    return torch.where(use, w, target_w), target_b


def copies():
    base, _ = target.load_parent(target.DEFAULT_PARENT)
    trained = torch.load(INPUT/'embedding_and_head.pt', map_location='cpu', weights_only=True)['state']
    reference = base.state_dict()
    common = {'embedding.weight', 'embedding.bias', 'risk_head.weight', 'risk_head.bias'}
    changed = {k for k, t in trained.items() if not torch.equal(t, reference[k])}
    if changed!=common:
        raise ValueError('archived embedding-only trainable scope differs')
    models = []
    for k in ARMS:
        m = deepcopy(base)
        w, b = mix_embedding(reference['embedding.weight'], reference['embedding.bias'],
            trained['embedding.weight'], trained['embedding.bias'], k)
        m.load_state_dict(dict(reference, **{'embedding.weight': w, 'embedding.bias': b,
                                             'risk_head.weight': trained['risk_head.weight'],
                                             'risk_head.bias': trained['risk_head.bias']}))
        models.append(m.cuda())
    return models


def run(out):
    start = monotonic()
    p, seal, selected = (read(INPUT/f'{n}.json') for n in ('protocol', 'seal', 'selected'))
    order, files = seal['order'], seal['files']
    if order!=p['order'] or len(order)!=360 or target.STRIDE!=seal['score_stride']:
        raise ValueError('requires the exact completed location control')
    out.mkdir()
    (out/'shots').mkdir()
    write(out/'protocol.json', dict(source_experiment=str(INPUT), order=order, arms=ARMS,
        label_scope='sealed after 360; no optimization; all use the same archived supervised risk head',
        clock=seal['score_stride'], thresholds=selected['fits']['embedding_and_head']['calibration']))
    models = copies()
    max_full = 0.
    for sid, f in zip(order, files, strict=True):
        s = target.load_signal(sid)
        _, a = target.curves(models, s, stride=target.STRIDE)
        with np.load(INPUT/'shots'/f, allow_pickle=False) as old:
            z = old['embedding_and_head'][::target.STRIDE]
        np.testing.assert_allclose(a[-1], z, rtol=0., atol=3e-6)
        max_full = max(max_full,float(abs(a[-1]-z).max()))
        with (out/'shots'/f).open('xb') as h:
            np.save(h, a, allow_pickle=False)
    write(out/'seal.json', dict(order=order, files=files, max_full_probability_error=max_full))
    score(out, order, files, selected)
    print(dict(stage='embedding_contribution_complete', seconds=monotonic()-start), flush=True)


def score(out, order, files, selected):
    cal = selected['fits']['embedding_and_head']['calibration']
    ref = read(INPUT/'per_shot.json')
    counts = {b:{k:[] for k in ARMS} for b in cal}
    ranking = {k:[] for k in ARMS}
    for i, (sid, f) in enumerate(zip(order,files,strict=True)):
        s = target.load_signal(sid)
        y, d = io23.read_outcome('EAST',sid)
        a = np.load(out/'shots'/f, allow_pickle=False)
        times = s['times'][::target.STRIDE]
        for k,z in zip(ARMS,a,strict=True):
            ranking[k].append(dict(disruptive=y, peak=peak_for_shot(times,z,y,d)))
        for b,v in cal.items():
            for k,z in zip(ARMS,a,strict=True):
                row = dict(shot=sid,disruptive=y,down_s=d,**score_curve(times,z,y,d,v['threshold']))
                if k=='full' and row!=ref[b]['embedding_and_head'][i]:
                    raise ValueError('full embedding comparator must reproduce its own recorded decision')
                counts[b][k].append(row)
    write(out/'per_shot.json', counts)
    by_cap = {c:{k:lead.summarize(v) for k,v in r.items()} for c,r in counts.items()}
    primary = counts['0.2']
    write(out/'result.json',dict(metrics=by_cap,
        change_from_head_only={k:flips(primary['head_only'],primary[k]) for k in ARMS[1:]},
        distinct_frozen_thresholds=False, true_intra_updates=False))
    budget_checks(out, ranking)
    print({c:{k:(t['by_deadline']['30']['tp'],t['fp']) for k,t in v.items()}
           for c,v in by_cap.items()}, flush=True)


def budget_checks(out, ranking):
    caps = {str(n):{k:{i:r[i] for i in ('tp','fp')} for k,v in ranking.items()
                    for r in [best_at_count(v,n)]} for n in (18,27,36)}
    write(out/'postseal_ranking.json',
          dict(caps=caps,auc={k:timely_auc(v) for k,v in ranking.items()},
               scope='retrospective equal-FP ranking only; no deployable threshold'))


def rescore_ranking(out):
    s = read(out/'seal.json')
    ranking = {k:[] for k in ARMS}
    for sid, fn in zip(s['order'],s['files'],strict=True):
        x = target.load_signal(sid)
        times = x['times'][::target.STRIDE]
        y,d = io23.read_outcome('EAST',sid)
        v = np.load(out/'shots'/fn, allow_pickle=False)
        for k,z in zip(ARMS,v,strict=True):
            ranking[k].append(dict(disruptive=y,peak=peak_for_shot(times,z,y,d)))
    budget_checks(out,ranking)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('--rank', action='store_true', help='score already-sealed curves at equal false-alarm caps')
    args = cli.parse_args()
    (rescore_ranking if args.rank else run)(args.output.resolve())
