"""Audit the ID-ordered COH6 stream, freeze six versions on one common suffix, and save resumable state."""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from zero_cold_start import backbone, data, engine, experiments, metrics

CHECKS = (0, 20, 50, 100, 180, 270)
OUT = Path(__file__).with_name('shot_memory360.json')
ARM = 'COH6_Reflective_M100'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def contract(cfg):
    return {
        'source_sha256': digest(data.ARTIFACTS/'jtext_source.pt'),
        'config': json.loads(json.dumps(asdict(cfg))),
        'modules_sha256': {m.__name__: digest(m.__file__) for m in (data, backbone, engine, metrics, experiments)},
    }


def pack(state):
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in asdict(state).items()}


def restore(receipt, cfg, next_shot):
    """Resume this chronological stream only; never warm-start an earlier evaluation with later experience."""
    if receipt['contract'] != contract(cfg) or int(next_shot) <= receipt['last_shot']:
        raise ValueError('source/code/config mismatch or shot is not later than the memory')
    s = dict(receipt['state'])
    for k in ('target_center','target_scale','target_n_eff'):
        s[k] = np.asarray(s[k], dtype=float)
        if s[k].shape != (23,) or not np.all(np.isfinite(s[k])): raise ValueError(f'invalid memory: {k}')
    if np.any(s['target_scale'] <= 0) or np.any(s['target_n_eff'] < 0) or not cfg.mult_bounds[0] <= s['multiplier'] <= cfg.mult_bounds[1]:
        raise ValueError('invalid memory scale, support or multiplier')
    return engine.InterShotState(**s)


def main():
    model, ckpt = backbone.load_source_model(); cfg = experiments.locked_configs(float(ckpt['val_threshold_fpr5']))[ARM]
    c = contract(cfg); assert c['source_sha256'] == '45b17a4d56b710c8ba349af53778c175a46112776ed1525dcd508953bb1d51e7'
    ids = sorted(json.loads((data.ARTIFACTS/'manifests.json').read_text())['east360']); assert len(set(ids)) == len(ids) == 360
    sc, ss = experiments.source_coords(ckpt, cfg.align_mode)
    state = engine.InterShotState.from_source(sc,ss,cfg); snapshots = {0:deepcopy(state)}; witness = {}
    def run(sid, sig, st):
        return engine.run_shot_causal(sid, sig['times'], sig['values'], sig['available'],
                                      lambda x,m:backbone.predict_shot_proba(model,x,m),st,cfg,sc,ss)
    def score(pred, y, down):
        row = metrics.score_shot(pred.times,pred.risk_curve,pred.threshold_curve,y,down,hold_ms=cfg.hold_ms,deadline_ms=30.)
        assert pred.alarm_time == row['alarm_time']
        early = np.round((down-pred.times)*1000.,6)>30. if y else np.ones(len(pred.times),dtype=bool)
        row['timely_peak'] = float(pred.risk_curve[early].max(initial=0.))
        return row
    online=[]; probe={n:[] for n in CHECKS}
    for rank,sid in enumerate(ids,1):
        sig = data.read_signals('EAST',sid)
        live = run(sid,sig,state)
        frozen = {n:run(sid,sig,st) for n,st in snapshots.items()} if rank>270 else {}
        if rank==271:
            resumed=restore(json.loads(json.dumps(dict(contract=c,last_shot=ids[269],state=witness[270]))),cfg,sid)
            r=run(sid,sig,resumed); assert np.array_equal(live.risk_curve,r.risk_curve) and np.array_equal(live.threshold_curve,r.threshold_curve) and live.alarm_time==r.alarm_time
        y, down = data.read_outcome('EAST',sid)  # all first alarms for this shot are sealed
        online.append(score(live,y,down))
        for n,r in frozen.items(): probe[n].append(score(r,y,down))
        engine.reflect_and_update(live,sig['values'],sig['available'],y,down,state,cfg,sc,ss)
        if rank in CHECKS: snapshots[rank]=deepcopy(state); witness[rank]=pack(state)
    archive = json.loads((data.ARTIFACTS.parent/'results/dev360_locked_comparison.json').read_text())
    reference = archive['ordering']['chrono']
    def summarize(rows):
        m=metrics.summarize_stream(rows,checkpoints=()); out={k:m[k] for k in ('n_disruptive','n_normal','tp_timely','tardy','miss','fp','tn','timely_tpr','fpr','roc_auc')}
        budget=(m['n_normal']-1)//5; neg=sorted((r['timely_peak'] for r in rows if not r['disruptive']),reverse=True)
        thresh=np.nextafter(neg[budget], np.inf); out['max_fp']=budget
        out['diagnostic_fixed_score_capacity_tp']=sum(r['timely_peak']>=thresh for r in rows if r['disruptive'])
        out['diagnostic_fixed_score_capacity_fp']=sum(r['timely_peak']>=thresh for r in rows if not r['disruptive'])
        return {k:int(v) if isinstance(v,np.integer) else v for k,v in out.items()}
    actual=summarize(online)
    keys=('tp_timely','tardy','miss','fp','tn')
    canonical=experiments.run_stream(ids,experiments.load_cache(ids),model,ckpt,cfg,checkpoints=())
    assert all(actual[k]==canonical[k] for k in keys)
    archived_hash=archive['provenance']['modules_sha256']['engine.py']
    if archived_hash == c['modules_sha256'][engine.__name__]: assert all(actual[k]==reference[k] for k in keys)
    witness[0]=pack(engine.InterShotState.from_source(sc,ss,cfg))
    assert all(pack(st)==witness[n] for n,st in snapshots.items())
    result = dict(arm=ARM, contract=c, order='numeric ID ascending; recorded signals, no PCS or EAST AdvancedTime fields used',
                  last_shot=ids[-1], state=pack(state), all360_prequential=actual,
                  archive_comparator={'engine_sha256':archived_hash, **{k:reference[k] for k in keys}}, probe_ids=ids[270:],
                  probe_note='Earlier model snapshots are frozen on the same already-exposed dev suffix; distinct live branch continues updates, never shared into the snapshots. Capacity is post-hoc only.',
                  common90={n:dict(M=snapshots[n].multiplier,**summarize(rows)) for n,rows in probe.items()},
                  checks=['All360 primary confusion counts equal canonical replay of this code version; older engine counts are listed separately','Every probe snapshot identical before/after all 90 shots',
                          'Checkpoint270 JSON round trip reproduces next actual shot (risk, cutoff, first alarm)'])
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n'); loaded=json.loads(OUT.read_text());assert pack(restore(loaded,cfg,ids[-1]+1)) == pack(state)
    try: restore(loaded,cfg,ids[0])
    except ValueError: pass
    else: raise AssertionError('loading future experience for an earlier shot must fail')
    print('Full360:',actual);print('Checkpoints:',result['common90']);print('Final memory saved; next-shot restore and back-scoring guard passed.')


if __name__=='__main__': main()
