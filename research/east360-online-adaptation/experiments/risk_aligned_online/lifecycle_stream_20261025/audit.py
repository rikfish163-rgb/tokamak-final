"""Independent post-run scoring from sealed curves; never used by learning."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from .. import io23


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(times,scores,disruptive,down,threshold):
    valid=np.ones(len(times),bool) if not disruptive else (down-times)*1000>=-1e-6
    where=np.flatnonzero(valid & (scores>=threshold))
    lead=float(round((down-times[where[0]])*1000,9)) if disruptive and len(where) else None
    return dict(alarm=bool(len(where)),lead_ms=lead,
                timely=bool(lead is not None and lead>30))


def counts(rows):
    nd=sum(r['disruptive'] for r in rows);ns=len(rows)-nd
    tp=sum(r['timely'] for r in rows)
    fp=sum(r['alarm'] and not r['disruptive'] for r in rows)
    return dict(n=len(rows),n_disrupt=nd,n_safe=ns,tp=tp,fp=fp,
                tpr=tp/nd if nd else None,fpr=fp/ns if ns else None,
                J=tp/nd-fp/ns if nd and ns else None,
                early_over3000=sum(r['timely'] and r['lead_ms']>3000 for r in rows))


def flips(reference,candidate):
    result={k:[] for k in ('rescued_D','lost_D','removed_FP','added_FP')}
    for a,b in zip(reference,candidate,strict=True):
        assert a['shot']==b['shot'] and a['disruptive']==b['disruptive']
        if a['disruptive']:
            if b['timely'] and not a['timely']:result['rescued_D'].append(a['shot'])
            if a['timely'] and not b['timely']:result['lost_D'].append(a['shot'])
        else:
            if a['alarm'] and not b['alarm']:result['removed_FP'].append(a['shot'])
            if b['alarm'] and not a['alarm']:result['added_FP'].append(a['shot'])
    return result


def audit(root):
    completed=json.loads((root/'completed.json').read_text())
    if completed['status']!='COMPLETED':raise ValueError('complete stream required')
    p=json.loads((root/'protocol.json').read_text());arms=p['arms']
    order=json.loads((root/'order.json').read_text());ids=[r['shot'] for r in order]
    assert len(ids)==len(set(ids))==360 and set(ids)==set(p['contract']['shot_ids'])
    assert p['warmup_in_primary_score'] and p['warmup']==20 and p['external_east_labels']==0
    assert digest(Path(p['source_checkpoint']))==p['source_sha256']
    assert digest(root/'order.json')==p['order_sha256']
    with zipfile.ZipFile(root/'code_snapshot.zip') as z:
        for path,expected in p['code_sha256'].items():
            assert hashlib.sha256(z.read(path)).hexdigest()==expected
    mapping=json.loads((root/'input_map.json').read_text())
    assert mapping['calibration_ids']==ids[:20]
    training_ids=set(io23.source_roles()['train'])
    events=json.loads((root/'source_replay.json').read_text())
    assert all(e['shot'] in training_ids for e in events)
    rows={k:[] for k in arms};maxima={k:[] for k in arms}
    unchanged=0;accepted={k:0 for k in ('inter','both')}
    input_hash=json.loads((root/'input_hashes.json').read_text())
    for index,sid in enumerate(ids,1):
        prefix=root/'shots'/f'{index:04d}_{sid}'
        sealed=json.loads(Path(str(prefix)+'_seal.json').read_text())
        path=prefix.with_suffix('.npz')
        assert digest(path)==sealed['sha256'] and not sealed['outcome_available']
        assert sealed['index']==index and sealed['model_version_before_shot']==index-1
        assert digest(io23.path_for('EAST',sid))==input_hash[str(sid)]
        disruptive,down=io23.read_outcome('EAST',sid)
        receipt=json.loads(Path(str(prefix)+'_audit.json').read_text())
        for k in accepted:
            u=receipt['updates'].get(k)
            if u:
                assert set(u['history_ids'])<=set(ids[:index-1])
                if u['accepted']:
                    accepted[k]+=1
                    assert all(u['guard_after'][g]<=v+1e-7 for g,v in u['guard_before'].items())
                else:assert all(v==0 for v in u['retained_delta'].values())
        with np.load(path) as f:
            times=f['times']
            if index<=20:
                assert all(np.array_equal(f[k],f['source']) for k in arms)
                unchanged+=1
            eligible=np.ones(len(times),bool) if not disruptive else np.round((down-times)*1000,9)>30
            for k in arms:
                s=f[k];assert s.shape==times.shape and np.isfinite(s).all()
                rows[k].append(dict(shot=sid,disruptive=disruptive,
                                   **classify(times,s,disruptive,down,p['threshold'])))
                maxima[k].append(float(s[eligible].max()) if eligible.any() else -np.inf)
    metrics={k:counts(r) for k,r in rows.items()}
    for k,m in metrics.items():
        reported=completed['metrics'][k]['full360']
        for field in ('tp','fp','n_disrupt','n_safe','tpr','fpr','J','early_over3000'):
            assert m[field]==reported[field],(k,field,m[field],reported[field])
    pairs={b+'__minus__'+a:flips(rows[a],rows[b]) for a,b in (
        ('source','aligned'),('aligned','cold'),('cold','intra'),('cold','inter'),
        ('intra','both'),('both_shadow','both'),('intra_reverse','intra'))}
    workpoints={};d=np.array([r['disruptive'] for r in rows['source']])
    for k,v in maxima.items():
        v=np.asarray(v);curve=[]
        for cap in (.1,.15,.2):
            choices=[]
            for threshold in np.unique(np.r_[v,np.nextafter(v, np.inf),np.nextafter(1.,2.)]):
                if not np.isfinite(threshold):continue
                hit=v>=threshold;tpr=float(hit[d].mean());fpr=float(hit[~d].mean())
                if fpr<=cap:choices.append((tpr,-fpr,threshold))
            tpr,negfpr,threshold=max(choices)
            curve.append(dict(cap=cap,tpr=tpr,fpr=-negfpr,threshold=threshold,
                              scope='POSTHOC_DEVELOPMENT_WORKPOINT_NOT_DEPLOYED'))
        workpoints[k]=curve
    output=dict(complete360=True,identical_cold_start_predictions=unchanged,
        prediction_hashes_verified=360,source_checkpoint_unchanged=True,code_archive_verified=True,
        outcome_reconstruction_independent_of_trainer=True,metrics=metrics,flips=pairs,
        threshold=p['threshold'],workpoints=workpoints,accepted_updates=accepted,
        warmup=counts(rows['source'][:20]),source_event_intervals=len(events),
        included_source_events=sum(e['status']=='included' for e in events),
        total_seconds=completed['seconds'],real_time_latency_verified=False,
        independent_future_validation=False)
    with (root/'independent_audit.json').open('x') as f:json.dump(output,f,ensure_ascii=False,indent=2)
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    audit(parser.parse_args().root)
