"""Offline curve-amplitude bounds; these are NOT achieved online rescues."""
import argparse
from pathlib import Path

import numpy as np


def describe_shot(times,cold,candidate,reverse,threshold,flag,down,mutable_start):
    times,cold,candidate,reverse=[np.asarray(x,dtype=np.float64) for x in (times,cold,candidate,reverse)]
    if (times.ndim!=1 or not len(times) or any(x.shape!=times.shape for x in (cold,candidate,reverse))
            or not all(np.isfinite(x).all() for x in (times,cold,candidate,reverse))
            or (np.diff(times)<=0).any() or not 0<threshold<1
            or any(((x<0)|(x>1)).any() for x in (cold,candidate,reverse))
            or not isinstance(mutable_start,int) or not 0<=mutable_start<=len(times)
            or (flag and (not isinstance(down,(float,int,np.floating,np.integer)) or not np.isfinite(down)))):
        raise ValueError('finite increasing times, probabilities and bounded start required')
    if (not np.array_equal(cold[:mutable_start],candidate[:mutable_start])
            or not np.array_equal(cold[:mutable_start],reverse[:mutable_start])):
        raise ValueError('immutable prefix changed')
    # A safe shot has no disruption time: do not manufacture one for validation.
    lead=np.round((down-times)*1000,9) if flag else None
    valid=lead>=-1e-6 if flag else np.ones(len(times),bool)
    if not valid.any():raise ValueError('no valid observations')
    timely=valid&(lead>30) if flag else valid
    mutable=np.arange(len(times))>=mutable_start

    def decision(values):
        cross=np.flatnonzero(valid&(values>=threshold));alarm=bool(len(cross))
        warning=float(lead[cross[0]]) if flag and alarm else None
        return dict(is_disrupt=bool(flag),alarm=alarm,lead_ms=warning,timely=bool(warning is not None and warning>30))

    b=decision(cold);a=decision(candidate)
    wrong=not b['timely'] if flag else b['alarm']
    delta=candidate-cold;rev=reverse-cold;amplitude=np.abs(delta)
    lower=np.clip(cold-amplitude,0,1);upper=np.clip(cold+amplitude,0,1)
    correctable=bool(np.any(timely&(upper>=threshold))) if flag else bool(np.all(lower[valid]<threshold))
    d=delta[valid];r=rev[valid];den=float(np.linalg.norm(d)*np.linalg.norm(r))
    peak_delta=float(candidate[timely].max()-cold[timely].max()) if timely.any() else None
    return dict(cold=b,candidate=a,cold_wrong=bool(wrong),
        envelope_correctable=bool(wrong and correctable),
        immutable_alarm=bool(np.any(valid&~mutable&(cold>=threshold))),
        mutable_timely_points=int(np.sum(timely&mutable)) if flag else None,
        operational_peak_delta=peak_delta,
        cold_operational_peak=float(cold[timely].max()) if timely.any() else None,
        max_abs_change=float(amplitude[valid].max()),
        max_real_reverse_difference=float(np.abs(candidate[valid]-reverse[valid]).max()),
        real_reverse_cosine=float(d@r/den) if den>0 else None)


def run(root,output):
    # Outcome access below is intentionally post-hoc and is never an online input.
    from .. import io23
    from ..context_memory_20261028.checkpoint import read
    from ..gradient_split_20261029.audit import count
    from ..simple_online import _durable_json
    from ..simple_train import sha

    if output.exists() or output.absolute().parent!=io23.ROOT/'probe_outputs':
        raise ValueError('new diagnostic output in worktree probe_outputs required')
    p=read(root,'protocol.json');done=read(root,'complete.json');audit=read(root,'audit.json')
    declared=read(root,'per_shot.json');receipts=read(root,'receipts.json')
    if (not done['complete'] or not audit['complete'] or audit['complete_sha256']!=sha(root/'complete.json')
            or len(p['order'])!=360 or len(set(p['order']))!=360 or len(receipts)!=360):
        raise ValueError('audited full360 required')
    for path,h in p['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('original stream implementation changed')
    ref=Path(p['reference']);hashes=read(ref,'input_hashes.json')
    rows=[];records={k:[] for k in p['arms']}
    for i,sid in enumerate(p['order']):
        name=f'{i+1:04d}_{sid}';path=root/'shots'/f'{name}.npz';seal=read(root/'shots',f'{name}_seal.json')
        if (seal!=receipts[i] or seal['shot']!=sid or seal['index']!=i+1 or seal['outcome_used']
                or seal['sha256']!=sha(path) or sha(io23.path_for('EAST',sid))!=hashes[str(sid)]):
            raise ValueError('curve seal or raw input changed')
        flag,down=io23.read_outcome('EAST',sid)
        with np.load(path) as z:
            start=len(z['times']) if i<20 else min(256,len(z['times']))
            row=dict(shot=sid,index=i+1,flag=flag,arms={})
            for k in p['arms'][1:]:
                r=describe_shot(z['times'],z['cold'],z[k],z['reverse_target'],p['threshold'],flag,down,start)
                for label,key in (('cold','cold'),(k,'candidate')):
                    dec=dict(shot=sid,**r[key])
                    if dec!=declared[label][i]:raise ValueError('actual shot decision changed')
                row['arms'][k]=r;records[k].append(dict(shot=sid,**r['candidate']))
            records['cold'].append(dict(shot=sid,**r['cold']));rows.append(row)
    for arm,data in records.items():
        measured=count(data)
        for key,value in measured.items():
            if value!=done['results'][arm]['full360'][key]:raise ValueError('fullstream recount differs')
    summary={}
    for arm in p['arms'][1:]:
        summary[arm]={}
        for split,subset in (('full360',rows),('cold20',rows[:20]),('suffix340',rows[20:])):
            stats={}
            for flag,label in ((True,'disruptive'),(False,'safe')):
                group=[r for r in subset if r['flag']==flag];wrong=[r for r in group if r['arms'][arm]['cold_wrong']]
                changes=[r['arms'][arm]['operational_peak_delta'] for r in group]
                changes=[x for x in changes if x is not None]
                stats[label]=dict(n=len(group),cold_wrong=len(wrong),
                    envelope_correctable=[r['shot'] for r in wrong if r['arms'][arm]['envelope_correctable']],
                    immutable_alarm=[r['shot'] for r in wrong if r['arms'][arm]['immutable_alarm']],
                    no_timely_mutable_point=[r['shot'] for r in wrong if flag and r['arms'][arm]['mutable_timely_points']==0],
                    median_operational_peak_delta=float(np.median(changes)) if changes else None)
            cosine=[r['arms'][arm]['real_reverse_cosine'] for r in subset]
            cosine=[x for x in cosine if x is not None]
            stats['median_real_reverse_cosine']=float(np.median(cosine)) if cosine else None
            stats['median_shot_max_abs_change']=float(np.median([r['arms'][arm]['max_abs_change'] for r in subset]))
            summary[arm][split]=stats
    result=dict(complete=True,scope='POSTHOC_CURVE_AMPLITUDE_BOUND_NOT_AN_ONLINE_POLICY',
        source=str(root),complete_sha256=sha(root/'complete.json'),audit_sha256=sha(root/'audit.json'),
        code_sha256=sha(Path(__file__)),plan_sha256=sha(io23.ROOT/'ATTENTION_PROTECTED_MARGIN_DIAGNOSTIC.md'),
        counts_recomputed=True,summary=summary,per_shot=rows,independent_confirmation=False,
        warning='Envelope corrections use known outcomes and arbitrary pointwise signs; not achieved rescues or a neural-step guarantee.')
    _durable_json(output,result)
    print({k:v['suffix340'] for k,v in summary.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.root,a.output)
