"""Raw, parameter-free current change: matched opportunity vs whole-shot false alarms."""
import json
from pathlib import Path
import sys
from hashlib import sha256

import h5py
import numpy as np

from paired_forensics import classify, cut_at_budget, dump, digest, elapsed, endpoint, score_counts, data


def verify_stream(parent_first):
    """Independent physical-clock run, without current labels until after both firsts are sealed."""
    tn_peaks=[]; fp=0; cutoff=np.inf; firsts={}; digest_log=sha256(); observed=[]
    for i,sid in enumerate(data._east360()):
        if i and i%90==0:
            allowed=(len(tn_peaks)+fp-1)//5-fp
            cutoff=np.inf if allowed<0 else np.nextafter(sorted(tn_peaks,reverse=True)[allowed],np.inf)
        with h5py.File(data._path('EAST',sid),'r') as f:
            ds=f['data/ip']; x=ds[:].astype('float64'); rate=float(ds.attrs['SampleRate']); start=float(ds.attrs['StartTime'])
        v=np.lib.stride_tricks.sliding_window_view(x,128); ok=np.isfinite(v).all(1)&(v[:,0]!=0)
        loss=np.full(len(v),-np.inf); loss[ok]=(v[ok,0]-v[ok,-1])/abs(v[ok,0]); hits=np.flatnonzero(loss>=cutoff)
        extra=None if not len(hits) else start+(127+int(hits[0]))/rate
        f=parent_first[sid]; actual=min(f,extra) if f is not None and extra is not None else f if extra is None else extra
        firsts[sid]=[extra,actual,None if not len(hits) else float(loss[hits[0]])]
        digest_log.update(repr((sid,cutoff,f,extra,actual)).encode())
        y,down=data.read_outcome('EAST',sid)
        outcome=('TN' if actual is None else 'FP') if not y else 'MISS' if actual is None or actual>down+1e-9 else 'TP_timely' if round((down-actual)*1000,6)>30 else 'TARDY'
        observed.append(outcome)
        if not y:
            if f is not None: fp+=1
            else: tn_peaks.append(float(loss.max()))
    return firsts,digest_log.hexdigest(),score_counts(observed)


def run(tmp):
    ref=json.loads(Path(__file__).with_name('paired_forensics.json').read_text()); rows=ref['rows']; curves={}; evidence={}; safe=[]
    for r in rows:
        with np.load(tmp/f"{r['shot']}.npz") as d:
            t=d['t']; v=d['v'][:,9].astype(float); a=d['a'][:,9]; y=r['down']; g=np.full(len(t),-np.inf)
            valid=np.convolve(a.astype(int),np.ones(128,int),'valid')==128; valid &= abs(v[:-127])>0
            ix=np.flatnonzero(valid); g[ix+127]=(v[ix]-v[ix+127])/abs(v[ix]); curves[r['shot']]=g
            elig=np.ones(len(t),bool) if y is None else np.round(1000*(y-t),6)>30
            win=(t<t[-1]-.05) if y is None else (((t-t[0])*1000<=100)&(t<y-.25)); x=v[win&a]
            assert len(x)>1 and np.isfinite(g[elig].max())
            if r['rank']>1:
                safe.append([float(x.min()),float(np.subtract(*np.percentile(x,[75,25])))])
            evidence[r['shot']]=dict(peak=float(g[elig].max()),full_peak=float(g.max()),parent_first=r['first'])
    frozen=np.array(safe); upper=1e5+10*2065.7734375
    checks=dict(later_feedback_windows=len(frozen),all_above=int((frozen[:,0]>upper).sum()),min_amps=float(frozen[:,0].min()),
                raw_iqr_positive=int((frozen[:,1]>0).sum()),initial_clip_upper_amps=upper)
    def rule(past):
        N=[r for r in past if r['down'] is None]; tn=[r for r in N if r['kind']=='TN']; budget=(len(N)-1)//5
        room=budget-(len(N)-len(tn)); return np.inf if room<0 else np.nextafter(sorted([evidence[r['shot']]['full_peak'] for r in tn],reverse=True)[room],np.inf)
    N=[r for r in rows if r['down'] is None]; D=[r for r in rows if r['down'] is not None]
    alone=cut_at_budget([evidence[r['shot']]['peak'] for r in N]); either=rule(rows)
    hits=lambda r,cut:evidence[r['shot']]['peak']>=cut
    oracle=dict(alone=[sum(hits(r,alone) for r in D),sum(hits(r,alone) for r in N)],alone_cut=alone,either_cut=either,
        either=[sum(r['kind']=='TP_timely' or hits(r,either) for r in D),sum(r['kind']=='FP' or hits(r,either) for r in N)],
        gained=[r['shot'] for r in D if r['kind']!='TP_timely' and hits(r,either)])
    issued=[]; blocks=[]; cut=None
    for r in rows:
        b=r['block']; i=r['rank']-1
        if i%90==0: cut=rule(rows[:i]) if b else np.inf; blocks.append(dict(cut=None if not np.isfinite(cut) else cut))
        sid=r['shot']; ix=np.flatnonzero(curves[sid]>=cut) if np.isfinite(cut) else []
        first=None if not len(ix) else r['start']+int(ix[0])/r['rate']; old=r['first']
        union=old if first is None else first if old is None else min(first,old)
        issued.append(classify(union,r['down'])); evidence[sid].update(drop_first=first,union_first=union)
    for i,k in enumerate(blocks): k['counts']=score_counts(issued[90*i:90*i+90])
    by={r['shot']:r for r in rows}; matched=[]
    verify,stamp,graded=verify_stream({s:r['parent_first'] for s,r in evidence.items()}); assert graded==score_counts(issued)
    for s,v in verify.items():
        assert v[:2]==[evidence[s]['drop_first'],evidence[s]['union_first']]
        evidence[s]['first_drop_fraction']=v[2]
    gained=[r for r,k in zip(rows,issued) if r['kind'] in ('TARDY','MISS') and k=='TP_timely']
    added=[r for r,k in zip(rows,issued) if r['kind']=='TN' and k=='FP']; near=[]
    for p in gained:
        for q in added:
            if p['block']!=q['block']: continue
            time=lambda r:evidence[r['shot']]['drop_first']-r['start']
            di=abs(np.log(p['ip']/q['ip'])); dt=abs(np.log(time(p)/time(q)))
            if di<=np.log(1.25) and dt<=np.log(2): near.append((di/np.log(1.25)+dt/np.log(2)+abs(p['rank']-q['rank'])/90,p['shot'],q['shot']))
    for pair in ref['paired']['Derror_N']['details']:
        p,q=pair['pair']; l=[]
        for lead in (125,65,31):
            i=endpoint(by[p],elapsed(by[p])-lead); j=endpoint(by[q],i*1000/by[p]['rate']); l.append([curves[p][i],curves[q][j]])
        matched.append(dict(pair=[p,q],kind=by[p]['kind'],relative_drops=l))
    return dict(script_sha256=digest(__file__),checks={**checks,'raw_stream_seal':stamp,'independent_counts':graded},posthoc=oracle,blocks=blocks,actual=graded,
        added_timely=[r['shot'] for r in gained],added_FP=[r['shot'] for r in added],posthoc_closest_new_TP_FP=min(near) if near else None,rows=evidence,matched=matched)


if __name__=='__main__': dump(Path(__file__).with_suffix('.json'),run(Path(sys.argv[1])))
