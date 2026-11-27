"""Independent CPU ledger audit. No training code, archived labels, or model from the experiment is imported."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parent
REFERENCE='3fefd51a0a1e16bb82735f59e8b332857eecd1c0d66102e6b4b413371a93ce97'


def grade(first,down):
    if down is None: return 'TN' if first is None else 'FP'
    if first is None or first>down+1e-9: return 'MISS'
    return 'TP_timely' if round((down-first)*1000,6)>30 else 'TARDY'


def main():
    sets={s:json.loads((ROOT/(s+'.json')).read_text()) for s in ('temporal_depth','temporal_release','temporal_parent_check')}
    ids=[r['shot'] for r in sets['temporal_depth']['ledger']];assert len(ids)==len(set(ids))==360 and ids==sorted(ids)
    assert sha256(','.join(map(str,ids)).encode()).hexdigest()==REFERENCE
    meta={}; names={};total=0
    for sid in ids:
        path=ROOT.parents[3]/'data/EAST/unified_hdf5'/str(sid//100*100)/f'{sid}.hdf5'
        with h5py.File(path,'r') as f:
            ip=f['data/ip'];y=int(f['meta/IsDisrupt'][()]);assert y in (0,1)
            down=float(f['meta/DownTime'][()]) if y else None
            meta[sid]=(down,float(ip.attrs['StartTime']),float(ip.attrs['SampleRate']),len(ip))
    for name,j in sets.items():
        assert sha256((ROOT/(name+'.py')).read_bytes()).hexdigest()==j['script'] and j['order']==REFERENCE
        if 'training_script' in j:assert j['training_script']==sets['temporal_depth']['script']
        rows=j['ledger']; assert ids==[r['shot'] for r in rows];firsts={}
        for r in rows:
            sid=r['shot'];down,t0,rate,n=meta[sid];assert r['down']==down and len(r['pre_reveal_sha'])==64
            for k,f in r['first'].items():
                if f is not None:
                    ii=(f-t0)*rate;assert -.00001<=ii<=n-1+.00001 and abs(ii-round(ii))<1e-6
                firsts[sid,k]=grade(f,down);total+=1
        def chk(sub,stats):
            for k,a in stats.items():
                rr=[r for r in sub if k in r['first']]; cc=Counter(firsts[r['shot'],k] for r in rr)
                assert all(cc[m]==v for m,v in a['counts'].items()) and sum(cc.values())==len(rr)
                lead=[round(1000*(r['down']-r['first'][k]),6) for r in rr if firsts[r['shot'],k]=='TP_timely']
                assert [sum(t<=125 for t in lead),sum(125<t<=3000 for t in lead),sum(t>3000 for t in lead)]==a['timely_lead_bins']
                p=[r['peak'][k] for r in rr if r['down'] is not None];n=sorted([r['peak'][k] for r in rr if r['down'] is None],reverse=True)
                for gate,v in a['posthoc'].items():
                    top=(int(gate)*len(n)-1)//100;threshold=n[top]
                    assert v==[sum(z>threshold for z in p),sum(z>threshold for z in n)]
        chk(rows,j['all360'])
        for b,d in enumerate(j['blocks']):chk(rows[90*b:90*b+90],d)
        if 'suffix' in j:chk(rows[270:],j['suffix'])
        for edge,res in j['paired'].items():
            a,b=edge.split('__'); pairs=[(s,firsts[s,a],firsts[s,b]) for s in ids]
            for cat,lab in (('gain_D','TP_timely'),('add_FP','FP'),('lose_D','TP_timely'),('drop_FP','FP')):
                assert [s for s,i,k in pairs if i!=k and (k if cat in ('gain_D','add_FP') else i)==lab]==res[cat]
        history={s:i for i,s in enumerate(ids)}; audit=j.get('training',j.get('training_donors'))
        for i,u in enumerate(audit):
            donors=u['donors'] if isinstance(u,dict) else u; reserved=name=='temporal_release'
            for y,coll in zip((True,False),donors):
                assert len(set(coll))==len(coll)<=8 and all(s in ids[:i] and (meta[s][0] is not None)==y for s in coll)
                if reserved and not y: assert all((history[s]+1)%3!=0 for s in coll)
            if isinstance(u,dict):
                ds,ss,hz,n=meta[ids[i]];picks=np.array(u['after_reveal_indices']);assert len(np.unique(picks))==8 and np.all((picks>=0)&(picks<n))
                if ds is None:assert np.min(np.diff(np.sort(picks)))*1000/hz>=16-1e-7
                else:
                    lead=np.round((ds-(ss+picks/hz))*1000,6);assert np.all((lead>30)&(lead<=125))
        if name=='temporal_release':
            for i,r in enumerate(j['releases']):
                prefix=ids[:i*20];calids=[s for z,s in enumerate(prefix,1) if z%3==0 and meta[s][0] is None];assert r['cal_ids']==calids
                for k,c in r['offset'].items():assert c==np.nextafter(sorted(r['peaks'][k])[int(np.ceil(.8*(len(calids)+1)))-1],np.inf)
        names[name]=len(j['all360'])
    assert sum(meta[s][0] is not None for s in ids)==180;assert grade(.97,1.)=='TARDY' and grade(.969999,1.)=='TP_timely'
    print('Verified raw HDF5 outcomes, within-shot sample clocks, full/90/suffix counts, registered pair flips, budget cutoffs and past-only gradient/calibration donor membership.')
    print('Audited issued arm-shot firsts:',total,'; main tables:',names)


if __name__=='__main__': main()
