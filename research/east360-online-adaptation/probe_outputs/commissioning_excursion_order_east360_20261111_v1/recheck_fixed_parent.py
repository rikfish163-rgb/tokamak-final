"""Independent HDF5 verification plus a locked-risk ordered-term probe."""

from pathlib import Path
import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.history_gradient_metric_20261102.run import read, sha, write


ROOT = Path('probe_outputs/commissioning_excursion_order_east360_20261111_v1')
p, seal, result = (read(ROOT/k) for k in ('protocol.json','seal.json','result.json'))
assert len(p['east_order']) == len(set(p['east_order'])) == 360
assert len(seal['prefix_witnesses']) == 3
for name, expected in seal['assets'].items():
    assert sha(ROOT/name) == expected
assert sha(ROOT/'executed_probe.py') == p['producer_sha256']
assert sha(Path(p['reference'])/'seal.json') == p['reference_seal']
roles = io23.source_roles()
assert p['fit'] == roles['train'] and p['held_source'] == roles['val']
assert not set(roles['test']).intersection(p['fit']+p['held_source'])

def load_table(name):
    with np.load(ROOT/name, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}

train, val, east = (load_table(k) for k in ('source_train.npz','source_val.npz','east_windows.npz'))
for table, ids in ((train,roles['train']),(val,roles['val']),(east,p['east_order'])):
    assert set(table['shot']) <= set(ids)
    assert ((table['endpoint']-p['first_effect']) % p['stride'] == 0).all()
    assert all(np.isfinite(v).all() for v in table.values())
    assert len(set(zip(table['shot'],table['endpoint']))) == len(table['shot'])

clocks = {}
advanced = io23.advanced_times()
for device, ids in (('J-TEXT',roles['train']+roles['val']),('EAST',p['east_order'])):
    for sid in ids:
        with h5py.File(io23.path_for(device,sid),'r') as f:
            d = np.asarray(f['meta/IsDisrupt'][()]).item()
            assert d in (0,1)
            down = float(np.asarray(f['meta/DownTime'][()]).item()) if d else None
            ds = f['data/ip']
            start = float(np.asarray(ds.attrs['StartTime']).item())
            rate = float(np.asarray(ds.attrs['SampleRate']).item())
            horizon = (io23.effective_horizon_ms(sid,advanced)[0] if device=='J-TEXT' else 125.)
            clocks[device,sid] = (d,down,start,rate,len(ds),horizon)

def verify_clock(table, device):
    for sid in np.unique(table['shot']):
        take = table['shot']==sid
        _,_,start,rate,n,_ = clocks[device,sid]
        q = table['endpoint'][take]
        assert (q<n).all()
        np.testing.assert_array_equal(table['time'][take],start+q/rate)

for t, device in ((train,'J-TEXT'),(val,'J-TEXT'),(east,'EAST')):
    verify_clock(t,device)

index = {name:{(sid,q):i for i,(sid,q) in enumerate(zip(t['shot'],t['endpoint']))}
         for name,t in (('train',train),('val',val),('east',east))}
checked = 0
details = {}
for role, r in result['comparisons'].items():
    which = 'train' if role.startswith('train') else 'val' if role.startswith('held') else 'east'
    t = {'train':train,'val':val,'east':east}[which]
    device = 'EAST' if which=='east' else 'J-TEXT'
    limit = 30. if which=='east' or role.endswith('_30') else 10.
    pairs = np.asarray([[index[which][tuple(a)],index[which][tuple(b)]] for a,b in r['matched']])
    ia,ib = pairs.T
    assert len(pairs)==r['pairs'] and len(np.unique(t['shot'][ia]))==len(np.unique(t['shot'][ib]))==len(pairs)
    gap = np.abs(t['match'][ia]-t['match'][ib])
    assert (gap<=np.asarray(p['calipers'])+1e-12).all()
    lead = []
    for i in ia:
        d,down,_,_,_,h = clocks[device,t['shot'][i]]
        ms = np.round((down-t['time'][i])*1000,9)
        assert d and limit < ms <= h+1e-6
        lead.append(float(ms))
    assert all(clocks[device,t['shot'][i]][0]==0 for i in ib)
    checked += 2*len(pairs)
    if which=='train':
        np.testing.assert_array_equal(pairs,train['pairs'])
        n=t['unordered'].shape[1]
        wsp=train['ordered_weight']
        common=t['unordered']@wsp[:n]
        order=t['ordered']@wsp[n:]
        existing=dict(parent=t['unordered'][:,0],unordered_fit=t['unordered']@train['unordered_weight'],
                      ordered_true=common+order,same_model_no_order=common,swap_first_two=common-order)
    else:
        existing={k:t['score_'+k] for k in r['scores']}
    for k,v in existing.items():
        z=v[ia]-v[ib]
        assert int((z>0).sum())==r['scores'][k]['wins']
        np.testing.assert_allclose(np.logaddexp(0.,-z).mean(),r['scores'][k]['mean_pair_log_loss'])
    details[role]=(t,ia,ib,lead)

# The previously frozen parent score is an offset, not a fitted coefficient.
# Same 47 training pairs, same ridge, only the 8 ordered coordinates fitted.
t, ia, ib, _ = details['train_source_10']
scale=np.maximum(t['ordered'][np.r_[ia,ib]].std(0),.05)
x=(t['ordered'][ia]-t['ordered'][ib])/scale
parent=t['unordered'][ia,0]-t['unordered'][ib,0]
def objective(w):
    z=parent+x@w
    return (np.logaddexp(0.,-z).mean()+.05*(w@w),
            -(x.T@expit(-z))/len(z)+.10*w)
fit=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',
             options={'gtol':1e-9,'ftol':1e-13})
assert fit.success
weight=fit.x/scale
metrics={}
for name,(t,ia,ib,lead) in details.items():
    base=t['unordered'][ia,0]-t['unordered'][ib,0]
    step=(t['ordered'][ia]-t['ordered'][ib])@weight
    rows={k:dict(wins=int((v>0).sum()), ties=int((v==0).sum()),
                 pair_logloss=float(np.logaddexp(0.,-v).mean()))
          for k,v in (('frozen_parent',base),('ordered',base+step),('swapped',base-step))}
    metrics[name]=dict(pairs=len(ia),wins=rows,median_lead_ms=float(np.median(lead)),
                       over30=sum(v>30 for v in lead),
                       median_abs_ordered_delta=float(np.median(np.abs(step))))
audit=dict(status='PASS_EXCURSION_HDF5_AND_MATCH_AUDIT',checked_windows=sum(len(t['shot'])
           for t in (train,val,east)),checked_pair_endpoints=checked,
           clock_shots=len(clocks),weights=weight.tolist(),
           scope='postseal fixed-parent auxiliary ranking; no new alarms; source TRAIN only',
           comparisons=metrics)
write(ROOT/'fixed_parent_and_hdf5_recheck.json',audit)
print(audit,flush=True)
