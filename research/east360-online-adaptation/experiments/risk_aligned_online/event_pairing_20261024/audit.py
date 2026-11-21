"""Audit auxiliary data contracts, pairing identities and independent AUCs."""
import argparse
from pathlib import Path

import numpy as np

from .. import simple_data
from ..causal_input import apply_input_contract
from ..event_prefix_probe import read_events,assign_events,DEFAULT_LABELS
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import features
from .run import READ,DEFAULT_PARENT,PARENT_SHA256,load_parent,shot_weights,PREVIOUS


def independent_auc(shot,y,score):
    w=shot_weights(shot,y);p=y==1;n=y==0
    delta=score[p,None]-score[n][None,:]
    mass=w[p,None]*w[n][None,:]
    return float(np.sum(((delta>0)+.5*(delta==0))*mass)/mass.sum())


def main(root):
    def read(name):
        with np.load(root/name) as z:return {k:z[k].copy() for k in z.files}
    d=read('source.npz');tr=d['role']=='train';va=d['role']=='val'
    train={k:v[tr] for k,v in d.items()};val={k:v[va] for k,v in d.items()}
    p=READ(root/'protocol.json');done=READ(root/'completed.json')
    for f,h in done['artifact_sha256'].items():assert sha(root/f)==h,f
    assert set(train['shot'])==set(p['train_ids']);assert set(val['shot'])==set(p['val_ids'])
    assert not set(train['shot'])&set(val['shot']);assert not set(d['shot'])&set(simple_data.source_roles()['test'])
    pair_checks=0
    for mode in ('risk','family','shuffle_131','shuffle_132','shuffle_133'):
        pp=read('pairs_'+mode+'.npz');y=pp['y'];shot=train['shot'];family=train['family']
        for f in range(6):assert np.sum(y[family==f])==np.sum(train['y'][family==f])
        for i in range(len(y)):
            pos=pp['positive'][i,:pp['positive_count'][i]];neg=pp['negative'][i,:pp['negative_count'][i]]
            assert len(set(shot[pos]))==len(pos) and len(set(shot[neg]))==len(neg)
            assert np.all(shot[pos]!=shot[i]) and np.all(shot[neg]!=shot[i])
            assert np.all(family[pos]==family[i]) and np.all(y[pos]==y[i])
            if mode=='family':assert np.all(family[neg]!=family[i]) and np.all(y[neg]==y[i])
            else:assert np.all(family[neg]==family[i]) and np.all(y[neg]!=y[i])
            pair_checks+=1
    parent,_=load_parent(DEFAULT_PARENT);center=parent.center.numpy();scale=parent.scale.numpy()
    events,_=read_events(DEFAULT_LABELS);advanced=simple_data.advanced_times();selected=[]
    for role in ('train','val'):
        for family in range(6):
            for y in (0,1):selected.append(int(np.flatnonzero((d['role']==role)&(d['family']==family)&(d['y']==y))[0]))
    for sid in np.unique(d['shot'][selected]):
        s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('J-TEXT',int(sid))),'J-TEXT',advanced)
        target,_=assign_events(s['times'],events[int(sid)])
        for i in [j for j in selected if d['shot'][j]==sid]:
            end=int(d['end'][i]);assert s['valid'][end] and int(s['y'][end])==int(d['y'][i])
            assert target[end]==d['family'][i] and s['times'][end]==d['time'][i]
            assert not s['is_disrupt'] or round((s['down']-s['times'][end])*1000,9)>10
            x,m=simple_data.window(s,end,128)
            np.testing.assert_array_equal(features(x,m,center,scale),d['x'][i])
    east=read('east20.npz')
    with np.load(PREVIOUS/'bank.npz') as z:
        recomputed=np.stack([features(x,m,center,scale) for x,m in zip(z['x'],z['mask'])])
        np.testing.assert_array_equal(east['x'],recomputed)
    estimates=READ(root/'results.json');scores=read('scores.npz');auc_checks=0
    for name,result in estimates.items():
        ss=scores[name+'_val'];expected=independent_auc(val['shot'],val['y'],ss)
        np.testing.assert_allclose(expected,result['val']['auc'],atol=1e-14,rtol=0);auc_checks+=1
        for family in range(6):
            use=val['family']==family;expected=independent_auc(val['shot'][use],val['y'][use],ss[use])
            np.testing.assert_allclose(expected,result['val']['family_auc'][chr(65+family)],atol=1e-14,rtol=0);auc_checks+=1
    pairs=READ(root/'same_history_recheck/paired_scores.json');results=READ(root/'same_history_recheck/results.json');pair_aucs=0
    for name,rr in pairs.items():
        values=[]
        for row in rr:
            positive,negative=row['positive_scores'],row['negative_scores']
            value=sum((a>b)+.5*(a==b) for a in positive for b in negative)/(len(positive)*len(negative))
            assert value==row['auc'];values.append(value);pair_aucs+=1
            assert len(row['history_shots'])==18 and not set(row['held_shots'])&set(row['history_shots'])
        assert np.mean(values)==results[name]['east_pair_auc']
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    report=dict(original_completed_sha256=sha(root/'completed.json'),
        recheck_completed_sha256=sha(root/'same_history_recheck/completed.json'),
        pairing_anchor_checks=pair_checks,raw_source_windows_rebuilt=len(selected),
        raw_east_historical_windows_rebuilt=len(east['x']),source_auc_recomputed=auc_checks,
        east_pair_auc_recomputed=pair_aucs,train=dict(windows=len(train['y']),shots=len(set(train['shot'])),positive=int(train['y'].sum())),
        val=dict(windows=len(val['y']),shots=len(set(val['shot'])),positive=int(val['y'].sum())),
        source_risk_unchanged=True,test360_read=False,scope='mechanics and data-contract audit; not independent efficacy')
    _durable_json(root/'verification.json',report);print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
