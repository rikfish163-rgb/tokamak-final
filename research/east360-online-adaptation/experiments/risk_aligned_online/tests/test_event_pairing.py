import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.event_pairing_20261024.learning import (
    features, reverse_features, mean_features, pair_candidates, sample_triplets,
    pair_loss, neighbour_score,
)


def test_masked_causal_features_and_reversal():
    x=np.tile(np.arange(128)[:,None],(1,23)).astype('float32');m=np.ones_like(x,bool)
    a=features(x,m,np.zeros(23),np.ones(23))
    b=features(x[::-1].copy(),m[::-1].copy(),np.zeros(23),np.ones(23))
    np.testing.assert_allclose(reverse_features(a),b,atol=1e-6)
    np.testing.assert_array_equal(mean_features(a),mean_features(b))
    x[:,2]=1e20
    np.testing.assert_array_equal(features(x,m,np.zeros(23),np.ones(23)),a)
    m[:,3]=False;c=features(x,m,np.zeros(23),np.ones(23));x[:,3]=np.nan
    np.testing.assert_array_equal(features(x,m,np.zeros(23),np.ones(23)),c)
    assert a.shape==(391,)


def synthetic():
    family=np.repeat(np.arange(3),12);y=np.tile(np.repeat([0,1],6),3)
    shots=np.arange(36);x=np.arange(36*23,dtype=float).reshape(36,23)/100
    return x,shots,y,family


def test_pairing_preserves_family_and_hazard_but_excludes_same_shot():
    x,s,y,f=synthetic();pairs=pair_candidates(x,s,y,f,mode='risk')
    rng=np.random.default_rng(17);t=sample_triplets(pairs,rng,100)
    a,p,n=t.T
    assert np.all(f[a]==f[p]) and np.all(f[a]==f[n])
    assert np.all(y[a]==y[p]) and np.all(y[a]!=y[n])
    assert np.all(s[a]!=s[p]) and np.all(s[a]!=s[n])
    family=pair_candidates(x,s,y,f,mode='family')
    a,p,n=sample_triplets(family,rng,100).T
    assert np.all(f[a]==f[p]) and np.all(f[a]!=f[n])
    assert np.all(y[a]==y[p]) and np.all(y[a]==y[n])
    with pytest.raises(ValueError):pair_candidates(x,s,np.zeros(36,int),f,mode='risk')


def test_contrastive_loss_gradient_and_batch_target_shape():
    z=torch.tensor([[[1.,0.],[0.,1.],[1.,0.]]],requires_grad=True)
    before=pair_loss(z);before.backward();assert z.grad.abs().sum()>0
    after=pair_loss(z.detach()-.1*z.grad)
    assert after<before
    with pytest.raises(ValueError):pair_loss(torch.zeros(3,4))


def test_whole_shot_exclusion_and_no_query_labels_in_neighbour_score():
    memory=np.array([[1.,0.],[.8,.2],[0.,1.],[.2,.8]],float)
    shot=np.array([1,2,3,4]);y=np.array([1,1,0,0])
    scores=neighbour_score(memory[:1],memory,shot,y,exclude=np.array([1]),k=1)
    assert scores[0]>0
    changed=memory.copy();changed[0]=[-100.,-100.]
    np.testing.assert_array_equal(scores,neighbour_score(memory[:1],changed,shot,y,exclude=np.array([1]),k=1))
    with pytest.raises(ValueError):neighbour_score(memory[:1],memory,shot,y,exclude=np.array([1]),k=3)


def test_pair_out_uses_identical_history_for_both_query_shots():
    from experiments.risk_aligned_online.event_pairing_20261024.evaluation import pair_out_auc
    x=np.array([[3.],[4.],[5.],[0.],[1.],[2.]])
    shot=np.arange(6);y=np.array([1,1,1,0,0,0])
    score,rows=pair_out_auc(x,x,shot,y,k=1)
    expected=[]
    for row in rows:
        held=row['held_shots'];use=~np.isin(shot,held)
        assert not set(held)&set(row['history_shots'])
        assert set(row['history_shots'])==set(shot[use])
        positive,negative=[neighbour_score(x[[sid]],x[use],shot[use],y[use],
            exclude=np.array([-1]),k=1)[0] for sid in held]
        value=float(positive>negative)+.5*float(positive==negative)
        assert row['auc']==value;expected.append(value)
    assert len(rows)==9 and score==np.mean(expected)
    tied,rr=pair_out_auc(np.zeros_like(x),np.zeros_like(x),shot,y,k=1)
    assert tied==.5 and all(r['auc']==.5 for r in rr)
    with pytest.raises(ValueError):pair_out_auc(x,x,shot,y,k=3)


def test_screen_does_not_promote_below_chance_or_unstable_target():
    from experiments.risk_aligned_online.event_pairing_20261024.evaluation import qualifies
    controls={k:dict(val_conditional_auc=.60,east_auc=.30) for k in ('mean','shuffle','family','reverse')}
    values={**controls,'risk':dict(val_conditional_auc=.75,east_auc=.48)}
    assert not qualifies(values,.04,[.42,.60,.42])
    values['risk']['east_auc']=.70
    assert not qualifies(values,.60,[.9,.3,.9])
    assert qualifies(values,.60,[.7,.7,.7])
    values['reverse']['east_auc']=.72
    assert not qualifies(values,.60,[.7,.7,.7])
