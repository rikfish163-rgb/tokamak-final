import numpy as np
import pytest

from experiments.risk_aligned_online import io23


def record(sid=1,device='J-TEXT',coefficient=.374):
    x=np.ones((160,23),np.float32)
    x[:,io23.RAW23.index('Bt')]=np.linspace(1.5,2.5,160)
    x[:,io23.RAW23.index('ip')]=np.linspace(1e5,2e5,160)
    x[:,io23.RAW23.index('qa_proxy')]=coefficient*x[:,0]/x[:,9]
    return dict(shot=sid,device=device,values=x,available=np.ones_like(x,bool))


def functions():
    from experiments.risk_aligned_online.qa_convention import fit_source_qa_convention,rebuild_qa
    return fit_source_qa_convention,rebuild_qa


def test_fit_uses_source_signals_and_equally_weights_shots_not_outcomes():
    fit,_=functions()
    a,b=record(),record(2,coefficient=.375)
    b['values']=np.tile(b['values'],(10,1));b['available']=np.tile(b['available'],(10,1))
    result=fit([a,b])
    assert result['coefficient']==pytest.approx(.3745,abs=1e-7)
    assert result['source_ids']==[1,2] and result['target_information_used']==0


@pytest.mark.parametrize('fault',['target','label','duplicate','missing','nonfinite','variable','mask'])
def test_unsupported_source_convention_fails_explicitly(fault):
    fit,_=functions();r=record();records=[r]
    if fault=='target':r['device']='EAST'
    elif fault=='label':r['is_disrupt']=True
    elif fault=='duplicate':records=[r,r]
    elif fault=='missing':r['available'][:,13]=False
    elif fault=='nonfinite':r['values'][:,13]=np.nan
    elif fault=='variable':r['values'][80:,13]*=2
    else:r['available']=r['available'].astype(float)
    with pytest.raises(ValueError):fit(records)


def test_rebuild_changes_only_qa_and_never_reads_storage_qa_or_outcome():
    _,rebuild=functions();r=record(device='EAST',coefficient=1.)
    old=r['values'].copy();oldmask=r['available'].copy()
    r['is_disrupt']=True;r['down']=5.;r['values'][0,13]=np.nan;r['available'][0,13]=False
    out=rebuild(r,.374)
    np.testing.assert_allclose(out['values'][:,13],.374*old[:,0]/old[:,9])
    assert out['available'][:,13].all()
    others=np.arange(23)!=13
    np.testing.assert_array_equal(out['values'][:,others],old[:,others])
    np.testing.assert_array_equal(out['available'][:,others],oldmask[:,others])
    assert np.isnan(r['values'][0,13]) and not r['available'][0,13]
    changed=dict(r,is_disrupt=False,down=100.,y=np.ones(160));changed['values']=r['values'].copy();changed['values'][:,13]=1e10
    again=rebuild(changed,.374)
    np.testing.assert_array_equal(again['values'],out['values'])
    np.testing.assert_array_equal(again['available'],out['available'])


def test_pointwise_causality_and_idempotence():
    _,rebuild=functions();r=record(device='EAST');out=rebuild(r,.374)
    changed=dict(r,values=r['values'].copy());changed['values'][100:,[0,9]]*=3
    next_out=rebuild(changed,.374)
    np.testing.assert_array_equal(next_out['values'][:100],out['values'][:100])
    twice=rebuild(out,.374)
    np.testing.assert_array_equal(twice['values'],out['values'])


def test_missing_denominator_and_nonfinite_inputs_are_not_filled_by_future():
    _,rebuild=functions();r=record(device='EAST')
    r['values'][0,9]=0.;r['available'][1,0]=False;r['values'][2,0]=np.nan
    out=rebuild(r,.374)
    assert not out['available'][:3,13].any() and not out['values'][:3,13].any()
    assert out['available'][3:,13].all()


@pytest.mark.parametrize('coefficient',[0.,-1.,float('nan'),float('inf')])
def test_invalid_mapping_coefficient_rejected(coefficient):
    _,rebuild=functions()
    with pytest.raises(ValueError):rebuild(record(device='EAST'),coefficient)
