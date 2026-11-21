import numpy as np
import pytest

from experiments.risk_aligned_online.crossfit_teacher_20261025.design import folds,fit_fold_map
from experiments.risk_aligned_online.warmstart_alignment_20261022.alignment import fit_statistics


def _record(sid,device,density):
    x=np.ones((32,23),np.float32);x[:,12]=density
    return dict(shot=sid,device=device,values=x,available=np.ones_like(x,bool))


def test_four_fold_identity_cover_and_stratification():
    ids=list(range(20));flags={i:i in (1,6,11,16) for i in ids}
    plans=folds(ids,flags)
    assert len(plans)==4 and sorted(s for p in plans for s in p['held'])==ids
    for p in plans:
        assert len(p['train'])==15 and len(p['held'])==5
        assert not set(p['train'])&set(p['held'])
        assert set(p['train']+p['held'])==set(ids)
        assert sum(flags[s] for s in p['held'])==1
    assert plans==folds(ids[::-1],flags)
    with pytest.raises(ValueError):folds(ids[:-1],flags)
    with pytest.raises(ValueError):folds(ids,{i:False for i in ids})


def test_mapping_uses_only_exact_training_identities():
    source=fit_statistics([_record(100,'J-TEXT',4)],device='J-TEXT',allowed_ids=[100])
    records=[_record(i,'EAST',2) for i in range(20)]
    chosen=list(range(15))
    first=fit_fold_map(source,records,chosen)
    assert first['gain'][12]==pytest.approx(2.)
    assert first['calibration_ids']==chosen
    for r in records[15:]:r['values'][:]=100000
    assert first==fit_fold_map(source,records,chosen)
    assert first['channels'][2]['status']=='not_selected_identity'
    with pytest.raises(ValueError):fit_fold_map(source,records,chosen+[15])
    with pytest.raises(ValueError):fit_fold_map(source,records[:-10],chosen)


def test_mapping_rejects_label_fields_in_training_records():
    source=fit_statistics([_record(100,'J-TEXT',4)],device='J-TEXT',allowed_ids=[100])
    records=[_record(i,'EAST',2) for i in range(20)]
    records[0]['is_disrupt']=True
    with pytest.raises(ValueError):fit_fold_map(source,records,list(range(15)))
