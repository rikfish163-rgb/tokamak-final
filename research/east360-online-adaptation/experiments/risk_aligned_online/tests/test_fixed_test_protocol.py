import numpy as np
import pytest

from experiments.risk_aligned_online import simple_data
from experiments.risk_aligned_online.fixed_test_protocol import (
    SPECS, load_test_contract, score_fixed_test, validate_test_ids,
)


@pytest.mark.parametrize('device,total,lead', [('J-TEXT',220,10.),('EAST',360,30.)])
def test_canonical_test_identities_and_timing_are_locked(device,total,lead):
    contract=load_test_contract(device)
    assert len(contract['shot_ids'])==total
    assert contract['lead_ms']==lead
    assert contract['expected_disrupt']==contract['expected_safe']==total//2
    assert contract['independent_confirmation'] is False
    assert validate_test_ids(device,list(reversed(contract['shot_ids'])))==contract
    for ids in (contract['shot_ids'][:-1],
                contract['shot_ids'][:-1]+[contract['shot_ids'][0]],
                contract['shot_ids'][:-1]+[99999999]):
        with pytest.raises(ValueError):validate_test_ids(device,ids)


def test_validation_set_cannot_be_presented_as_source_test():
    with pytest.raises(ValueError):
        validate_test_ids('J-TEXT',simple_data.source_roles()['val'])


@pytest.mark.parametrize('device', ['J-TEXT','EAST'])
def test_exact_lead_boundary_is_late_and_early_false_alarms_count(device):
    contract=load_test_contract(device)
    ids=contract['shot_ids'];half=len(ids)//2;lead=contract['lead_ms']
    rows=[]
    for index,sid in enumerate(ids):
        disruptive=index<half
        # The first disruptive shot is exactly at the limit; the second
        # exceeds it by 1us. One safe shot alarms at its first observed point.
        advance=lead+(0.001 if index==1 else 0.)
        rows.append(dict(shot=sid,is_disrupt=disruptive,down=1. if disruptive else None,
            times=np.array([1.-advance/1000.]),
            scores=np.array([.8 if index in (0,1,half) else .1])))
    result=score_fixed_test(device,rows,.5)
    assert result['tp']==1 and result['fp']==1
    assert not result['predictions'][0]['timely']
    assert result['predictions'][1]['timely']
    assert result['threshold']==.5
    rows[half]['is_disrupt']=True
    rows[half]['down']=1.
    with pytest.raises(ValueError,match='class counts'):
        score_fixed_test(device,rows,.5)


def test_unknown_device_is_not_mapped_to_an_existing_test():
    with pytest.raises(ValueError):load_test_contract('other')


def test_changed_manifest_must_not_silently_redefine_the_test(monkeypatch,tmp_path):
    path=tmp_path/'changed.json';path.write_text('{}')
    monkeypatch.setitem(SPECS,'EAST',dict(SPECS['EAST'],path=path))
    with pytest.raises(ValueError,match='manifest changed'):load_test_contract('EAST')
