import json

import pytest

from . import run
from .evaluate import acceptance


def test_default_stream_is_no_longer_balanced360():
    assert run.DEFAULT_ORDER.parent.name == 'east_all_nominal_stream_20261004'
    assert run.DEFAULT_ROSTER.parent == run.DEFAULT_ORDER.parent
    ids, provenance = run.load_roster(run.DEFAULT_ROSTER)
    assert len(ids) == 2953
    assert provenance['selection_protocol']['rate_contract']=='nominal'
    assert provenance['rejected']==[]


def test_explicit_roster_is_not_required_to_be_balanced360(tmp_path):
    path = tmp_path/'roster.json'
    path.write_text(json.dumps({'shot_ids':[12,15,18], 'selection':'all readable shots'}))
    ids, provenance = run.load_roster(path)
    assert ids == [12,15,18]
    assert provenance['selection'] == 'all readable shots'
    run.validate_order([{'shot':15},{'shot':12},{'shot':18}], ids)


def test_new_and_archived_rate_contracts_cannot_silently_differ():
    # Archived strict manifests declare atol but predate the named field.
    old={'selection_protocol':{'rate_atol_hz':1e-9,'rate_rtol':0.}}
    run.validate_roster_rate_contract(old,'strict')
    with pytest.raises(ValueError,match='rate contracts differ'):
        run.validate_roster_rate_contract(old,'nominal')
    new={'selection_protocol':{'rate_contract':'nominal','rate_atol_hz':.5}}
    run.validate_roster_rate_contract(new,'nominal')
    with pytest.raises(ValueError,match='rate contracts differ'):
        run.validate_roster_rate_contract(new,'strict')


@pytest.mark.parametrize('ids', [[],[1,1],[True,2],[1.5,2],['1',2],[-1,2]])
def test_bad_explicit_roster_fails(tmp_path, ids):
    path = tmp_path/'roster.json'
    path.write_text(json.dumps({'shot_ids':ids,'selection':'all readable shots'}))
    with pytest.raises(ValueError,match='roster'):
        run.load_roster(path)


def test_roster_requires_selection_provenance(tmp_path):
    path = tmp_path/'roster.json'
    path.write_text(json.dumps({'shot_ids':[1,2]}))
    with pytest.raises(ValueError,match='selection'):
        run.load_roster(path)


def test_non360_complete_stream_can_be_assessed_without_relaxing_gain_gates():
    def m(tp,fp):
        return dict(tp=tp,fp=fp,n_disrupt=50,n_safe=100,
                    tpr=tp/50,fpr=fp/100,J=tp/50-fp/100)
    d={'static':m(40,15),'intra':m(42,14),'inter':m(43,13),'both':m(44,12)}
    verdict=acceptance(d, expected_count=150)
    assert verdict['full_locked_stream'] and verdict['all_required_gates_pass']
    assert not acceptance(d, expected_count=151)['all_required_gates_pass']


def test_large_stream_does_not_relax_independent_intra_requirement():
    def m(tp,fp):
        return dict(tp=tp,fp=fp,n_disrupt=50,n_safe=100,
                    tpr=tp/50,fpr=fp/100,J=tp/50-fp/100)
    d={'static':m(40,15),'intra':m(40,15),'inter':m(43,13),'both':m(44,12)}
    assert not acceptance(d, expected_count=150)['all_required_gates_pass']
