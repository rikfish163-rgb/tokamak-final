"""The matched static arm must disable BOTH fast and persistent FFN changes."""
import pytest
import json
from types import SimpleNamespace
import numpy as np

from experiments.risk_aligned_online.operational_memory_experiment import arm_flags
from experiments.risk_aligned_online.redesign_20261004.run import run_arm


@pytest.mark.parametrize('arm,expected',[
    ('mapped_static',(False,False)),('intra_control',(True,False)),
    ('inter_only',(False,True)),('both',(True,True))])
def test_explicit_arm_factors(arm,expected):
    assert arm_flags(arm)==expected


def test_unknown_arm_is_not_implicitly_treated_as_both():
    with pytest.raises(ValueError):arm_flags('unknown')


@pytest.mark.parametrize('arm,intra',[('static',False),('intra',True)])
def test_postseal_auxiliary_feedback_is_independent_of_ffn_arm(tmp_path,arm,intra):
    path=tmp_path/'stream';events=[]
    class Data:
        def read_shot(self,device,sid):
            events.append(('read',sid))
            return dict(shot=sid,device=device,times=np.array([0.,.01,.02]),
                        values=np.zeros((3,23),np.float32),available=np.ones((3,23),bool))
        def attach_outcome(self,shot,device,advanced):
            sid=shot['shot']
            assert (path/f'{sid:04d}_{sid}_seal.json').is_file()
            events.append(('reveal',sid))
            return dict(shot,is_disrupt=sid==1,down=.1 if sid==1 else None,
                        horizon=125.,valid=np.ones(3,bool))
    def predict(model,shot,**kw):
        assert kw['intra']==intra and 'is_disrupt' not in shot
        events.append(('predict',shot['shot']))
        return np.full(3,.1),dict(accepted_updates=int(intra))
    def feedback(model,shot,replay,**kw):
        sid=shot['shot'];events.append(('feedback',sid))
        assert [s['shot'] for s in replay]==list(range(1,sid))
        assert '_sealed_scores' in shot and shot['horizon']==125.
        return dict(accepted=True,head=dict(actual_steps=1),memory=dict(actual_steps=0))
    result=run_arm(SimpleNamespace(),order=[dict(shot=1),dict(shot=2)],arm=arm,threshold=.7,
                   output_dir=path,data_api=Data(),predictor=predict,updater=feedback,
                   postseal_feedback=True,device='cpu')
    assert events==[(stage,sid) for sid in (1,2) for stage in ('read','predict','reveal','feedback')]
    assert result['observed_count']==2
    assert json.loads((path/'0002_2_audit.json').read_text())['persistent']['head']['actual_steps']==1


def test_auxiliary_feedback_requires_an_explicit_callback(tmp_path):
    with pytest.raises(ValueError,match='explicit updater'):
        run_arm(SimpleNamespace(),order=[],arm='static',threshold=.7,output_dir=tmp_path/'x',
                postseal_feedback=True)
