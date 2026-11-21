import numpy as np
import torch

from experiments.risk_aligned_online.history_tangent_20261023.learning import history_basis, logit
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
from experiments.risk_aligned_online import simple_data
from experiments.risk_aligned_online.tests.test_task_geometry import small_model


def test_historical_candidates_preserve_label_window_and_actual_peak_gradient():
    model=small_model();rng=np.random.default_rng(431)
    records=[]
    for sid,disrupt in ((1,False),(2,True)):
        records.append(dict(shot=sid,device='EAST',horizon=125.,
            values=rng.normal(size=(200,23)).astype('float32'),available=np.ones((200,23),bool),
            times=np.arange(200)/1000.,valid=np.ones(200,bool),is_disrupt=disrupt,
            down=.2 if disrupt else None,y=np.r_[np.zeros(75),np.ones(125)] if disrupt else np.zeros(200)))
    bases,anchors,rows,receipts=history_basis(model,records,device='cpu')
    assert receipts[0]['candidate_count']==200 and receipts[1]['candidate_count']==95
    assert 75<=receipts[1]['end']<170
    assert bases['history'].shape[0]==2 and rows.shape[0]==2
    fast=initial_encoder(model)
    for s,r in zip(records,receipts):
        ends=range(75,170) if s['is_disrupt'] else range(200)
        logits=[]
        for end in ends:
            x,m=[torch.as_tensor(a[None]) for a in simple_data.window(s,end,128)]
            logits.append(float(logit(model,x,m,fast).detach()[0]))
        assert r['end']==list(ends)[np.argmax(logits)]
        assert abs(r['parent_logit']-max(logits))<1e-5
    assert anchors['x'].shape==(2,128,23)
    torch.testing.assert_close(bases['history']@bases['history'].T,torch.eye(2))
