"""Post-launch regression checks; deliberately separate from frozen run sources."""
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.shot_objective_alignment_20261023.learning import (
    logits_at,bag_loss,backward_bag,
)
from experiments.risk_aligned_online.simple23 import Config,Simple23
from experiments.risk_aligned_online.paired_feedback_20261022.model import PairedFeedback


@pytest.mark.parametrize('disruptive',[False,True])
@pytest.mark.parametrize('mode',['mean','peak'])
def test_padded_masked_positive_and_negative_gradient_equivalence(disruptive,mode):
    torch.set_num_threads(1);torch.manual_seed(113)
    model=PairedFeedback(Simple23(Config(dim=8,heads=2,fast_site='tokens'),np.zeros(23),np.ones(23)),np.ones(23))
    model.set_stage('frozen');names=set(model.encoder_names())
    for k,p in model.named_parameters():p.requires_grad_(k in names)
    values=np.random.default_rng(131).normal(size=(7,23)).astype(np.float32)
    available=np.ones_like(values,dtype=bool);available[:3,5]=False;values[:3,5]=0
    signal=dict(values=values,available=available);ends=np.arange(7)
    params=[p for p in model.parameters() if p.requires_grad]
    full=bag_loss(logits_at(model,signal,ends,device='cpu'),disruptive,.7,mode)
    (.25*full).backward();expected=[p.grad.clone() for p in params];model.zero_grad(set_to_none=True)
    result=backward_bag(model,signal,ends,disruptive,.7,mode,weight=.25,device='cpu',batch_size=3)
    assert result['loss']==pytest.approx(float(full.detach()),abs=2e-6)
    for p,g in zip(params,expected):torch.testing.assert_close(p.grad,g,rtol=4e-5,atol=4e-6)
    assert all(p.grad is None for k,p in model.named_parameters() if k not in names)
