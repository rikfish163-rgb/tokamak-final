import torch
import pytest

from .run import DEFAULT_SOURCE, load_parent


def test_embedding_site_starts_from_same_qualified_classifier():
    model, state, gate, _ = load_parent(DEFAULT_SOURCE,'cpu',persistent_site='embedding')
    assert gate['qualified']
    params=list(model.persistent_parameters())
    assert sum(p.numel() for p in params) == 23616
    assert all(p.requires_grad for p in params)
    assert not any(p.requires_grad for p in model.base.parameters())
    for name, value in model.base.state_dict().items():
        assert torch.equal(value,state['model'][name])


def test_unknown_persistent_site_rejected():
    with pytest.raises(ValueError,match='site'):
        load_parent(DEFAULT_SOURCE,'cpu',persistent_site='score_head')
