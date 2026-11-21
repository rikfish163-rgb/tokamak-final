import torch

from ..simple23 import Config, Simple23
from .model import ContextModel


def models():
    torch.manual_seed(131)
    base = Simple23(Config(fast_site='tokens',fast_rule='normalized'),torch.zeros(23),torch.ones(23)).eval()
    return base,ContextModel(base),ContextModel(base,context_mode='ordered')


def test_ordered_context_distinguishes_interior_patch_permutations():
    base,old,new = models()
    x = torch.randn(1,128,23)
    mask = torch.ones_like(x,dtype=torch.bool)
    patches = x.reshape(1,16,8,23)
    order = list(range(16))
    order[3],order[10] = order[10],order[3]
    changed = patches[:,order].reshape_as(x)
    torch.testing.assert_close(old._context(x,mask),old._context(changed,mask),atol=1e-6,rtol=1e-6)
    assert (new._context(x,mask)-new._context(changed,mask)).norm() > 1e-4


def test_ordered_zero_map_preserves_source_and_handles_missing_segments():
    base,_,new = models()
    x = torch.randn(2,128,23)
    mask = torch.ones_like(x,dtype=torch.bool)
    mask[0,:64] = False
    mask[1,:] = False
    with torch.no_grad():
        expected = base(x,mask)
        actual = new(x,mask)
        for key in expected:
            torch.testing.assert_close(actual[key],expected[key],atol=0,rtol=0)
        assert torch.isfinite(new._context(x,mask)).all()
    assert sum(p.numel() for p in new.persistent_parameters()) == 1024
