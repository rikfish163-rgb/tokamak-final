import numpy as np
import torch

from experiments.risk_aligned_online.simple23 import Config,Simple23
from experiments.risk_aligned_online.temporal_relation_20261029.model import OrderJudge,permutation
from experiments.risk_aligned_online.temporal_relation_20261029.diagnose import probe


def test_independent_window_gradients_match_single_window_probes():
    torch.set_num_threads(1);torch.manual_seed(12);torch.backends.mha.set_fastpath_enabled(False)
    base=Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23)).eval().requires_grad_(False)
    net=OrderJudge().eval().requires_grad_(False)
    x=torch.randn(2,128,23);a=torch.ones_like(x,dtype=torch.bool);future=torch.randn_like(x)
    p=permutation(2,np.random.default_rng(8));rows=probe(base,net,x,a,future,a,p,None)
    for i in range(2):
        one=probe(base,net,x[i:i+1],a[i:i+1],future[i:i+1],a[i:i+1],p[i:i+1],None)[0]
        assert rows[i]['eligible'] and one['eligible']
        for key in ('margin','gradient_norm','future_logit_derivative','future_logit_change','updated_margin'):
            np.testing.assert_allclose(rows[i][key],one[key],atol=2e-6,rtol=1e-3)
    assert all(p.grad is None for p in base.parameters())
