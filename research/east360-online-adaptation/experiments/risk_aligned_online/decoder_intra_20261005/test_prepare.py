import numpy as np
import torch

from experiments.risk_aligned_online.decoder_intra_20261005.prepare import cache_events,risk_probe
from experiments.risk_aligned_online.decoder_intra_20261005.test_model import make
from experiments.risk_aligned_online.ffn_intra_20261005.model import FfnModel


def test_frozen_token_cache_retains_causal_anchor_and_forward_features():
    base,_,_=make();model=FfnModel(base)
    before={k:v.clone() for k,v in model.state_dict().items()}
    rng=np.random.default_rng(133)
    shot=dict(device='J-TEXT',shot=1,is_disrupt=False,down=0.,
        times=np.arange(60)/1000,valid=np.ones(60,dtype=bool),
        values=rng.normal(size=(60,23)).astype('float32'),available=np.ones((60,23),dtype=bool))
    cache,rows,excluded=cache_events(model,[shot])
    assert [r['origin'] for r in rows]==[0,26] and excluded==0
    assert cache['features'].shape==(2,17*64)
    assert not cache['features'].requires_grad
    assert cache['target'].shape==(2,25,23)
    assert torch.equal(cache['anchor'][0],torch.asinh(torch.as_tensor(shot['values'][0])))
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    # Changing signals that have not arrived by the origin cannot change its
    # encoder state/anchor; only the OFFLINE fitting target/activity may change.
    shot['values'][1:]*=10
    other,_,_=cache_events(model,[shot])
    assert torch.equal(cache['features'][0],other['features'][0])
    assert torch.equal(cache['anchor'][0],other['anchor'][0])
    assert risk_probe(model,shot).shape==(2,)
