import numpy as np
import torch

from experiments.risk_aligned_online.ffn_intra_20261005.test_stream import case
from experiments.risk_aligned_online.ffn_intra_20261005.stream import predict_shot
from experiments.risk_aligned_online.decoder_intra_20261005.model import AnchoredFfnModel


def test_untrained_zero_readout_has_no_ffn_adaptation_effect():
    parent,shot=case(70);model=AnchoredFfnModel(parent.base,initialize=True)
    static,_=predict_shot(model,shot,intra=False)
    updated,audit=predict_shot(model,shot,intra=True,step=.1,radius=1.)
    # Static batching and maturity-delimited batching differ in GEMM batch size.
    # CPU float32 differs by one ULP; exact zero gradient/state is tested too.
    np.testing.assert_allclose(static,updated,rtol=0,atol=2e-7)
    assert all(e['delta_norm']==0 and e['gradient_norm']==0 for e in audit['events'])


def test_anchored_replay_maturity_suffix_and_per_shot_reset():
    parent,shot=case(85);model=AnchoredFfnModel(parent.base,initialize=False)
    original={k:v.clone() for k,v in model.state_dict().items()}
    static,_=predict_shot(model,shot,intra=False)
    first,audit=predict_shot(model,shot,intra=True,step=.1,radius=1.)
    np.testing.assert_allclose(first[:26],static[:26],rtol=0,atol=2e-7)
    assert audit['events'][0]['effective_index']==26
    changed={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in shot.items()}
    changed['values'][53:]*=10
    second,_=predict_shot(model,changed,intra=True,step=.1,radius=1.)
    np.testing.assert_array_equal(first[:53],second[:53])
    again,_=predict_shot(model,shot,intra=True,step=.1,radius=1.)
    np.testing.assert_array_equal(first,again)
    assert all(torch.equal(v,original[k]) for k,v in model.state_dict().items())
