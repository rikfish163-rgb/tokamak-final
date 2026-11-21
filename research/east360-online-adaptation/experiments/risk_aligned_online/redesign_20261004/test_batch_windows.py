from types import SimpleNamespace

import numpy as np
import torch

from . import intra
from .. import simple_data


def test_batch_windows_keeps_the_original_padded_causal_arrays():
    values=np.arange(140*23,dtype=np.float32).reshape(140,23)
    values[7,4]=np.nan
    available=np.ones(values.shape,bool)
    available[8,5]=False
    signals={'values':values,'available':available}
    model=SimpleNamespace(cfg=SimpleNamespace(window=128),center=torch.zeros(23))
    endpoints=[0,7,9,127,139]
    x,m=intra._batch_windows(simple_data,signals,endpoints,model,'cpu')
    pairs=[simple_data.window({k:v[:end+1] for k,v in signals.items()},end,128)
           for end in endpoints]
    expected_x=torch.from_numpy(np.stack([p[0] for p in pairs]))
    expected_m=torch.from_numpy(np.stack([p[1] for p in pairs])) & torch.isfinite(expected_x)
    torch.testing.assert_close(x,expected_x,rtol=0,atol=0,equal_nan=True)
    assert torch.equal(m,expected_m)


def test_custom_provider_still_sees_only_each_arrived_prefix():
    seen=[]
    class Provider:
        @staticmethod
        def window(prefix,end,length):
            assert set(prefix)=={'values','available'}
            assert len(prefix['values'])==end+1
            seen.append(end)
            return simple_data.window(prefix,end,length)
    signals={'values':np.zeros((140,23),np.float32),'available':np.ones((140,23),bool)}
    model=SimpleNamespace(cfg=SimpleNamespace(window=128),center=torch.zeros(23))
    intra._batch_windows(Provider,signals,[2,20,80],model,'cpu')
    assert seen==[2,20,80]
