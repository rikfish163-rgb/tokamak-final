"""Single-endpoint observed-window latency; no batching of future endpoints."""
import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch

from .. import simple_data
from ..causal_input import apply_input_contract
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .model import LocalTransition
from .run import READ,load_model


def measure(root):
    p,done=READ(root/'protocol.json'),READ(root/'completed.json')
    assert done['status']=='COMPLETED'
    torch.set_num_threads(2);torch.manual_seed(131)
    parent,_=load_model();model=LocalTransition(copy.deepcopy(parent.base)).cuda()
    ck=root/'adaptive_trained.pt';assert sha(ck)==done['checkpoint_sha256']['adaptive_trained']
    model.load_state_dict(torch.load(ck,map_location='cuda',weights_only=True)['model'])
    model.eval().requires_grad_(False)
    sid=p['selected_ids'][0]
    s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
    assert len(s['values'])>=248
    signal=transform({k:s[k] for k in ('values','available')},READ(Path(p['reference'])/'input_map.json'))
    def once(t,mode):
        torch.cuda.synchronize();start=time.perf_counter()
        x,m=simple_data.window(signal,t,128)
        x=torch.as_tensor(x[None],device='cuda');m=torch.as_tensor(m[None],device='cuda')
        result=model(x,m,mode=mode)['logit'].sigmoid().detach().cpu().numpy()
        torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
        assert np.isfinite(result).all()
        return elapsed
    for t in range(128,148):
        once(t,'prior');once(t,'fit')
    elapsed={'prior':[],'fit':[]}
    # Alternate mode order to reduce a systematic thermal/order bias.
    for t in range(148,248):
        for mode in (('prior','fit') if t%2 else ('fit','prior')):
            elapsed[mode].append(once(t,mode))
    summary={k:dict(n=len(v),median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),
                    maximum_ms=float(max(v)),mean_ms=float(np.mean(v))) for k,v in elapsed.items()}
    result=dict(shot=sid,checkpoint_sha256=sha(ck),script_sha256=sha(Path(__file__)),
        gpu=torch.cuda.get_device_name(),batch_size=1,warmup_endpoints_per_mode=20,
        includes='CPU causal-window construction, host/device copies, model and local solve, sigmoid and CPU output; CUDA synchronization',
        excludes='file reading, fixed alignment construction, diagnostic acquisition/transport, alarm decision and production scheduling',
        no_realtime_certification=True,mean_sampling_interval_ms=float(np.diff(s['times']).mean()*1000),
        measurements_ms=elapsed,summary=summary)
    _durable_json(root/'latency.json',result);print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);measure(p.parse_args().root)
