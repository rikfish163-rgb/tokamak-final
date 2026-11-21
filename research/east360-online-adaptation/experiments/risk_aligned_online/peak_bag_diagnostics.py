"""Post-run functional invariance, NOT an online score or validation dataset."""
import argparse
import json
from pathlib import Path

import torch

from .ended_decoder import DecoderFfnModel
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from .simple_online import _durable_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    output=args.run/'final_head_functional_check.json'
    if output.exists():parser.error('new functional check receipt required')
    torch.set_num_threads(2)
    base,_=load_parent(DEFAULT_PARENT)
    model=DecoderFfnModel(base).requires_grad_(False).eval()
    frozen={k:v.clone() for k,v in model.base.state_dict().items()}
    torch.manual_seed(131)
    x=torch.randn(3,128,23)*model.scale+model.center
    available=torch.rand(3,128,23)>.1
    fast=model.initial_fast(3)
    original=model(x,available,fast)['logit'].detach()
    errors={}
    for arm in ('bag_early_fixed','bag_early_continual'):
        saved=torch.load(args.run/arm/'final_head.pt',map_location='cpu',weights_only=True)
        model.decoder.load_state_dict(saved['decoder'])
        current=model(x,available,fast)['logit'].detach()
        torch.testing.assert_close(current,original,rtol=0,atol=0)
        for key,value in model.base.state_dict().items():
            torch.testing.assert_close(value,frozen[key],rtol=0,atol=0)
        errors[arm]=float((current-original).abs().max())
    _durable_json(output,dict(zero_fast_logit_max_errors=errors,synthetic_windows=3,
        source_state_unchanged=True,online_performance_evidence=False,
        interpretation='final auxiliary head cannot directly alter zero-fast risk; not a causal replay score'))
    print(json.dumps(json.loads(output.read_text())),flush=True)


if __name__=='__main__':main()
