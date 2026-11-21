"""Same fast bias and clock, but source-constant instead of cross information.

This is an explicit ablation, never a missing-model fallback. The constant
was fixed from the source TRAIN signal targets before any EAST experiment.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import torch
from torch import nn

from . import simple_data
from .cross_density import CrossDensityModel, load_head, snapshot
from .redesign_20261004.run import (DEFAULT_SOURCE, DEFAULT_ORDER, DEFAULT_ROSTER,
    load_parent, load_roster, validate_order, validate_roster_rate_contract, run_arm)
from .simple_online import _durable_json, _hash


class ConstantHead(nn.Module):
    def __init__(self, reference, like):
        super().__init__(); self.register_buffer('reference',like.new_tensor(reference))

    def forward(self, features): return self.reference.expand(len(features),25)


class ConstantDensityModel(CrossDensityModel):
    def __init__(self,parent,reference):
        if not np.isfinite(reference):raise ValueError('explicit finite source constant required')
        super().__init__(parent)
        self.cross_head=ConstantHead(float(reference),self.center)

    def cross_features(self,values,available):
        # The control deliberately receives NO cross-channel information.
        # The risk forward and direct forecast inverse are unchanged.
        return values.new_zeros(len(values),152)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--head',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('new output required')
    torch.set_num_threads(4);torch.manual_seed(131)
    parent,state,gate,_=load_parent(DEFAULT_SOURCE,'cuda',persistent_site='embedding')
    source_model=CrossDensityModel(parent).cuda().eval()
    receipt=load_head(args.head,source_model)
    screen=json.loads((args.head.parent/'validation.json').read_text())
    model=ConstantDensityModel(parent,screen['constant']).cuda().eval()
    original={k:v.detach().clone() for k,v in parent.state_dict().items()}
    order=json.loads(DEFAULT_ORDER.read_text());ids,roster=load_roster(DEFAULT_ROSTER)
    validate_order(order,ids);validate_roster_rate_contract(roster,'nominal')
    args.output.mkdir(parents=True)
    code=snapshot(args.output)
    with zipfile.ZipFile(args.output/'code_snapshot.zip','a',zipfile.ZIP_DEFLATED) as z:
        z.write(Path(__file__),'cross_density_constant.py')
    protocol=dict(scope='DEVELOPMENT EXPLICIT CONSTANT AUXILIARY ABLATION',
        source_sha256=gate['checkpoint_sha256'],threshold=float(state['threshold']),
        source_head_sha256=_hash(args.head),source_receipt=receipt,
        source_reference=screen['constant'],reference_origin='same source TRAIN target median',
        reference_uses_target_information=False,selected_shots=120,temporary_parameters=1,
        stride=25,step=1.,radius=12.,objective='forward_holdout',
        source_training=False,jtext_online=False,target_scaler_fit=False,inter_updates=False,
        risk_correction_head=False,alarm_tuning=False,natural_outcomes=True,
        order_sha256=_hash(DEFAULT_ORDER),roster_sha256=_hash(DEFAULT_ROSTER),
        control_code_sha256=_hash(Path(__file__)),code_sha256=code,
        code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'))
    _durable_json(args.output/'protocol.json',protocol)
    _durable_json(args.output/'order.json',order);_durable_json(args.output/'roster.json',roster)
    provider=SimpleNamespace(window=simple_data.window,attach_outcome=simple_data.attach_outcome,
        read_shot=lambda device,sid:simple_data.read_shot(device,sid,rate_contract='nominal'))
    result=run_arm(model,order=order,arm='intra',threshold=float(state['threshold']),
        output_dir=args.output/'constant_intra',limit=120,device='cuda',stride=25,step=1.,radius=12.,
        objective='forward_holdout',data_api=provider)
    if any(not torch.equal(parent.state_dict()[k],v) for k,v in original.items()) or _hash(DEFAULT_SOURCE)!=gate['checkpoint_sha256']:
        raise RuntimeError('original source changed in constant control')
    _durable_json(args.output/'comparison.json',dict(constant_intra=result))
    print(json.dumps({k:v for k,v in result['metrics'].items() if k!='predictions'}),flush=True)


if __name__=='__main__':main()
