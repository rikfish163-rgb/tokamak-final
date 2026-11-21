"""Post-hoc forecast decomposition; no training or risk-model result."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..delayed_recurrent_memory_20261025.model import patch_signal
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .forecast import DirectForecast, observed_features


@torch.no_grad()
def run(root, forecast_root):
    output = root/'level_change_diagnostic.json'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    p = read(root,'protocol.json'); done = read(root,'complete.json')
    if not done['complete']:
        raise ValueError('completed 360 component result required')
    stream = Path(p['stream']); mapping = read(stream,'input_map.json')
    source = read(forecast_root,'protocol.json')
    checkpoint = forecast_root/'forecast_epoch10.pt'
    if sha(checkpoint) != p['forecast_sha256']:
        raise ValueError('frozen forecast changed')
    f = DirectForecast().eval().requires_grad_(False)
    f.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
    base,_ = load_parent(DEFAULT_PARENT)
    weights = torch.tensor(read(Path(source['reference']),'forecast_weights.json')['weights'])
    rows = []
    for recorded in done['rows']:
        sid, index = recorded['shot'], recorded['index']
        raw = apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        signal = {k:raw[k] for k in ('values','available')}
        if index > 20:
            signal = transform(signal,mapping)
        x,a = patch_signal(signal,base.center,base.scale)
        features = observed_features(x,a); full=f(features)
        seal=read(root/'seals',f'{index:04d}_{sid}.json')
        if hashlib.sha256(full.numpy().tobytes()).hexdigest()!=seal['predictions_sha256']:
            raise ValueError('forecast reconstruction differs from sealed original')
        no_change=features.clone(); no_change[:,46:69]=0
        level=f(no_change); change=full-level
        y=torch.stack([x[h:len(x)-4+h]-x[:-4] for h in range(1,5)],1)
        valid=torch.stack([a[h:len(x)-4+h]&a[:-4] for h in range(1,5)],1)
        flag,down=io23.read_outcome('EAST',sid)
        times=np.stack([raw['times'][(np.arange(len(x)-4)+h+1)*8-1] for h in range(1,5)],1)
        within=times<=down if flag else np.ones_like(times,dtype=bool)
        w=valid*torch.tensor(within)[...,None]*weights; denominator=w.sum()
        def score(pred):
            return float(((pred[:-4]-y).square()*w).sum()/denominator)
        actual=score(full)
        if not np.isclose(actual,recorded['prediction'],atol=1e-8,rtol=1e-6):
            raise ValueError('full forecast error differs')
        rows.append(dict(shot=sid,index=index,disruptive=flag,full=actual,
            level_only=score(level),change_response_only=score(change),persistence=recorded['persistence'],
            level_energy=float((level[:-4].square()*w).sum()/denominator),
            change_energy=float((change[:-4].square()*w).sum()/denominator),
            full_energy=float((full[:-4].square()*w).sum()/denominator)))
    summary={}
    for section,part in (('full360',rows),('suffix340',rows[20:])):
        summary[section]={}
        for group in ('all','D','S'):
            subset=[r for r in part if group=='all' or r['disruptive']==(group=='D')]
            summary[section][group]={k:float(np.mean([r[k] for r in subset])) for k in
                ('full','level_only','change_response_only','persistence','level_energy','change_energy','full_energy')}
            summary[section][group]['change_response_improved']=sum(r['change_response_only']<r['persistence'] for r in subset)
    _durable_json(output,dict(scope='POSTHOC_COMPONENT_DECOMPOSITION_NOT_A_VALIDATED_NEW_METHOD',
        identity='F(level,change)=F(level,0)+[F(level,change)-F(level,0)] with availability unchanged',
        caution='Nonlinear response difference, not proof of causality or calibrated physical derivative; no risk results.',
        model_updates=0,summary=summary,rows=rows,code_sha256=sha(Path(__file__)),
        source_complete_sha256=sha(root/'complete.json')))
    print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('forecast_root',type=Path)
    args=parser.parse_args();run(args.root,args.forecast_root)
