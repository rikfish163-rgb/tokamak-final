"""Post-hoc channel/time attribution of the failed slope probe; no new model."""
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
from .matured_slope import SlopeMemory


@torch.no_grad()
def run(root):
    path=root/'failure_attribution.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(2)
    p=read(root,'protocol.json');done=read(root,'complete.json')
    stream=Path(p['stream']);mapping=read(stream,'input_map.json');raw_hash=read(stream,'input_hashes.json')
    model=SlopeMemory(64,torch.tensor(p['prior']),torch.tensor(p['ridge']),torch.tensor(p['weights']))
    base,_=load_parent(DEFAULT_PARENT);weights=torch.tensor(p['weights'])
    contributions={m:[] for m in ('memory','off','persistence')};details=[];timing=[]
    for saved in done['rows']:
        sid,i=saved['shot'],saved['index']
        if sha(io23.path_for('EAST',sid)) != raw_hash[str(sid)]:
            raise ValueError('raw input changed')
        raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        sig={k:raw[k] for k in ('values','available')}
        if i>20:sig=transform(sig,mapping)
        x,a=patch_signal(sig,base.center,base.scale)
        t=model.trace(x,a,mode='memory')
        sealed=read(root/'seals',f'{i:04d}_{sid}.json')
        if hashlib.sha256(t['forecast'].numpy().tobytes()).hexdigest()!=sealed['sha256']['memory']:
            raise ValueError('cannot reproduce issued forecast')
        flag,down=io23.read_outcome('EAST',sid)
        n=len(x);y=torch.stack([x[h:n-4+h]-x[:-4] for h in range(1,5)],1)
        valid=torch.stack([a[h:n-4+h]&a[:-4] for h in range(1,5)],1)
        times=np.stack([raw['times'][(np.arange(n-4)+h+1)*8-1] for h in range(1,5)],1)
        within=times<=down if flag else np.ones_like(times,dtype=bool)
        w=valid*weights*torch.tensor(within)[...,None]
        forecasts={'memory':t['forecast'][:-4],'off':t['baseline'][:-4],'persistence':torch.zeros_like(y)}
        for name,f in forecasts.items():
            element=(f-y).square()*w
            v=element.sum((0,1))/w.sum()
            if not np.isclose(float(v.sum()),saved['errors'][name],rtol=2e-6,atol=1e-8):
                raise ValueError('channel decomposition differs')
            contributions[name].append(v.numpy())
        extra=((forecasts['memory']-y).square()-y.square())*w
        flat=int(extra.argmax());o,h,c=np.unravel_index(flat,extra.shape)
        lead=(down-raw['times'][(o+1)*8-1])*1000 if flag else None
        details.append(dict(shot=sid,disruptive=flag,origin_patch=int(o),horizon=int(h+1),channel=io23.RAW23[c],
            source_coefficient=float(model.prior[h,c]),adapted_coefficient=float(t['coefficients'][o,h,c]),
            predicted_change=float(forecasts['memory'][o,h,c]),actual_change=float(y[o,h,c]),
            lead_ms=lead,weighted_excess=float(extra[o,h,c]/w.sum()),
            total_excess=saved['errors']['memory']-saved['errors']['persistence']))
        if flag:
            origins=raw['times'][(np.arange(n-4)+1)*8-1]
            leads=(down-origins)*1000
            bands={}
            for name,select in (('lead_over125',leads>125),('lead30_to125',(leads>30)&(leads<=125)),
                                ('lead0_to30',(leads>=0)&(leads<=30))):
                bw=w*torch.tensor(select)[:,None,None]
                bands[name]=dict(weight=float(bw.sum()),
                    numerator={m:float(((f-y).square()*bw).sum()) for m,f in forecasts.items()})
            timing.append(dict(shot=sid,bands=bands))
    average={m:np.mean(v,0) for m,v in contributions.items()}
    channels=[dict(channel=c,**{m:float(v[j]) for m,v in average.items()},
                   excess=float(average['memory'][j]-average['persistence'][j])) for j,c in enumerate(io23.RAW23)]
    channels.sort(key=lambda v:v['excess'],reverse=True)
    details.sort(key=lambda v:v['total_excess'],reverse=True)
    result=dict(scope='POSTHOC_FAILURE_LOCALIZATION_NOT_POLICY_OR_RISK',verified_forecast_seals=len(details),
        reference_complete_sha256=sha(root/'complete.json'),code_sha256=sha(Path(__file__)),
        channel_contributions=channels,largest_local_errors=details,disruptive_time_bands=timing)
    _durable_json(path,result)
    print(channels[:8],flush=True);print(details[:8],flush=True)
    for band in timing[0]['bands']:
        counts=sum(r['bands'][band]['weight']>0 for r in timing)
        values={m:np.mean([r['bands'][band]['numerator'][m]/r['bands'][band]['weight']
                          for r in timing if r['bands'][band]['weight']>0]) for m in forecasts}
        print(band,counts,values,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    run(parser.parse_args().root)
