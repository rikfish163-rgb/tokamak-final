"""Source event/hazard similarity routes past supervised encoder gradients.

No query outcome, event label or new risk head. The query is an arrived raw
window; the historical gradients are supervised. This is not pure SSL.
"""
import numpy as np
import torch

from ..event_pairing_20261024.learning import features,mean_features,reverse_features,Encoder
from ..paired_feedback_20261022.model import initial_encoder
from ..trajectory_memory_20261024.learning import (
    observed_window,select_weights,combine_step,predict as stream_predict,
)

ARMS=('risk','mean','shuffle','family','reverse','uniform')


def embed(x,models,mean,scale,mode,device):
    if mode=='mean':x=mean_features(x)
    elif mode=='reverse':x=reverse_features(x)
    elif mode not in ('risk','shuffle','family'):raise ValueError('unknown signal view')
    if len(models)!=3:raise ValueError('all three source seeds required')
    xx=torch.as_tensor(((x-mean)/scale).astype('float32'),device=device)
    with torch.no_grad():out=np.concatenate([m(xx).cpu().numpy() for m in models],axis=-1)
    return out.astype('float64')/np.sqrt(3.)


def prepare_bank(bank,heads,normalizer,device='cuda'):
    bank['normalizer']=normalizer;bank['metric_models']={};bank['metric_bank']={};bank['metric_temperature']={}
    x=np.stack([features(a,b,bank['center'],bank['scale']) for a,b in zip(bank['x'],bank['mask'])])
    for mode in ('risk','mean','shuffle','family'):
        models=[]
        for seed in (131,132,133):
            m=Encoder().to(device);m.load_state_dict(heads[f'{mode}_{seed}']);m.eval()
            for p in m.parameters():p.requires_grad_(False)
            models.append(m)
        bank['metric_models'][mode]=models
        z=embed(x,models,normalizer['mean'],normalizer['scale'],mode,device)
        bank['metric_bank'][mode]=z
        d=((z[:,None]-z[None,:])**2).sum(-1)
        d[bank['shots'][:,None]==bank['shots'][None,:]]=np.inf
        temperature=float(np.median(d.min(1)))
        if not np.isfinite(temperature) or temperature<=0:raise ValueError('metric temperature unidentified')
        bank['metric_temperature'][mode]=temperature


def update_all(model,signal,count,bank,device='cuda'):
    x,m=observed_window(signal,count)
    query=features(x,m,bank['center'],bank['scale'])[None]
    enough=int((query[0,207:].reshape(8,23).sum(0)>0).sum())>=8
    states={};receipts={};parent=initial_encoder(model);length=None
    for arm in ARMS:
        mode='risk' if arm in ('reverse','uniform') else arm
        temperature=bank['metric_temperature'][mode]
        if arm=='uniform':
            d=np.full(len(bank['shots']),np.inf)
            for sid in np.unique(bank['shots']):
                ids=np.flatnonzero(bank['shots']==sid);d[ids[len(ids)//2]]=0.
        else:
            q=embed(query,bank['metric_models'][mode],bank['normalizer']['mean'],
                    bank['normalizer']['scale'],arm,device)[0]
            d=((bank['metric_bank'][mode]-q)**2).sum(1)
        if not enough:d[:]=np.inf
        ix,w=select_weights(d,bank['shots'],bank['labels'],temperature)
        if not len(ix):
            states[arm]=initial_encoder(model)
            r=dict(gradient_norm=0.,requested_step_norm=0.,step_norm=0.,reason='fewer_than8_observed_channels')
        else:
            states[arm],r=combine_step(parent,bank['gradients'][ix],torch.as_tensor(w,device=device),
                                      learning_rate=.01,length=length)
        if arm=='risk':length=r['requested_step_norm']
        receipts[arm]=dict(r,selected=ix.tolist(),weights=w.tolist(),distances=d[ix].tolist(),
            last_observation=count-1,first_affected_prediction=count)
    return states,receipts


def predict(model,signal,bank,device='cuda'):
    return stream_predict(model,signal,bank,device,arms=ARMS,updater=update_all)
