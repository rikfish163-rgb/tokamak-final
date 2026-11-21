"""Post-hoc training/held-shot gap at the same five-point feedback deadline."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .run import (READ,REFERENCE,io23,simple_data,sha,SEEDS,tensors,_durable_json)
from .model import EvidenceProbe


@torch.no_grad()
def main(root):
    if (root/'fit_held_gap.json').exists():raise FileExistsError(root/'fit_held_gap.json')
    done=READ(root/'completed.json');p=READ(root/'protocol.json')
    assert READ(root/'verification.json')['completed_sha256']==sha(root/'completed.json')
    torch.set_num_threads(2);rows=[]
    for fold in p['folds']:
        f=fold['fold'];folder=root/f'fold{f}'
        with np.load(REFERENCE/f'fold{f}'/'fit.npz') as z:
            h=z['features'];d=z['target'][:,:5];m=z['valid'][:,:5];ids=z['ids'];ends=z['origins']
        with np.load(folder/'normalization.npz') as z:n={k:z[k] for k in z.files}
        critical={};outcomes={}
        for sid in fold['fit']:
            disrupted,down=io23.read_outcome('EAST',sid)
            times=simple_data.read_shot('EAST',sid,rate_contract='nominal')['times']
            use=np.flatnonzero(ids==sid);origin=ends[use]
            keep=(np.round((down-times[origin])*1000,9)<=125)&(np.round((down-times[origin+5])*1000,9)>30) if disrupted else np.ones(len(use),bool)
            critical[sid]=use[keep];outcomes[sid]=disrupted
        for seed in SEEDS:
            state=torch.load(folder/f'weights_{seed}.pt',map_location='cpu',weights_only=True)
            net=EvidenceProbe().cuda().eval();net.load_state_dict(state['real'])
            prior=[];post=[]
            for lo in range(0,len(h),128):
                sl=slice(lo,lo+128);a,b=net(*tensors(h[sl],d[sl],m[sl],n))
                prior.append(a.cpu().numpy());post.append(b.cpu().numpy())
            prior=np.concatenate(prior);post=np.concatenate(post)
            for sid,use in critical.items():
                if not len(use):raise ValueError('no matching training evaluation interval')
                row=dict(fold=f,shot=sid,seed=seed,disrupted=bool(outcomes[sid]),n=len(use))
                for name,z in (('prior',prior),('real',post)):
                    a=z[use].astype(float);row[name+'_bce']=float(np.mean(np.logaddexp(0,a)-float(outcomes[sid])*a))
                row['median_increment']=float(np.median(post[use]-prior[use]));rows.append(row)
    summary={}
    for group,y in (('safe',False),('danger',True)):
        selected=[r for r in rows if r['disrupted']==y]
        summary[group]=dict(shot_fold_seed_records=len(selected),unique_shots=len(set(r['shot'] for r in selected)),
            prior_bce=float(np.mean([r['prior_bce'] for r in selected])),
            real_bce=float(np.mean([r['real_bce'] for r in selected])),
            correct_direction_records=sum((r['median_increment']>0 if y else r['median_increment']<0) for r in selected))
    result=dict(scope='POST_HOC_DIAGNOSTIC; training identities repeat across folds/seeds, not independent samples',
        completed_sha256=sha(root/'completed.json'),code_sha256=sha(Path(__file__)),training=summary,rows=rows)
    _durable_json(root/'fit_held_gap.json',result);print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
