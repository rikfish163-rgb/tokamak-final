"""Re-read signals, replay probe weights and independently recount CV20 losses."""
import argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from .run import (READ,REFERENCE,REF_SHA,io23,simple_data,apply_input_contract,sha,
    load_parent,DEFAULT_PARENT,PARENT_SHA256,SEEDS,ARMS,normalize,shot_weights,predict,
    train_pair,_durable_json)
from .model import EvidenceProbe,feedback_arrays


def main(root):
    done=READ(root/'completed.json');p=READ(root/'protocol.json')
    for f,h in done['artifact_sha256'].items():assert sha(root/f)==h,f
    for f,h in p['code_sha256'].items():assert sha(io23.ROOT/f)==h,f
    for f,h in p['consumed_sha256'].items():assert sha(REFERENCE/f)==h,f
    assert sha(REFERENCE/'completed.json')==REF_SHA
    torch.set_num_threads(2)
    base,_=load_parent(DEFAULT_PARENT);center=base.center.numpy();scale=base.scale.numpy()
    outcomes={r['shot']:r for r in READ(root/'evaluation_access.json')}
    stored={(r['shot'],r['seed'],r['arm']):r for r in READ(root/'per_shot.json')}
    records=[];replayed=0;retrained=False;checks=0

    def deny(*args,**kwargs):raise AssertionError('no outcomes during probe replay')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for fold in p['folds']:
            f=fold['fold'];folder=root/f'fold{f}'
            assert not set(fold['fit'])&set(fold['held'])
            with np.load(REFERENCE/f'fold{f}'/'fit.npz') as z:
                fit={k:z[k].copy() for k in ('features','ids','y','origins')}
                fit['delta']=z['target'][:,:5].copy();fit['mask']=z['valid'][:,:5].copy()
            n=normalize(fit['features'],fit['delta'],fit['mask'],shot_weights(fit['ids'],fit['y']))
            with np.load(folder/'normalization.npz') as z:
                for k,v in n.items():np.testing.assert_array_equal(v,z[k])
            if f==0:
                nets,log,receipt=train_pair(fit,131,n)
                saved=torch.load(folder/'weights_131.pt',map_location='cpu',weights_only=True)
                for k,net in nets.items():
                    for name,v in net.state_dict().items():torch.testing.assert_close(v.cpu(),saved[k][name],atol=0,rtol=0)
                assert log==READ(folder/'training_131.json')
                wanted=next(r for r in READ(root/'fit_receipts.json') if r['fold']==0 and r['seed']==131)
                assert receipt==wanted['samples_sha256'];retrained=True
            for sid in fold['held']:
                raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                with np.load(folder/f'{sid}_inputs.npz') as z:
                    a={k:z[k].copy() for k in z.files}
                with np.load(REFERENCE/f'fold{f}'/f'{sid}.npz') as z:
                    np.testing.assert_array_equal(a['features'],z['features'])
                    np.testing.assert_array_equal(a['origins'],z['origins'])
                np.testing.assert_array_equal(a['times'],raw['times'][a['origins']])
                np.testing.assert_array_equal(a['arrived_times'],raw['times'][a['origins']+5])
                d,m,od,om=feedback_arrays({k:raw[k] for k in ('values','available')},a['origins'],center,scale)
                m[:,:,io23.RAW23.index('mode_number_n')]=False;om[:,:,io23.RAW23.index('mode_number_n')]=False
                for k,v in dict(delta=d,mask=m,earlier=od,earlier_mask=om).items():np.testing.assert_array_equal(a[k],v)
                r=outcomes[sid];disrupted=r['disrupted']
                use=(np.round((r['down']-a['times'])*1000,9)<=125)&(np.round((r['down']-a['arrived_times'])*1000,9)>30) if disrupted else np.ones(len(d),bool)
                for seed in SEEDS:
                    states=torch.load(folder/f'weights_{seed}.pt',map_location='cpu',weights_only=True)
                    nets={k:EvidenceProbe().cuda().eval() for k in states}
                    for k,net in nets.items():net.load_state_dict(states[k])
                    out=predict(nets,a['features'],d,m,od,om,n)
                    with np.load(folder/f'{sid}_{seed}.npz') as z:
                        for arm in ARMS:np.testing.assert_array_equal(out[arm],z[arm])
                    replayed+=1
                    for arm,z in out.items():
                        zz=torch.as_tensor(z[use],dtype=torch.float64)
                        loss=float(F.binary_cross_entropy_with_logits(zz,torch.full_like(zz,float(disrupted))))
                        inc=z[use]-out['mispaired_prior' if arm.startswith('mispaired') else 'prior'][use]
                        expected=stored[sid,seed,arm]
                        assert expected['n']==len(zz)
                        np.testing.assert_allclose(loss,expected['bce'],atol=1e-12,rtol=1e-12)
                        assert float(np.median(inc))==expected['median_increment']
                        records.append(expected);checks+=1
    summary=READ(root/'summary.json')['arms']
    for arm in ARMS:
        rows=[r for r in records if r['arm']==arm]
        safe=np.mean([r['bce'] for r in rows if not r['disrupted']]);danger=np.mean([r['bce'] for r in rows if r['disrupted']])
        np.testing.assert_allclose([safe,danger,(safe+danger)/2],
            [summary[arm]['safe_bce'],summary[arm]['danger_bce'],summary[arm]['balanced_bce']],atol=1e-12,rtol=1e-12)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    for sid,r in outcomes.items():assert io23.read_outcome('EAST',sid)==(r['disrupted'],r['down'])
    result=dict(status='VERIFIED_ARRIVED_EVIDENCE_CV20',completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),replayed_shot_seed_pairs=replayed,independent_bce_recounts=checks,
        fold0_seed131_training_exactly_reproduced=retrained,source_unchanged=True,test360_accessed=False,
        scope='Numerical/input/replay verification only; not independent scientific confirmation')
    _durable_json(root/'verification.json',result);print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
