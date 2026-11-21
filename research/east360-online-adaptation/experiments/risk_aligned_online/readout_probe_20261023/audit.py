"""Independent count/gradient checks, exact fit and selected GPU replay."""
import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..increment_forecast_20261023.learning import predict
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..temporal_scope_20261023.audit import recount
from .run import READ,ARMS,KINDS,REFERENCE,BOTH,models_for,training_features,install_head,static_scores,HEAD
from .fit import endpoint_manifest,fit_readout,RIDGE


def main(root):
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    start=time.monotonic();p=READ(root/'protocol.json');done=READ(root/'completed.json')
    for name,h in done['artifact_sha256'].items():assert sha(root/name)==h,name
    for name,h in p['code_sha256'].items():assert sha(io23.ROOT/name)==h,name
    for name,h in p['input_sha256'].items():assert sha(name)==h,name
    for path,h in p['reference_completed_sha256'].items():assert sha(Path(path)/'completed.json')==h
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    torch.set_num_threads(2);torch.manual_seed(131)
    result=recount(root,p,ARMS)
    _durable_json(root/'threshold_diagnostic.json',result)
    access=READ(root/'training_label_access.json');fits=[];replays=[];unchanged_updates=0
    for plan in p['folds']:
        f=plan['fold'];out=root/f'fold{f}';selected=READ(out/'selection.json')['selected']
        assert selected==sorted(READ(REFERENCE/f'fold{f}'/'selection.json')['selected']['diverse'])
        assert [r['shot'] for r in access if r['fold']==f]==selected
        assert len(selected)==20 and not set(selected)&set(plan['held'])
        target=[simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{}) for sid in selected]
        manifest=endpoint_manifest(target);assert manifest==READ(out/'endpoint_manifest.json')
        models,_=models_for(f)
        def deny(*args,**kwargs):raise AssertionError('no outcome access during replay')
        with patch.object(io23,'read_outcome',side_effect=deny):
            for kind,model in models.items():
                with np.load(out/f'{kind}_fit.npz') as z:
                    x=z['x'].copy();y=z['y'].copy();weights=z['weights'].copy();origin=z['origin'].copy();theta=z['theta'].copy()
                np.testing.assert_array_equal(y,[r['label'] for r in manifest])
                np.testing.assert_array_equal(weights,[r['weight'] for r in manifest])
                # Different derivative implementation: torch autograd, not the scipy analytic Jacobian.
                tt=torch.tensor(theta,dtype=torch.float64,requires_grad=True)
                z=torch.tensor(x,dtype=torch.float64)@tt[:-1]+tt[-1]
                loss=(torch.tensor(weights)*F.binary_cross_entropy_with_logits(z,torch.tensor(y,dtype=torch.float64),reduction='none')).sum()
                loss=loss+RIDGE/2*(tt-torch.tensor(origin)).square().sum()
                gradient=torch.autograd.grad(loss,tt)[0]
                assert gradient.abs().max()<1e-6
                rerun,info=fit_readout(x,y,weights,origin)
                np.testing.assert_array_equal(rerun,theta)
                assert info==READ(out/f'{kind}_fit.json')
                if f==0:np.testing.assert_array_equal(training_features(model,target,manifest),x)
                before=state_cpu(model);install_head(model,theta);assert_unchanged(model,before,HEAD)
                fits.append(dict(fold=f,kind=kind,gradient_inf=float(gradient.abs().max()),fit_exact=True,features_replayed=f==0))
            if f==0:
                for sid in (54314,63226):
                    assert sid in plan['held']
                    s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                    signal={k:s[k] for k in ('values','available')}
                    with np.load(out/'shots'/f'{sid}.npz') as z:
                        np.testing.assert_array_equal(static_scores(models['source'],signal),z['source_readout'])
                        replays.append(dict(fold=f,shot=sid,kind='source',mode='static'))
                        for kind in ('first','both'):
                            before=state_cpu(models[kind]);scores,updates,_=predict(models[kind],signal)
                            for a,curve in scores.items():np.testing.assert_array_equal(curve,z[kind+'_readout_'+a])
                            assert updates==READ(out/'shots'/f'{sid}_updates.json')[kind]
                            assert_unchanged(models[kind],before)
                            replays.append(dict(fold=f,shot=sid,kind=kind,mode='static+intra'))
        for sid in plan['held']:
            updates=READ(out/'shots'/f'{sid}_updates.json')
            assert updates['first']==READ(REFERENCE/f'fold{f}'/'shots'/f'{sid}_updates.json')['diverse']
            assert updates['both']==READ(BOTH/f'fold{f}'/'shots'/f'{sid}_updates.json')
            unchanged_updates+=sum(len(v) for v in updates.values())
        print(dict(stage='audit',fold=f,seconds=time.monotonic()-start),flush=True)
    verified=dict(status='VERIFIED_COUNTS_FITS_AND_SELECTED_REPLAY',completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),threshold_diagnostic_sha256=sha(root/'threshold_diagnostic.json'),
        shot_arm_checks=result['shot_arm_checks'],fits=fits,replays=replays,
        identical_auxiliary_update_receipts=unchanged_updates,
        source_unchanged=sha(DEFAULT_PARENT)==PARENT_SHA256,seconds=time.monotonic()-start)
    _durable_json(root/'verification.json',verified);print(verified,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
