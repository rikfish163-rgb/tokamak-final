"""Post-hoc source TRAIN retention only, on the exact replay draw manifests.

This cannot establish source TEST quality or causal catastrophic forgetting.
No target outcomes are read and no model is updated.
"""
import argparse
import copy
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..credit_horizon_20261023.run import READ,load_model
from ..redesign_20261004.prepare_prior import load_source_train,DEFAULT_PARENT,PARENT_SHA256
from ..simple_train import sha
from ..simple_online import _durable_json
from .learning import replay_loss
from .run import ARMS,PREVIOUS,PREVIOUS_SHA


def main(root):
    path=root/'source_retention_comparison.json'
    if path.exists():raise FileExistsError(path)
    assert READ(root/'verification.json')['status']=='VERIFIED'
    assert sha(PREVIOUS/'completed.json')==PREVIOUS_SHA and sha(DEFAULT_PARENT)==PARENT_SHA256
    start=time.monotonic();torch.set_num_threads(2);p=READ(root/'protocol.json')
    ids=io23.source_roles()['train'];read=io23.read_outcome;access=[]
    def guard(device,sid):
        assert device=='J-TEXT' and sid in ids
        access.append(sid);return read(device,sid)
    with patch.object(io23,'read_outcome',side_effect=guard):source,source_ids=load_source_train()
    assert source_ids==access==ids;byid={s['shot']:s for s in source};rows=[]
    for plan in p['folds']:
        f=plan['fold'];warm=load_model(f);models={'warm':warm}
        for kind,folder in (('no_replay',PREVIOUS),('replay',root)):
            for a in ARMS:
                m=copy.deepcopy(warm)
                m.load_state_dict(torch.load(folder/f'fold{f}'/f'{a}.pt',map_location='cuda',weights_only=True)['model'])
                m.set_stage('frozen');models[f'{kind}_{a}']=m
        values={a:[] for a in models};history=READ(root/f'fold{f}'/'training.json')
        for r in history:
            examples=[]
            for e in r['source_examples']:
                s=byid[e['shot']];assert s['valid'][e['end']] and int(s['y'][e['end']]>.5)==e['label']
                x,m=simple_data.window(s,e['end'],128);examples.append((x,m,np.float32(e['label'])))
            x,m,y=[torch.as_tensor(np.stack(v),device='cuda') for v in zip(*examples)]
            with torch.no_grad():
                for a,model in models.items():values[a].append(float(replay_loss(model,x,m,y)))
        saved=READ(root/f'fold{f}'/'source_training_retention.json')
        assert values['warm']==saved['warm']
        for a in ARMS:assert values[f'replay_{a}']==saved[a]
        rows.append(dict(fold=f,mean_bce={a:float(np.mean(v)) for a,v in values.items()},batch_bce=values))
    _durable_json(path,dict(stage='POSTHOC_TRAINING_RETENTION',code_sha256=sha(Path(__file__)),
        result_sha256=sha(root/'completed.json'),source_checkpoint_unchanged=sha(DEFAULT_PARENT)==PARENT_SHA256,
        source_access_ids=ids,folds=rows,seconds=time.monotonic()-start,
        limitation='Exact training draws, not TEST220, not evidence of learned physics or sufficient explanation of EAST errors.'))
    print([dict(fold=r['fold'],**r['mean_bce']) for r in rows],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
