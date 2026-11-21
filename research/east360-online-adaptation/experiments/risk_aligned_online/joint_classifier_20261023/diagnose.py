"""Post-run description: training-set fit versus held-shot errors, no tuning."""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..diverse_support_20261023.run import make_parent,READ
from ..simple_online import _durable_json
from ..simple_train import metrics,sha
from ..warmstart_alignment_20261022.run import predict_many
from .run import REFERENCE,METHODS


def main(root):
    output=root/'posthoc_fit_diagnostic.json'
    if output.exists():raise FileExistsError(output)
    done=READ(root/'completed.json');p=READ(root/'protocol.json');results=READ(root/'per_shot.json')
    for name,h in done['artifact_sha256'].items():assert sha(root/name)==h
    torch.set_num_threads(2);torch.manual_seed(131);parent,_=make_parent();origin=state_cpu(parent)
    models={'source':parent}
    for method,path in [('first_fixed_head',REFERENCE/'fold0/blind_mixed/prepared.pt')]+[(m,root/'fold0'/m/'prepared.pt') for m in METHODS]:
        m=copy.deepcopy(parent);m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['model']);m.set_stage('frozen');models[method]=m
    rows={m:[] for m in models};reader=io23.read_outcome
    def guard(device,sid):
        assert device=='EAST' and sid in p['selected']
        return reader(device,sid)
    with patch.object(io23,'read_outcome',side_effect=guard):
        for sid in p['selected']:
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'));signal={k:s[k] for k in ('values','available')}
            curves={m:predict_many(model,dict(static=signal),device='cuda')['static'] for m,model in models.items()}
            s=simple_data.attach_outcome(s,'EAST',{})
            for m,v in curves.items():rows[m].append(dict(shot=sid,times=s['times'][s['valid']],scores=v[s['valid']],is_disrupt=s['is_disrupt'],down=s['down']))
    pairs={};updates={}
    for method in METHODS:
        a={r['shot']:r for r in results[method+'_static']['predictions']};b={r['shot']:r for r in results[method+'_intra']['predictions']}
        pairs[method]=dict(rescued=[s for s,r in a.items() if r['is_disrupt'] and not r['timely'] and b[s]['timely']],
            lost=[s for s,r in a.items() if r['timely'] and not b[s]['timely']],
            removed_fp=[s for s,r in a.items() if not r['is_disrupt'] and r['alarm'] and not b[s]['alarm']],
            added_fp=[s for s,r in a.items() if not r['is_disrupt'] and not r['alarm'] and b[s]['alarm']])
        u=[r for sid in p['shot_ids'] for r in READ(root/'fold0/shots'/f'{sid}_updates.json')[method] if r['gradient_steps']]
        updates[method]=dict(steps=len(u),loss_decreased=sum(r['loss_after']<r['loss_before'] for r in u),
            nonzero_parameter_change=sum(r['parameter_delta_norm']>0 for r in u),max_change=max(r['parameter_delta_norm'] for r in u))
    assert_unchanged(parent,origin)
    report=dict(completed_sha256=sha(root/'completed.json'),script_sha256=sha(Path(__file__)),
        warning='POSTHOC training-set fit diagnostic, NOT new validation or model selection',
        train_fit={m:metrics(r,p['threshold'],lead=30.) for m,r in rows.items()},intra_flips=pairs,updates=updates)
    _durable_json(output,report)
    print(dict(train_fit={m:{k:v[k] for k in ('tp','fp','n_disrupt','n_safe')} for m,v in report['train_fit'].items()},intra_flips=pairs,updates=updates),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
