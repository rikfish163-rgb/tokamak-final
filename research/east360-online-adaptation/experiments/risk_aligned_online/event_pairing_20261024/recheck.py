"""Repair screening evaluation without overwriting the original evidence."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23
from ..simple_online import _durable_json
from ..simple_train import sha
from .evaluation import pair_out_auc,qualifies
from .learning import Encoder,mean_features,reverse_features,neighbour_score
from .run import READ,MODES,encode,metrics,fit,DEFAULT_PARENT,PARENT_SHA256


def main(root):
    output=root/'same_history_recheck'
    if output.exists():raise FileExistsError(output)
    done=READ(root/'completed.json');p=READ(root/'protocol.json')
    for f,h in done['artifact_sha256'].items():assert sha(root/f)==h,f
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    def read(name):
        with np.load(root/name) as z:return {k:z[k].copy() for k in z.files}
    data=read('source.npz');east=read('east20.npz');norms=read('standardizer.npz')
    scores=read('scores.npz');train=data['role']=='train';val=data['role']=='val'
    tr={k:v[train] for k,v in data.items()};va={k:v[val] for k,v in data.items()}
    norm=lambda x:((x-norms['mean'])/norms['scale']).astype('float32')
    heads=torch.load(root/'encoders.pt',map_location='cpu',weights_only=True)
    original=READ(root/'results.json');results={};details={};val_checks=0
    output.mkdir()
    _durable_json(output/'protocol.json',dict(original_completed_sha256=sha(root/'completed.json'),
        scope='POST_HOC_EVALUATION_REPAIR; development only',
        reason='pooled leave-one-shot-out ranks compare different historical banks',
        comparison='64 pairs; hold out both entire shots; same3D15S bank for each pair',
        extra_gate='target pair-AUC >.5 for each of all3 fixed seeds',
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in Path(__file__).parent.glob('*.py')},
        new_EAST_training=False,test360_read=False,source_risk_changed=False))
    def evaluate(name,a,b,c,query=None):
        nonlocal val_checks
        vs=neighbour_score(b,a,tr['shot'],tr['y'],exclude=np.full(len(b),-1))
        np.testing.assert_array_equal(vs,scores[name+'_val']);val_checks+=1
        vm=metrics(va,vs);assert vm==original[name]['val']
        target=c if query is None else query
        auc,rows=pair_out_auc(target,c,east['shot'],east['y'])
        results[name]=dict(val=vm,east_pair_auc=auc,
            by_positive_shot={str(s):float(np.mean([r['auc'] for r in rows if r['held_shots'][0]==s]))
                              for s in np.unique(east['shot'][east['y']==1])})
        details[name]=rows
        print(dict(name=name,pair_auc=auc,source_macro=vm['conditional_macro_auc']),flush=True)
    def deny(*args,**kwargs):raise AssertionError('recheck uses archived inputs; no new labels')
    with patch.object(io23,'read_outcome',side_effect=deny):
        evaluate('raw',norm(tr['x']),norm(va['x']),norm(east['x']))
        evaluate('raw_mean',norm(mean_features(tr['x'])),norm(mean_features(va['x'])),norm(mean_features(east['x'])))
        for seed in p['seeds']:
            for mode in MODES:
                name=f'{mode}_{seed}';model=Encoder().cuda();model.load_state_dict(heads[name]);model.eval()
                view=mean_features if mode=='mean' else lambda x:x
                a,b,c=[encode(model,norm(view(s['x']))) for s in (tr,va,east)]
                evaluate(name,a,b,c)
                if mode=='risk':
                    b=encode(model,norm(reverse_features(va['x'])))
                    target=encode(model,norm(reverse_features(east['x'])))
                    evaluate(f'reverse_{seed}',a,b,c,target)
        # Reproduce one complete SOURCE training run, not only inference.
        pairs=read('pairs_risk.npz')
        replay,trace=fit(norm(tr['x']),pairs,131,p['steps'])
        for name,value in replay.state_dict().items():
            torch.testing.assert_close(value.cpu(),heads['risk_131'][name],rtol=0,atol=0)
        assert trace==READ(root/'training.json')['risk_131']
    aggregate={mode:dict(val_conditional_auc=float(np.mean([results[f'{mode}_{s}']['val']['conditional_macro_auc'] for s in p['seeds']])),
        east_auc=float(np.mean([results[f'{mode}_{s}']['east_pair_auc'] for s in p['seeds']]))) for mode in (*MODES,'reverse')}
    accepted=qualifies(aggregate,results['raw']['east_pair_auc'],[results[f'risk_{s}']['east_pair_auc'] for s in p['seeds']])
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(output/'results.json',results);_durable_json(output/'paired_scores.json',details)
    report=dict(aggregate=aggregate,eligible_for_online_trial=accepted,candidate_promoted=False,
        scientific_goal_achieved=False,raw_auc=results['raw']['east_pair_auc'],
        source_val_replays=val_checks,source_training_replay='risk131 all1024steps EXACT',
        source_risk_unchanged=True,shot_pairs=64,independent_disruptive_shots=4,
        artifact_sha256={str(f.relative_to(output)):sha(f) for f in output.iterdir() if f.is_file()})
    _durable_json(output/'completed.json',report)
    _durable_json(root/'decision.json',dict(supersedes='completed.json eligibility only; original files retained',
        original_completed_sha256=sha(root/'completed.json'),recheck_completed_sha256=sha(output/'completed.json'),
        eligible_for_online_trial=accepted,candidate_promoted=False,scientific_goal_achieved=False,
        reason='evaluation repaired to same-history pairs and absolute chance/seed floor'))
    print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
