"""Verify the added oracle mask, reused safety curves and disruptive replays."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .. import io23
from ..contextual_feedback_20261023.run import read_signal
from ..continual_episode_20261023.run import labeled
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..diverse_support_20261023.run import READ,make_parent
from ..heldout_threshold_20261023.run import REFERENCE,models_for_calibration
from ..history_matched_rule_20261023.probe import PREVIOUS
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from .audit import correct
from .learning import predict
from .selective import teaching_mask,PLAN


def expected_updates(record,parent_record):
    if record['shot']!=parent_record['shot']:
        raise ValueError('matching parent shot required for replay receipts')
    # Safe trajectories are exactly reused, not recomputed by the selective
    # run. Verify against the originating receipt instead of omitting checks.
    return parent_record['updates'] if record['reused_safe_parent'] else record['updates']


def main(parent,root):
    torch.set_num_threads(2);torch.manual_seed(131)
    p=READ(root/'protocol.json');done=READ(root/'completed.json');old=READ(parent/'per_shot.json')
    assert p['parent_completed_sha256']==sha(parent/'completed.json')
    for f,h in done['artifact_sha256'].items():assert sha(root/f)==h,f
    for f,h in p['code_sha256'].items():assert sha(io23.ROOT/f)==h,f
    assert sha(PLAN)==p['plan_sha256']
    stored=READ(root/'per_shot.json');receipts=READ(root/'updates.json')
    parent_receipts=READ(parent/'updates.json')
    outcomes={r['shot']:r for r in READ(PREVIOUS/'outcomes.json')}
    differences={a:dict(good=[],bad=[],over3000=0,new_steps=0) for a in ('oracle1','oracle4')}
    assert stored['static']==old['static']
    reused=0
    for i,sid in enumerate(p['shot_ids']):
        row=receipts[i];assert row['shot']==sid;o=outcomes[sid]
        with np.load(root/'shots'/f'{sid}.npz') as z, np.load(parent/'shots'/f'{sid}.npz') as previous:
            np.testing.assert_array_equal(z['times'],previous['times'])
            if not o['is_disrupt']:
                assert row['reused_safe_parent'] and row['parent_curve_sha256']==sha(parent/'shots'/f'{sid}.npz')
                for a in differences:np.testing.assert_array_equal(z[a],previous[a])
                reused+=1
            else:
                assert not row['reused_safe_parent']
                y,valid=io23.labels_at(z['times'],True,o['down'],125.);mask=teaching_mask(y,valid,True)
                for a in differences:
                    for r in row['updates'][a]:
                        count=r['first_affected_prediction'];ends=np.arange(count-113,count,16);ends=ends[mask[ends]]
                        assert r['supervised_ends']==ends.tolist()
                        assert r['gradient_steps']==(0 if not len(ends) else (1 if a=='oracle1' else 4))
                        differences[a]['new_steps']+=r['gradient_steps']
            for a in differences:
                before=stored['static']['predictions'][i];after=stored[a]['predictions'][i]
                differences[a]['over3000']+=int(after['timely'] and after['lead_ms']>3000)
                if correct(before)!=correct(after):
                    differences[a]['good' if correct(after) else 'bad'].append(dict(shot=sid,before=before,after=after))
    base,_=make_parent();model=models_for_calibration(base)['mapped_fixed'];snapshot=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');replayed=[]
    # Cases with early suppressions, late rescue, and final stream position.
    for sid in (64581,87537,87861,96375):
        if sid not in p['shot_ids']:continue
        s=labeled(read_signal(sid),io23.read_outcome('EAST',sid),mapping)
        curves,rs=predict(model,{k:s[k] for k in ('values','available')},s['y'],teaching_mask(s['y'],s['valid'],s['is_disrupt']))
        with np.load(root/'shots'/f'{sid}.npz') as z:
            for a in curves:np.testing.assert_array_equal(curves[a],z[a])
        index=p['shot_ids'].index(sid)
        assert rs==expected_updates(receipts[index],parent_receipts[index]);replayed.append(sid)
    assert_unchanged(model,snapshot);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(root/'verification.json',dict(status='VERIFIED_NONDEPLOYABLE_SELECTIVE_ORACLE',
        completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        reused_safe_curves_verified=reused,replayed_shots=replayed,changes=differences))
    print({a:{k:len(v) if isinstance(v,list) else v for k,v in r.items()} for a,r in differences.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();main(args.parent,args.output)
