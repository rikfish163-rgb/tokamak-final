"""POSTHOC curve sensitivity, with exact neural replay of the largest case.

No policy or parameters change. First false-alarm TIME is retained here because
the operational scalar scorer records safe-shot alarm presence, not its time.
"""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..credit_horizon_20261023.run import READ
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..external65_20261023.run import models_for, MODELS
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import predict


def compare(times, early, late, threshold, *, disruptive, down):
    times,early,late = map(np.asarray,(times,early,late))
    if (times.ndim!=1 or early.shape!=times.shape or late.shape!=times.shape
            or not len(times) or not all(np.isfinite(v).all() for v in (times,early,late))
            or (disruptive and (down is None or not np.isfinite(down)))):
        raise ValueError('finite matching curves, times, and valid outcome required')
    valid=np.ones(len(times),bool) if not disruptive else (down-times)*1000>=-1e-6
    if not valid.any(): raise ValueError('no valid observation')
    index=int(np.argmax(np.where(valid,np.abs(early-late),-1)))
    result=dict(valid_points=int(valid.sum()),max_difference=float(abs(early[index]-late[index])),
        max_difference_index=index,max_difference_time=float(times[index]),
        early_at_max=float(early[index]),late_at_max=float(late[index]),
        threshold_disagreements=int(np.sum(valid&((early>=threshold)!=(late>=threshold)))))
    for name,curve in (('early',early),('late',late)):
        ids=np.flatnonzero(valid&(curve>=threshold));first=int(ids[0]) if len(ids) else None
        result[f'{name}_first_index']=first
        result[f'{name}_first_time']=float(times[first]) if first is not None else None
    return result


def main(root):
    out=root/'posthoc_trajectory.json'
    if out.exists(): raise FileExistsError(out)
    done=READ(root/'completed.json');p=READ(root/'protocol.json');audit=READ(root/'verification.json')
    assert sha(root/'completed.json')==audit['completed_sha256']
    roster={r['shot']:r for r in p['roster']};rows=[]
    for f in range(4):
        for sid in p['shot_ids']:
            path=root/f'fold{f}'/'shots'/f'{sid}.npz'
            assert sha(path)==done['artifact_sha256'][str(path.relative_to(root))]
            r=roster[sid]
            with np.load(path) as z:
                for m in MODELS:
                    rows.append(dict(fold=f,model=m,shot=sid,is_disrupt=r['is_disrupt'],
                        **compare(z['times'],z[f'{m}_early1'],z[f'{m}_late1'],p['threshold'],
                                  disruptive=r['is_disrupt'],down=r['down'])))
    selected=max(rows,key=lambda r:r['max_difference'])
    f,sid,m=selected['fold'],selected['shot'],selected['model']
    torch.set_num_threads(2);torch.manual_seed(131)
    model=models_for(f)[m];frozen=state_cpu(model)
    def deny(*args,**kwargs): raise AssertionError('no outcome in posthoc neural replay')
    with patch.object(io23,'read_outcome',side_effect=deny):
        s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        signal=transform({k:s[k] for k in ('values','available')},READ(root/f'fold{f}'/'input_map.json'))
        scores,receipts=predict(model,signal)
    with np.load(root/f'fold{f}'/'shots'/f'{sid}.npz') as z:
        for a,v in scores.items(): np.testing.assert_array_equal(v,z[f'{m}_{a}'])
    assert receipts==READ(root/f'fold{f}'/'shots'/f'{sid}_updates.json')[m]
    assert_unchanged(model,frozen)
    summaries={}
    for name in MODELS:
        subset=[r for r in rows if r['model']==name]
        summaries[name]=dict(median_record_max_difference=float(np.median([r['max_difference'] for r in subset])),
            maximum_difference=max(r['max_difference'] for r in subset),
            threshold_disagreements=sum(r['threshold_disagreements'] for r in subset),
            first_alarm_time_changed=sum(r['early_first_index']!=r['late_first_index'] for r in subset))
    result=dict(status='POSTHOC_DIAGNOSTIC',completed_sha256=sha(root/'completed.json'),
        code_sha256=sha(Path(__file__)),summary=summaries,rows=rows,
        selected_after_results=selected,selected_neural_replay_exact=True,
        predictions_modified=False,policy_changed=False,
        limitation='Point counts are correlated within shots and across folds, not independent trials; no physical mechanism claim.')
    _durable_json(out,result);print(dict(summary=summaries,selected=selected,neural_replay_exact=True),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
