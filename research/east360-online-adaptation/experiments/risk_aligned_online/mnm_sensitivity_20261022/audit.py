"""Rescore all sealed MNM predictions and replay selected complete traces."""
import argparse
import json

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..fixed_test_protocol import load_test_contract, score_fixed_test
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import predict_many
from .run import restore_mnm


def main():
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('directory',type=Path)
    root=parser.parse_args().directory
    protocol=json.loads((root/'protocol.json').read_text())
    full=json.loads((root/'per_shot.json').read_text())
    mapping=json.loads((root/'density_map.json').read_text())
    seals=json.loads((root/'prediction_receipts.json').read_text())
    order=load_test_contract('EAST')['shot_ids']
    assert [r['shot'] for r in seals]==order and mapping['calibration_ids']==order[:20]
    for path,digest in protocol['code_sha256'].items(): assert sha(io23.ROOT/path)==digest
    assert sha(protocol['source_statistics_path'])==protocol['source_statistics_sha256']
    rows={k:[] for k in full}
    for index,(sid,seal) in enumerate(zip(order,seals)):
        path=root/'shots'/f'{sid}.npz'; assert sha(path)==seal['prediction_sha256']
        shot=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(path,allow_pickle=False) as trace:
            np.testing.assert_array_equal(trace['times'],shot['times'])
            for arm in rows:
                assert np.isfinite(trace[arm]).all()
                if index<20: np.testing.assert_array_equal(trace[arm],trace[arm.removesuffix('_density20')])
                v=shot['valid']
                rows[arm].append(dict(shot=sid,times=shot['times'][v],scores=trace[arm][v],
                                     is_disrupt=shot['is_disrupt'],down=shot['down']))
    for arm,records in rows.items():
        assert score_fixed_test('EAST',records,protocol['threshold'])==full[arm]
    flips={}
    for a,b in (('masked','restored'),('masked_density20','restored_density20')):
        flips[f'{a}_to_{b}']=[dict(before=x,after=y) for x,y in zip(full[a]['predictions'],full[b]['predictions'])
                             if (x['timely'],x['alarm'])!=(y['timely'],y['alarm'])]
    torch.set_num_threads(2)
    base,_=load_parent(DEFAULT_PARENT)
    model=PairedFeedback(base,np.ones(23)).cuda(); model.set_stage('frozen')
    checks=[]
    for index in (0,19,20,179,359):
        sid=order[index]
        shot=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        raw=io23.read_signals('EAST',sid,rate_contract='nominal')
        signal={k:shot[k] for k in ('values','available')}
        restored=restore_mnm(signal,raw['v'],raw['a'])
        inputs=dict(masked=signal,restored=restored,
            masked_density20=transform(signal,mapping) if index>=20 else signal,
            restored_density20=transform(restored,mapping) if index>=20 else restored)
        pred=predict_many(model,inputs,device='cuda')
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
            for arm in pred: np.testing.assert_array_equal(pred[arm],trace[arm])
        checks.append(sid)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    report=dict(status='PASS',rescored_shot_records=len(order)*len(rows),
        replayed_full_shots=checks,replayed_full_traces=len(checks)*len(rows),max_replay_difference=0,
        paired_flips=flips,source_archive_unchanged=True,
        very_early_over3000ms={k:sum(p['timely'] and p['lead_ms']>3000 for p in v['predictions'])
                             for k,v in full.items()})
    _durable_json(root/'audit.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
