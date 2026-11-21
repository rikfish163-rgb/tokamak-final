"""Post-training J-TEXT TEST220 retention check, never used for selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import simple_data
from ..fixed_test_protocol import load_test_contract, score_fixed_test
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.run import predict_many, public
from .run import ARMS


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('directory',type=Path)
    root=p.parse_args().directory
    if (root/'source_retention.json').exists(): raise FileExistsError('do not overwrite evidence')
    protocol=json.loads((root/'protocol.json').read_text())
    receipts=json.loads((root/'preparation_receipts.json').read_text())
    torch.set_num_threads(2); models={}
    for arm in ('original',*ARMS):
        base,_=load_parent(DEFAULT_PARENT)
        model=PairedFeedback(base,np.ones(23)).cuda()
        if arm!='original':
            assert sha(root/f'{arm}.pt')==receipts[arm]['checkpoint_sha256']
            model.load_state_dict(torch.load(root/f'{arm}.pt',map_location='cuda',weights_only=True)['model'])
        model.set_stage('frozen'); models[arm]=model
    rows={k:[] for k in models}; order=load_test_contract('J-TEXT')['shot_ids']
    advanced=simple_data.advanced_times()
    for sid in order:
        s=simple_data.read_shot('J-TEXT',sid)
        signal={k:s[k] for k in ('values','available')}
        scores={k:predict_many(m,dict(source=signal),device='cuda')['source'] for k,m in models.items()}
        np.testing.assert_array_equal(scores['original'],scores['density'])
        s=simple_data.attach_outcome(s,'J-TEXT',advanced)
        for arm in rows:
            v=s['valid']; rows[arm].append(dict(shot=sid,times=s['times'][v],scores=scores[arm][v],
                is_disrupt=s['is_disrupt'],down=s['down']))
    full={k:score_fixed_test('J-TEXT',v,protocol['threshold']) for k,v in rows.items()}
    assert full['original']['tp']==100 and full['original']['fp']==13
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(scope='post-training TEST220 descriptive retention, not checkpoint/threshold selection',
        results=full,parent_file_unchanged=True,density_predictions_bitwise_identical_to_parent=True,
        code_sha256=sha(__file__))
    _durable_json(root/'source_retention.json',result)
    print(json.dumps({k:public(v) for k,v in full.items()}),flush=True)


if __name__=='__main__': main()
