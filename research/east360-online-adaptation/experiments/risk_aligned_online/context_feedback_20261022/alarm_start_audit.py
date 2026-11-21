"""Post-hoc alarm-only control; never counted as a neural-adaptation gain."""
import json

import numpy as np

from .. import io23
from ..fixed_test_protocol import load_test_contract,score_fixed_test
from ..event_adapt_20261022.train import public
from ..simple_online import _durable_json


def main():
    run=io23.ROOT/'experiments/risk_aligned_online/runs/context_feedback_east_uniform_20261022'
    config=json.loads((run/'protocol.json').read_text());threshold=config['threshold']
    beginnings=(0,4,8,16,32,64,96,128);rows={n:[] for n in beginnings};early={n:[] for n in beginnings}
    for sid in load_test_contract('EAST')['shot_ids']:
        with np.load(run/'shots'/f'{sid}.npz') as data:
            d,down=io23.read_outcome('EAST',sid)
            _,valid=io23.labels_at(data['times'],d,down,125.)
            alarm=np.flatnonzero(valid & (data['static']>=threshold))
            for n in beginnings:
                keep=valid & (np.arange(len(valid))>=n)
                rows[n].append(dict(shot=sid,is_disrupt=d,down=down,times=data['times'][keep],scores=data['static'][keep]))
                if not d and len(alarm) and alarm[0]<n: early[n].append(sid)
    result=dict(status='VERIFIED',scope='post-hoc diagnostic; all rows disclosed; no threshold fitting; not a deployable neural result',
        warning='alarm waiting changes evaluation policy and may lose timely alarms; not adopted as the main comparison',
        rows={n:dict(metrics=public(score_fixed_test('EAST',r,threshold)),
            irrevocable_early_fp=len(early[n]),late_update_fpr_floor=len(early[n])/180.) for n,r in rows.items()})
    output=io23.ROOT/'outputs/source_event_rebuild_audit_20261022/context_alarm_start_diagnostic.json'
    if output.exists(): raise FileExistsError(output)
    _durable_json(output,result);print(json.dumps(result))


if __name__=='__main__':main()
