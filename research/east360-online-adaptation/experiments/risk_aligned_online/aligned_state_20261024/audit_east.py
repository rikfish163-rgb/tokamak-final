"""Independent whole-shot recount and deterministic CUDA replay after completion."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23
from ..simple_online import _durable_json
from ..simple_train import sha
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from .audit import independent_count
from .east import load_aligned,read_signal,transform,MAP
from .stream import ARMS,predict_stream


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();root=args.root
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    read=lambda f:json.loads((root/f).read_text())
    done=read('completed.json');protocol=read('protocol.json');start=time.monotonic()
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    thresholds=read('thresholds.json');cal=read('calibration_records.json')
    safe=sorted(r['peak'] for r in cal if not r['is_disrupt'])
    for alpha,item in thresholds.items():
        rank=math.ceil((len(safe)+1)*(1-Fraction(alpha)));assert rank==item['rank']
        if rank>len(safe):assert item['threshold'] is None
        else:assert item['threshold']==float(np.nextafter(np.float32(safe[rank-1]),np.float32(np.inf)))
    rows={a:[] for a in ARMS}
    for r in read('outcomes.json'):
        sid=r['shot'];flag,down=io23.read_outcome('EAST',sid)
        assert flag==r['is_disrupt'] and down==r['down']
        with np.load(root/'shots'/f'{sid}.npz') as z:
            # Independently form the preterminal interval, not the risk label.
            valid=(down-z['times'])*1000>=-1e-6 if flag else np.ones(len(z['times']),bool)
            for arm in ARMS:
                rows[arm].append(dict(shot=sid,is_disrupt=flag,down=down,times=z['times'][valid],scores=z[arm][valid]))
    counts={};comparisons={}
    for alpha,item in thresholds.items():
        if item['threshold'] is None:continue
        counts[alpha]={a:independent_count(r,item['threshold'],30.) for a,r in rows.items()}
        for arm,r in counts[alpha].items():
            for key in ('tp','fp','n_disrupt','n_safe','tpr','fpr','J'):assert r[key]==done['metrics'][alpha][arm][key]
        flags={r['shot']:r['is_disrupt'] for r in rows['off']}
        original={r['shot']:r for r in counts[alpha]['off']['decisions']}
        comparisons[alpha]={}
        for arm in ARMS[1:]:
            current={r['shot']:r for r in counts[alpha][arm]['decisions']}
            comparisons[alpha][arm]=dict(
                rescued=[sid for sid in original if flags[sid] and current[sid]['timely'] and not original[sid]['timely']],
                lost=[sid for sid in original if flags[sid] and original[sid]['timely'] and not current[sid]['timely']],
                fp_removed=[sid for sid in original if not flags[sid] and original[sid]['alarm'] and not current[sid]['alarm']],
                fp_added=[sid for sid in original if not flags[sid] and current[sid]['alarm'] and not original[sid]['alarm']])
    _durable_json(root/'paired_changes.json',comparisons)
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    model,_=load_aligned(Path(protocol['source']))
    prepared=torch.load(root/'prepared.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(prepared['model']);model.requires_grad_(False)
    mapping=json.loads(MAP.read_text());assert sha(MAP)==protocol['input_map_sha256']
    ids=protocol['shot_ids'];selected=[ids[k] for k in (0,89,179,269,359)];replays=[]
    def deny(*args,**kwargs):raise AssertionError('replay cannot read outcomes')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in selected:
            s=read_signal(sid);signal=transform({k:s[k] for k in ('values','available')},mapping)
            curves,trace=predict_stream(model,signal,interval=protocol['interval'],learning_rate=protocol['inner_rate'])
            old=read(f'shots/{sid}_updates.json');assert trace==old
            with np.load(root/'shots'/f'{sid}.npz') as z:
                for arm in ARMS:np.testing.assert_array_equal(curves[arm],z[arm])
            replays.append(dict(shot=sid,curve_points=len(curves['off']),update_records=len(trace)))
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    report=dict(status='METRIC_RECOUNT_AND_REPLAY_VERIFIED',completed_sha256=sha(root/'completed.json'),
        independent_whole_shot_decisions=len(ids)*len(ARMS)*sum(v['threshold'] is not None for v in thresholds.values()),
        replay=replays,source_unchanged=True,seconds=time.monotonic()-start,
        scope='numerical reproduction, causal API and manifest checks; not independent efficacy validation')
    _durable_json(root/'verification.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
