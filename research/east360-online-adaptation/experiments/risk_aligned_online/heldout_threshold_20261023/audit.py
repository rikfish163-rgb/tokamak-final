"""Independent recount and CUDA replay of calibration and all changed shots."""
import argparse
from pathlib import Path
import math
from unittest.mock import patch

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..diverse_support_20261023.run import READ, make_parent
from ..increment_forecast_20261023.learning import predict
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .run import REFERENCE, ARM_MAP, MODELS, models_for_calibration, model_curves


def main(root):
    if (root/'verification.json').exists(): raise FileExistsError(root/'verification.json')
    done=READ(root/'completed.json'); p=READ(root/'protocol.json'); expected=READ(root/'per_shot.json')
    for name,digest in done['artifact_sha256'].items(): assert sha(root/name)==digest,name
    for name,digest in p['code_sha256'].items(): assert sha(io23.ROOT/name)==digest,name
    for name,digest in p['input_sha256'].items(): assert sha(name)==digest,name
    thresholds=READ(root/'thresholds.json'); cal=READ(root/'calibration_records.json')
    assert sha(root/'thresholds.json')==READ(root/'threshold_seal.json')['sha256']
    for r in cal:
        assert io23.read_outcome('EAST',r['shot'])==(r['is_disrupt'],r['down'])
        with np.load(root/'calibration'/f'{r["shot"]}.npz') as z:
            assert r['peaks']=={name:float(z[name].max()) for name in MODELS}
    for model in MODELS:
        peaks=sorted(r['peaks'][model] for r in cal if not r['is_disrupt']); n=len(peaks)
        for a,rule in thresholds[model].items():
            # Count acceptable ranks without reusing the calibration routine.
            eligible=[k for k in range(1,n+1) if (n+1-k)/(n+1)<=float(a)]
            if not eligible: assert rule['threshold'] is None; continue
            k=min(eligible); assert k==rule['rank'] and rule['order_statistic']==peaks[k-1]
            assert rule['threshold']==float(np.nextafter(np.float32(peaks[k-1]),np.float32(np.inf)))
    checks=0; counts={k:dict(tp=0,fp=0) for k,v in expected.items() if 'predictions' in v}
    for sid in p['test_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(REFERENCE/'fold0/shots'/f'{sid}.npz') as z:
            for key in counts:
                arm,level=key.split('/'); threshold=expected[key]['threshold']
                hit=np.flatnonzero(s['valid']&(z[arm]>=threshold)); alarm=bool(len(hit))
                lead=float(round((s['down']-s['times'][hit[0]])*1000,9)) if s['is_disrupt'] and alarm else None
                row=dict(shot=sid,is_disrupt=s['is_disrupt'],alarm=alarm,lead_ms=lead,timely=lead is not None and lead>30.)
                assert row==next(v for v in expected[key]['predictions'] if v['shot']==sid)
                counts[key]['tp']+=int(row['timely']);counts[key]['fp']+=int(alarm and not s['is_disrupt']);checks+=1
    for k,c in counts.items(): assert all(expected[k][name]==v for name,v in c.items())
    flips={}
    for model in ('mapped_fixed','mapped_head'):
        before={v['shot']:v for v in expected[model+'_static/0.2']['predictions']}
        after={v['shot']:v for v in expected[model+'_intra/0.2']['predictions']}; changes=[]
        for sid,o in before.items():
            n=after[sid]; good=lambda v:v['timely'] if v['is_disrupt'] else not v['alarm']
            if good(o)!=good(n): changes.append(dict(shot=sid,beneficial=good(n),before=o,after=n))
        flips[model]=changes
    torch.set_num_threads(2);torch.manual_seed(131);parent,_=make_parent();models=models_for_calibration(parent)
    mapping=READ(REFERENCE/'input_map.json');replays=[]
    def deny(*args,**kwargs): raise AssertionError('no outcomes during replay')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in (89272,89584):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            curves=model_curves(models,s,mapping)
            with np.load(root/'calibration'/f'{sid}.npz') as z:
                for name,v in curves.items():np.testing.assert_array_equal(v,z[name])
            replays.append(dict(scope='calibration_all_four_models',shot=sid))
        for model,changes in flips.items():
            for r in changes:
                sid=r['shot'];s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal=transform({k:s[k] for k in ('values','available')},mapping)
                curves,updates,_=predict(models[model],signal)
                with np.load(REFERENCE/'fold0/shots'/f'{sid}.npz') as z:
                    for arm,v in curves.items():np.testing.assert_array_equal(v,z[model+'_'+arm])
                assert updates==READ(REFERENCE/'fold0/shots'/f'{sid}_updates.json')[model]
                replays.append(dict(scope='posthoc_changed_shot_complete_static_and_intra',model=model,shot=sid))
    result=dict(status='VERIFIED_COUNTS_CALIBRATION_AND_CHANGED_SHOT_REPLAY',completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),shot_arm_checks=checks,flips=flips,replays=replays,
        source_reproduction=done['source_exact_reproduction'],source_archive_sha256=p['source_archive_sha256'],
        note='Audit script added after run snapshot. Changed-shot replays selected post hoc; not independent validation.')
    _durable_json(root/'verification.json',result);print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
