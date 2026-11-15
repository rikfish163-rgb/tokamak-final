"""Separate held-out shot calibration from frozen neural intra comparisons."""
import argparse
import copy
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..diverse_support_20261023.run import make_parent, READ
from ..labeled_alignment_20261023.run import SPLIT
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha, metrics
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import predict_many, public
from .calibration import safe_threshold

REFERENCE = io23.ROOT/'experiments/risk_aligned_online/runs/mapped_head_factor20_test360_20261023'
REFERENCE_SHA = '85aed7fdd057df541ab91ff74fc7accd1ccea50a98f77aca1f9af0fef57b943e'
PLAN = io23.ROOT/'docs/research/source_event_rebuild_20261022/独立校准10炮_事前约定_20261023.md'
MODELS = ('source', 'mapped_source', 'mapped_fixed', 'mapped_head')
ARM_MAP = dict(source=('source',), mapped_source=('mapped_source_static',),
               mapped_fixed=('mapped_fixed_static','mapped_fixed_intra'),
               mapped_head=('mapped_head_static','mapped_head_intra'))
CALIBRATION_IDS = [89272,89284,89449,89584,89976,91285,91775,92011,92315,92601]


def models_for_calibration(parent):
    models = dict(source=parent, mapped_source=parent)
    for name in ('mapped_fixed', 'mapped_head'):
        model = copy.deepcopy(parent)
        model.load_state_dict(torch.load(REFERENCE/'fold0'/name/'prepared.pt', map_location='cuda', weights_only=True)['model'])
        model.set_stage('frozen'); models[name] = model
    return models


def model_curves(models, shot, mapping):
    raw = {k:shot[k] for k in ('values','available')}
    mapped = transform(raw, mapping)
    return {name:predict_many(model, {'static':raw if name=='source' else mapped}, device='cuda')['static']
            for name,model in models.items()}


def main(root):
    if root.exists(): raise FileExistsError(root)
    if not torch.cuda.is_available(): raise RuntimeError('rl_env CUDA required')
    assert sha(REFERENCE/'completed.json') == REFERENCE_SHA
    ref = READ(REFERENCE/'completed.json'); rp = READ(REFERENCE/'protocol.json')
    for name,digest in ref['artifact_sha256'].items(): assert sha(REFERENCE/name)==digest,name
    val = READ(SPLIT)['val']; ids = sorted(val['disruption_shots']+val['non_disruption_shots'])[:10]
    assert ids == CALIBRATION_IDS and not set(ids)&(set(rp['selected'])|set(rp['shot_ids']))
    torch.set_num_threads(2); torch.manual_seed(131); start=time.monotonic()
    parent, source_threshold = make_parent(); models=models_for_calibration(parent)
    states={k:state_cpu(v) for k,v in models.items()}; mapping=READ(REFERENCE/'input_map.json')
    root.mkdir(parents=True); (root/'calibration').mkdir(); (root/'source_test').mkdir()
    source_ids=io23.source_roles()['test']
    files=set(Path(__file__).parent.glob('*.py')); files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_heldout_threshold.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None): files.add(Path(module.__file__))
    p=dict(calibration_ids=ids, preparation_ids=rp['selected'], test_ids=rp['shot_ids'], source_test_ids=source_ids,
        total_target_configuration_budget=30, reference_sha256=REFERENCE_SHA, source_sha256=PARENT_SHA256,
        source_threshold=source_threshold, alphas=[.1,.15,.2], plan_sha256=sha(PLAN), split_sha256=sha(SPLIT),
        input_sha256={str(io23.path_for(device,sid)):sha(io23.path_for(device,sid))
            for device,shots in (('EAST',ids),('J-TEXT',source_ids)) for sid in shots},
        code_sha256={str(v.relative_to(io23.ROOT)):sha(v) for v in files},
        source_archive_sha256=sha(DEFAULT_PARENT.parent/'test220.json'), neural_training_steps=0,
        independent_confirmation=False, method='safe whole-shot ceil((n+1)*(1-alpha)) order statistic; static/intra share threshold',
        warning='Existing development data; not chronological commissioning; exchangeability unverified; no FPR guarantee',
        gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    _durable_json(root/'protocol.json',p)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files): z.write(path,str(path.relative_to(io23.ROOT)))
    def deny(*args,**kwargs): raise AssertionError('prediction may not read outcomes')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in ids:
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            curves=model_curves(models,s,mapping)
            np.savez_compressed(root/'calibration'/f'{sid}.npz',times=s['times'],**curves)
            print(dict(stage='calibration_predictions',shot=sid),flush=True)
        for n,sid in enumerate(source_ids,1):
            s=simple_data.read_shot('J-TEXT',sid)
            scores=predict_many(parent,{'source':{k:s[k] for k in ('values','available')}},device='cuda')['source']
            np.savez_compressed(root/'source_test'/f'{sid}.npz',times=s['times'],scores=scores)
            if n%25==0: print(dict(stage='source_predictions',shots=n,seconds=time.monotonic()-start),flush=True)
    _durable_json(root/'prediction_seals.json',{str(f.relative_to(root)):sha(f) for sub in ('calibration','source_test') for f in (root/sub).glob('*.npz')})
    calibration=[]
    for sid in ids:
        flag,down=io23.read_outcome('EAST',sid)
        with np.load(root/'calibration'/f'{sid}.npz') as z:
            calibration.append(dict(shot=sid,is_disrupt=flag,down=down,peaks={name:float(z[name].max()) for name in MODELS}))
    _durable_json(root/'calibration_records.json',calibration)
    thresholds={name:{str(alpha):safe_threshold([r['peaks'][name] for r in calibration if not r['is_disrupt']],alpha)
        for alpha in p['alphas']} for name in MODELS}
    _durable_json(root/'thresholds.json',thresholds)
    _durable_json(root/'threshold_seal.json',dict(sha256=sha(root/'thresholds.json'),test_outcomes_accessed_in_this_run=0))
    # Outcomes below are evaluation only. No network/threshold fitting follows.
    source_rows=[]
    for sid in source_ids:
        s=simple_data.attach_outcome(simple_data.read_shot('J-TEXT',sid),'J-TEXT',{})
        with np.load(root/'source_test'/f'{sid}.npz') as z:
            np.testing.assert_array_equal(z['times'],s['times'])
            source_rows.append(dict(shot=sid,times=s['times'][s['valid']],scores=z['scores'][s['valid']],is_disrupt=s['is_disrupt'],down=s['down']))
    source_result=metrics(source_rows,source_threshold,lead=10.)
    archived=READ(DEFAULT_PARENT.parent/'test220.json')['jtext']
    assert source_result==archived, 'current source prediction differs from archived source test'
    _durable_json(root/'source_reproduction.json',source_result)
    rows={a:[] for arms in ARM_MAP.values() for a in arms}
    for sid in rp['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(REFERENCE/'fold0/shots'/f'{sid}.npz') as z:
            np.testing.assert_array_equal(z['times'],s['times'])
            for arm in rows: rows[arm].append(dict(shot=sid,times=s['times'][s['valid']],scores=z[arm][s['valid']],is_disrupt=s['is_disrupt'],down=s['down']))
    results={}
    for model,arms in ARM_MAP.items():
        for alpha,rule in {'source_threshold':dict(threshold=source_threshold),**thresholds[model]}.items():
            for arm in arms:
                key=arm+'/'+alpha; threshold=rule['threshold']
                if threshold is None: results[key]=dict(status=rule['status']); continue
                m=metrics(rows[arm],threshold,lead=30.)
                m['very_early_over3000']=sum(v['timely'] and v['lead_ms']>3000 for v in m['predictions'])
                results[key]=m
    _durable_json(root/'per_shot.json',results)
    for name,model in models.items(): assert_unchanged(model,states[name])
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    done=dict(status='COMPLETED',metrics={k:public(v) for k,v in results.items()},thresholds=thresholds,
        calibration_disrupt=sum(r['is_disrupt'] for r in calibration),calibration_safe=sum(not r['is_disrupt'] for r in calibration),
        source_exact_reproduction=True,source_metrics=public(source_result),neural_training_steps=0,seconds=time.monotonic()-start,
        artifact_sha256={str(v.relative_to(root)):sha(v) for v in root.rglob('*') if v.is_file()})
    _durable_json(root/'completed.json',done); print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('root',type=Path); main(parser.parse_args().root)
