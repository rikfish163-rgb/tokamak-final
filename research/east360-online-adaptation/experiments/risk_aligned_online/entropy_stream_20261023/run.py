"""Complete TEST360 replay; no outcome access until all predictions are sealed."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..fixed_test_protocol import load_test_contract,score_fixed_test
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..task_geometry_20261023.run import REFERENCE,READ,load_model
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .learning import ARMS,predict

SCREEN=io23.ROOT/'experiments/risk_aligned_online/runs/learned_update_loss20_screen_20261023'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/熵反馈完整炮流_事前约定_20261023.md'
NAMES=('static','forecast')+ARMS


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    rp,rv=READ(REFERENCE/'protocol.json'),READ(REFERENCE/'verification.json')
    assert rv['status']=='VERIFIED' and sha(REFERENCE/'completed.json')==rv['completed_sha256']
    sv=READ(SCREEN/'verification.json')
    assert sv['status']=='VERIFIED' and sha(SCREEN/'completed.json')==sv['completed_sha256']
    for f,d in rp['code_sha256'].items():assert sha(io23.ROOT/f)==d,f
    parent=REFERENCE/'conditional_moments.pt'
    assert sha(parent)==READ(REFERENCE/'preparation_receipts.json')['conditional_moments']['checkpoint_sha256']
    contract=load_test_contract('EAST');assert contract==rp['contract']
    model,threshold=load_model();original=state_cpu(model);mapping=READ(REFERENCE/'input_map.json')
    root.mkdir(parents=True);(root/'shots').mkdir()
    paths={io23.ROOT/f for f in rp['code_sha256']};paths.update(Path(__file__).parent.glob('*.py'))
    for f in ('task_geometry_20261023/run.py','task_geometry_20261023/learning.py','tests/test_entropy_stream.py'):
        paths.add(io23.ROOT/'experiments/risk_aligned_online'/f)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):z.write(p,str(p.relative_to(io23.ROOT)))
    p=dict(reference=str(REFERENCE),source_checkpoint=str(DEFAULT_PARENT),source_sha256=PARENT_SHA256,
        parent=str(parent),parent_sha256=sha(parent),contract=contract,threshold=threshold,
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in paths},plan_path=str(PLAN),plan_sha256=sha(PLAN),
        reference_hashes={n:sha(REFERENCE/n) for n in ('protocol.json','completed.json','verification.json','input_map.json','prediction_receipts.json')},
        selection_screen=str(SCREEN),selection_screen_sha256=sha(SCREEN/'completed.json'),selected_ids=rp['selected_ids'],
        new_arms=list(ARMS),reused_curves=['conditional_moments_static','conditional_moments_intra'],
        first_update=256,interval=128,learning_rate=.001,gradient_clip=1.,parameter_scope=model.encoder_names(),
        selection='exploratory follow-up to passed historical entropy control; NOT unselected confirmation',
        no_alarm_change=True,no_inter=True,no_new_label_identities=True,
        causal_gate='all360 new traces sealed before any outcome access in this run',
        torch_version=torch.__version__,cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name())
    _durable_json(root/'protocol.json',p)
    reference_seals={r['shot']:r for r in READ(REFERENCE/'prediction_receipts.json')};seals=[]
    with patch.object(io23,'read_outcome',side_effect=AssertionError('outcome forbidden before all seals')):
        for i,sid in enumerate(contract['shot_ids']):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=transform({k:s[k] for k in ('values','available')},mapping)
            curves,updates=predict(model,signal)
            refpath=REFERENCE/'shots'/f'{sid}.npz';assert sha(refpath)==reference_seals[sid]['prediction_sha256']
            with np.load(refpath,allow_pickle=False) as old:
                np.testing.assert_array_equal(s['times'],old['times'])
                curves.update(static=old['conditional_moments_static'].copy(),forecast=old['conditional_moments_intra'].copy())
            for arm in ARMS:np.testing.assert_array_equal(curves[arm][:256],curves['static'][:256])
            np.testing.assert_array_equal(curves['local_reset'][:384],curves['cumulative'][:384])
            path=root/'shots'/f'{sid}.npz';up=root/'shots'/f'{sid}_updates.json'
            np.savez_compressed(path,times=s['times'],**curves);_durable_json(up,updates)
            seals.append(dict(shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up),reference_sha256=sha(refpath)))
            if (i+1)%30==0:
                progress=dict(shots=i+1,seconds=time.monotonic()-started)
                _durable_json(root/'progress.json',progress);print(json.dumps(progress),flush=True)
        _durable_json(root/'prediction_receipts.json',seals)
        _durable_json(root/'pre_eval_label_access.json',[])
    rows={a:[] for a in NAMES}
    for sid in contract['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
            for a in NAMES:
                rows[a].append(dict(shot=sid,times=s['times'][s['valid']],scores=z[a][s['valid']],
                                    is_disrupt=s['is_disrupt'],down=s['down']))
    results={a:score_fixed_test('EAST',r,threshold) for a,r in rows.items()}
    _durable_json(root/'per_shot.json',results)
    assert_unchanged(model,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256 and sha(parent)==p['parent_sha256']
    metrics={a:dict(**public(r),very_early_over3000=sum(q['timely'] and q['lead_ms']>3000 for q in r['predictions'])) for a,r in results.items()}
    passes={a:bool(metrics[a]['fpr']<=.20 and metrics[a]['tp']>=metrics['static']['tp'] and metrics[a]['J']>metrics['static']['J']) for a in ARMS}
    done=dict(status='COMPLETED',metrics=metrics,passes_descriptive_screen=passes,seconds=time.monotonic()-started,
        parent_unchanged=True,temporary_state_discarded=True,independent_confirmation=False,
        protocol_sha256=sha(root/'protocol.json'),per_shot_sha256=sha(root/'per_shot.json'),
        seals_sha256=sha(root/'prediction_receipts.json'))
    _durable_json(root/'completed.json',done);print(json.dumps(done),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
