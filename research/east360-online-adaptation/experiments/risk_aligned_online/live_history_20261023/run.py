"""Full TEST360 comparison of current tangents and finite-history checks."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..fixed_test_protocol import load_test_contract, score_fixed_test
from ..label_cone_20261023.run import load_geometry
from ..simple_online import _durable_json
from ..simple_train import sha
from ..task_geometry_20261023.run import READ, load_model
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .learning import ARMS, predict, HISTORY_TOL, FRACTIONS

REFERENCE=io23.ROOT/'experiments/risk_aligned_online/runs/label_cone20_test360_20261023'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/当前梯度与实际历史保护_事前约定_20261023.md'
OLD={'static':'static','forecast':'forecast','symmetric':'symmetric','fixed':'signed'}
NAMES=tuple(OLD)+ARMS


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    rp,rv=READ(REFERENCE/'protocol.json'),READ(REFERENCE/'verification.json')
    assert rv['status']=='VERIFIED' and sha(REFERENCE/'completed.json')==rv['completed_sha256']
    for f,digest in READ(REFERENCE/'completed.json')['artifact_hashes'].items():assert sha(REFERENCE/f)==digest,f
    for f,digest in rp['code_sha256'].items():assert sha(io23.ROOT/f)==digest,f
    for f,digest in rp['reference_hashes'].items():assert sha(Path(rp['reference'])/f)==digest,f
    contract=load_test_contract('EAST');assert contract==rp['contract']
    assert len(rp['selected_ids'])==20 and not set(rp['selected_ids'])&set(contract['shot_ids'])
    model,threshold=load_model();original=state_cpu(model);assert threshold==rp['threshold']
    _,anchors,_=load_geometry();assert int(anchors['labels'].sum())==4
    history=Path(rp['reference']);prepared=Path(READ(history/'protocol.json')['reference'])
    mapping=READ(prepared/'input_map.json')
    root.mkdir(parents=True);(root/'shots').mkdir()
    paths={io23.ROOT/f for f in rp['code_sha256']};paths.update(Path(__file__).parent.glob('*.py'))
    paths.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_live_history.py')
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(paths):archive.write(f,str(f.relative_to(io23.ROOT)))
    torch.save({k:v.cpu() for k,v in anchors.items()},root/'anchors.pt')
    p=dict(reference=str(REFERENCE),history_reference=str(history),prepared_reference=str(prepared),
        parent=rp['parent'],parent_sha256=rp['parent_sha256'],source_checkpoint=rp['source_checkpoint'],
        source_sha256=rp['source_sha256'],threshold=threshold,contract=contract,selected_ids=rp['selected_ids'],
        plan_path=str(PLAN),plan_sha256=sha(PLAN),old_curve_mapping=OLD,
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in paths},
        reference_hashes={f:sha(REFERENCE/f) for f in ('protocol.json','completed.json','verification.json','prediction_receipts.json')},
        input_map_sha256=sha(prepared/'input_map.json'),anchors_sha256=sha(root/'anchors.pt'),
        new_arms=list(ARMS),first_update=256,interval=128,lr=.001,clip_raw_gradient_norm=1.,
        history_tolerance=HISTORY_TOL,backtrack_fractions=FRACTIONS,
        historical_jacobians='each current fast state, batched VJP of20 outputs',
        guard='all20 BCE <= previous+tol AND <= immutable parent+tol; finite6 trials or reject',
        norm_control='raw gradient length matched to guarded accepted step',
        no_inter=True,no_new_label_identities=True,no_alarm_change=True,independent_confirmation=False,
        label_gate='authorized historical20 labels; TEST outcomes prohibited before all seals',
        torch_version=torch.__version__,gpu=torch.cuda.get_device_name())
    _durable_json(root/'protocol.json',p)
    print(json.dumps(dict(stage='ready',seconds=time.monotonic()-start)),flush=True)
    refseals={r['shot']:r for r in READ(REFERENCE/'prediction_receipts.json')};seals=[];drifts={}
    with patch.object(io23,'read_outcome',side_effect=AssertionError('TEST label before sealed curves')):
        for index,sid in enumerate(contract['shot_ids']):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=transform({k:s[k] for k in ('values','available')},mapping)
            curves,updates,drift=predict(model,signal,anchors);drifts[str(sid)]=drift
            refpath=REFERENCE/'shots'/f'{sid}.npz';assert sha(refpath)==refseals[sid]['prediction_sha256']
            with np.load(refpath,allow_pickle=False) as old:
                np.testing.assert_array_equal(s['times'],old['times'])
                curves.update({a:old[b].copy() for a,b in OLD.items()})
            for a in ARMS:np.testing.assert_array_equal(curves[a][:256],curves['static'][:256])
            path=root/'shots'/f'{sid}.npz';up=root/'shots'/f'{sid}_updates.json'
            np.savez_compressed(path,times=s['times'],**curves);_durable_json(up,updates)
            seals.append(dict(shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up),reference_sha256=sha(refpath)))
            if (index+1)%30==0:
                progress=dict(shots=index+1,seconds=time.monotonic()-start)
                _durable_json(root/'progress.json',progress);print(json.dumps(progress),flush=True)
        _durable_json(root/'prediction_receipts.json',seals)
        _durable_json(root/'historical_drift.json',drifts)
        _durable_json(root/'pre_eval_hdf5_label_access.json',[])
    rows={a:[] for a in NAMES}
    for sid in contract['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
            for a in NAMES:
                rows[a].append(dict(shot=sid,times=s['times'][s['valid']],scores=z[a][s['valid']],is_disrupt=s['is_disrupt'],down=s['down']))
    full={a:score_fixed_test('EAST',r,threshold) for a,r in rows.items()};_durable_json(root/'per_shot.json',full)
    assert_unchanged(model,original)
    assert sha(Path(p['source_checkpoint']))==p['source_sha256'] and sha(Path(p['parent']))==p['parent_sha256']
    metrics={a:dict(**public(r),very_early_over3000=sum(q['timely'] and q['lead_ms']>3000 for q in r['predictions'])) for a,r in full.items()}
    passes={a:bool(metrics[a]['fpr']<=.2 and metrics[a]['tp']>=metrics['static']['tp'] and metrics[a]['J']>metrics['static']['J']) for a in ARMS}
    done=dict(status='COMPLETED',metrics=metrics,passes_descriptive_screen=passes,seconds=time.monotonic()-start,
        parent_unchanged=True,temporary_state_discarded=True,independent_confirmation=False,
        artifact_hashes={f:sha(root/f) for f in ('protocol.json','anchors.pt','per_shot.json','prediction_receipts.json',
            'historical_drift.json','pre_eval_hdf5_label_access.json')})
    _durable_json(root/'completed.json',done);print(json.dumps(done),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
