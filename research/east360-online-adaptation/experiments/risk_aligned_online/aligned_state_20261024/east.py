"""Paired full EAST360: fixed target start, actual/constant/reversed within-shot updates."""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..fixed_test_protocol import load_test_contract, score_fixed_test
from ..heldout_threshold_20261023.calibration import safe_threshold
from ..internal_fast_window_20261023.learning import training_pools, sample_batch
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .model import AlignedState, observed_loss
from .stream import ARMS, predict_stream, predict_static
from .train import public

ROOT = io23.ROOT/'experiments/risk_aligned_online'
REFERENCE = ROOT/'runs/contextual_feedback360_20261023'
MAP = ROOT/'runs/mapped_head_factor20_test360_20261023/input_map.json'
PLAN = io23.ROOT/'docs/research/source_event_rebuild_20261023/对齐编码器_EAST炮内360_事前约定_20261024.md'
READ = lambda p:json.loads(p.read_text())


def load_aligned(path):
    done = READ(path.parent/'completed.json')
    if not done['source_qualified'] or sha(path) != done['source_sha256']:
        raise ValueError('qualified hash-matched source interface required')
    ck = torch.load(path,map_location='cpu',weights_only=True)
    base,_ = load_parent(DEFAULT_PARENT)
    if ck['config']['format'] != 'aligned-state-v1': raise ValueError('wrong source lineage')
    net = AlignedState(base,ck['model']['level_scale'])
    net.load_state_dict(ck['model'],strict=True); net.cuda().eval()
    return net,ck


def read_signal(sid):
    return apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))


def labeled(s,outcome,mapping):
    flag,down = outcome; y,valid = io23.labels_at(s['times'],flag,down,125.)
    return dict(transform({k:s[k] for k in ('values','available')},mapping),
                shot=s['shot'],times=s['times'],is_disrupt=flag,down=down,y=y,valid=valid)


def calibration_viability(records,threshold):
    """Reject an obvious target-start collapse, not certify generalization.

    The first paired run exposed this missing check: both ended calibration
    disruptions were below the safe-calibrated threshold. Do not change the
    threshold using TEST outcomes or spend another full stream on that start.
    One detected historical positive is necessary here, not sufficient for
    model qualification, a FPR guarantee, or promotion.
    """
    positives=[r for r in records if r['is_disrupt']]
    detected=(sum(r['timely_peak'] is not None and r['timely_peak']>=threshold
                  for r in positives) if threshold is not None else 0)
    allow=threshold is not None and bool(positives) and detected>0
    return dict(allow_full_stream=allow,disruptions=len(positives),timely_detected=detected,
                scope='necessary target-start sanity check, NOT deployment qualification')


def prepare(model,records,steps=512):
    model.set_alignment(encoder=True); names = set(model.encoder_names())
    opt = torch.optim.AdamW([dict(params=[p for k,p in model.named_parameters() if k in names],lr=1e-5),
                            dict(params=list(model.bridge.parameters()),lr=1e-3)],weight_decay=0.)
    rng = np.random.default_rng(131); groups = training_pools(records); history=[]
    for step in range(steps):
        x,m,y,samples = sample_batch(records,groups,rng,'cuda')
        opt.zero_grad(set_to_none=True)
        risk = F.binary_cross_entropy_with_logits(model(x,m)['logit'],y)
        rec = observed_loss(model,x,m); loss = risk+.20*rec
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite target preparation')
        loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        opt.step(); history.append(dict(step=step,risk=float(risk.detach()),reconstruction=float(rec.detach()),samples=samples))
    model.requires_grad_(False)
    return history


def evaluate(root,ids,thresholds,reader):
    rows = {a:[] for a in ARMS}; outcomes=[]
    for sid in ids:
        flag,down=reader('EAST',sid); outcomes.append(dict(shot=sid,is_disrupt=bool(flag),down=down))
        with np.load(root/'shots'/f'{sid}.npz') as z:
            _,valid=io23.labels_at(z['times'],flag,down,125.)
            for a in ARMS:
                rows[a].append(dict(shot=sid,is_disrupt=flag,down=down,times=z['times'][valid],scores=z[a][valid]))
    _durable_json(root/'outcomes.json',outcomes)
    results={}
    for alpha,cal in thresholds.items():
        if cal['threshold'] is None:
            results[alpha]=dict(status=cal['status']); continue
        metrics={a:score_fixed_test('EAST',r,cal['threshold']) for a,r in rows.items()}
        for m in metrics.values(): m['very_early_over3000']=sum(r['timely'] and r['lead_ms']>3000 for r in m['predictions'])
        _durable_json(root/f'per_shot_{alpha}.json',metrics)
        results[alpha]={a:public(m) for a,m in metrics.items()}
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.output
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);torch.backends.mha.set_fastpath_enabled(False)
    start=time.monotonic();model,ck=load_aligned(args.source)
    old=READ(REFERENCE/'protocol.json');mapping=READ(MAP)
    ids=load_test_contract('EAST')['shot_ids'];prep=old['preparation_ids'];cal=old['calibration_ids']
    assert not set(ids)&(set(prep)|set(cal)) and not set(prep)&set(cal)
    assert mapping['calibration_ids']==prep
    root.mkdir(parents=True);(root/'shots').mkdir();(root/'calibration').mkdir()
    protected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()
               if k not in set(model.encoder_names()) and not k.startswith('bridge.')}
    paths=set(Path(__file__).parent.glob('*.py'))
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):paths.add(Path(module.__file__))
    protocol=dict(format='aligned-state-east360-v1',source=str(args.source.resolve()),source_sha256=sha(args.source),
        original_source_sha256=PARENT_SHA256,preparation_ids=prep,calibration_ids=cal,shot_ids=ids,
        input_map_sha256=sha(MAP),plan_sha256=sha(PLAN),reference_protocol_sha256=sha(REFERENCE/'protocol.json'),
        arms=ARMS,interval=32,first_update_count=128,inner_rate=1e-4,inner_steps=1,
        prepare_steps=512,prepare_encoder_lr=1e-5,prepare_bridge_lr=1e-3,
        train_positive='ended disruptive windows lead>30ms and <=125ms',train_negative='ended safe windows',
        label_budget_preparation=20,label_budget_calibration=10,inter_updates=0,alarm_updates=0,
        input_map='inherited density_gain; not certified unit calibration',
        fast_reset='once per shot; carries across arrived windows then discarded',
        outcome_access='all test predictions sealed before test labels; no test labels in adaptation',
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in paths},
        evidence='DEVELOPMENT_ONLY; source/target sets exposed; not original first30 commissioning stream',
        gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    _durable_json(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(paths):z.write(f,str(f.relative_to(io23.ROOT)))
    reader=io23.read_outcome;records=[]
    for sid in prep:records.append(labeled(read_signal(sid),reader('EAST',sid),mapping))
    _durable_json(root/'preparation_outcomes.json',[{k:s[k] for k in ('shot','is_disrupt','down')} for s in records])
    def deny(*a,**kw):raise AssertionError('no outcomes in learning or prediction')
    with patch.object(io23,'read_outcome',side_effect=deny):
        history=prepare(model,records)
        _durable_json(root/'training.json',history)
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()}),root/'prepared.pt')
        for sid in cal:
            s=read_signal(sid);signal=transform({k:s[k] for k in ('values','available')},mapping)
            scores=predict_static(model,signal)
            np.savez_compressed(root/'calibration'/f'{sid}.npz',times=s['times'],scores=scores)
    calibration_seals={str(f.relative_to(root)):sha(f) for f in (root/'calibration').glob('*.npz')}
    _durable_json(root/'calibration_seals.json',calibration_seals)
    calibration=[]
    for sid in cal:
        flag,down=reader('EAST',sid)
        with np.load(root/'calibration'/f'{sid}.npz') as z:
            eligible=np.round((down-z['times'])*1000,9)>30 if flag else None
            timely_peak=float(z['scores'][eligible].max()) if flag and eligible.any() else None
            calibration.append(dict(shot=sid,is_disrupt=bool(flag),down=down,
                                    peak=float(z['scores'].max()),timely_peak=timely_peak))
    thresholds={str(a):safe_threshold([r['peak'] for r in calibration if not r['is_disrupt']],a) for a in (.1,.15,.2)}
    _durable_json(root/'calibration_records.json',calibration);_durable_json(root/'thresholds.json',thresholds)
    print(dict(stage='target_prepared_and_calibrated',thresholds=thresholds,seconds=time.monotonic()-start),flush=True)
    viability=calibration_viability(calibration,thresholds['0.2']['threshold'])
    _durable_json(root/'target_start_check.json',viability)
    if not viability['allow_full_stream']:
        _durable_json(root/'completed.json',dict(status='STOPPED_TARGET_START_CHECK',check=viability,
            test_predictions_generated=0,test_outcomes_accessed=0,candidate_promoted=False,
            source_unchanged=sha(DEFAULT_PARENT)==PARENT_SHA256,seconds=time.monotonic()-start))
        print(dict(stage='stopped_before_test360',**viability),flush=True)
        return
    parent_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};seals=[];updates=0
    with patch.object(io23,'read_outcome',side_effect=deny):
        for index,sid in enumerate(ids):
            s=read_signal(sid);signal=transform({k:s[k] for k in ('values','available')},mapping)
            curves,trace=predict_stream(model,signal)
            if index==0:
                np.testing.assert_allclose(curves['off'],predict_static(model,signal),rtol=0,atol=0)
            path=root/'shots'/f'{sid}.npz';np.savez_compressed(path,times=s['times'],**curves)
            tracepath=root/'shots'/f'{sid}_updates.json';_durable_json(tracepath,trace)
            seals.append(dict(shot=sid,prediction_sha256=sha(path),updates_sha256=sha(tracepath)))
            updates+=sum(r['gradient_steps'] for r in trace)
            if (index+1)%10==0:
                progress=dict(shots=index+1,neural_steps=updates,seconds=time.monotonic()-start)
                _durable_json(root/'progress.json',progress);print(progress,flush=True)
    _durable_json(root/'prediction_seals.json',seals)
    for k,v in parent_state.items():
        if not torch.equal(v,model.state_dict()[k].cpu()):raise AssertionError(f'persistent state changed: {k}')
    for k,v in protected.items():
        if not torch.equal(v,model.state_dict()[k].cpu()):raise AssertionError(f'fixed readout changed: {k}')
    results=evaluate(root,ids,thresholds,reader)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    main_result=results['0.2'];on,off=main_result['real'],main_result['off']
    done=dict(status='COMPLETED_ALIGNED_EAST360',metrics=results,seconds=time.monotonic()-start,
        source_unchanged=True,prepared_model_unchanged=True,neural_steps=updates,
        local_J_FPR_screen=on['fpr']<=.2 and on['J']>off['J'],
        temporal_controls_better=on['J']>max(main_result[a]['J'] for a in ('constant','reverse')),
        candidate_promoted=False,
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',done)
    print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':main()
