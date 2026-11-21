"""All ten predeclared full-stream curves; component failures stay failed."""
import argparse
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import fixed_test_protocol, io23, simple_data
from ..arrival_stream_20261030.run import comparisons
from ..attention_basis_20261029.learning import MatrixAttention
from ..causal_input import apply_input_contract
from ..context_vector_20261030.run_full import GlobalOutcomeGate, load_archived_shot
from ..maturity_jacobian_20261030.run import configure, read
from ..operational_transport_20261030.run import check_cold_archive, evaluate
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .core import ARMS, NAMES, VARIANTS, operational_status, validate_receipts
from .prepare import fit_policies
from .stream import predict

ARCHIVE = io23.ROOT/'probe_outputs/arrival_reference_east360_v1'
ARCHIVE_SHA = '57dbd7085fba526fb78acee65d14f1033bafc1e3e25d0d54dd7041413e641a35'
PLAN = io23.ROOT/'ANCHORED_FULLSTREAM_DIAGNOSTIC_PLAN_20261031.md'


def report_predictions(rows, threshold):
    results, predictions, _ = evaluate(rows, threshold)
    for name in NAMES:
        results[name]['last90'] = results[name].pop('high_proxy90')
        for row, pred in zip(rows[name], predictions[name], strict=True):
            hit = np.flatnonzero(row['scores'] >= threshold)
            pred['first_alarm_time_seconds'] = float(row['times'][hit[0]]) if len(hit) else None
    sections = {'full360': slice(None), 'cold20': slice(0,20),
                'suffix340': slice(20,None), 'last90': slice(-90,None)}
    paired = {s: comparisons({n: p[part] for n,p in predictions.items()}) for s,part in sections.items()}
    controls = {v: {a: {s: comparisons({'cold': predictions[v+'__'+a][part],
                                       'real': predictions[v+'__real'][part]})['real']
                        for s,part in sections.items()} for a in ARMS[1:]} for v in VARIANTS}
    return results, predictions, paired, controls


def code_files():
    root = io23.ROOT.resolve()
    files = {PLAN.resolve(), *Path(__file__).parent.glob('*.py'),
             root/'tests/test_anchored_fullstream.py', root/'tests/test_anchored_stream.py'}
    files.update(Path(m.__file__).resolve() for n,m in list(sys.modules.items())
                 if n.startswith('experiments.risk_aligned_online.') and getattr(m,'__file__',None))
    return {p.resolve() for p in files}


def seal_predictions(output, ids, gate, archive_barrier, raw_hashes, mapping, net, slow, policies, threshold):
    receipts = []

    def deny(*args, **kwargs):
        raise AssertionError('labels/outcomes blocked during prediction')

    with patch.object(io23,'read_outcome',side_effect=deny), \
            patch.object(io23,'labels_at',side_effect=deny), \
            patch.object(fixed_test_protocol,'load_test_contract',side_effect=deny):
        for i,sid in enumerate(ids):
            if sha(io23.path_for('EAST',sid)) != raw_hashes[str(sid)]:
                raise ValueError('raw input changed')
            times, cold, old_seal = load_archived_shot(ARCHIVE,i,sid,archive_barrier)
            if i < 20:
                curves, updates = {n: cold.copy() for n in NAMES}, []
            else:
                raw = apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                if not np.array_equal(raw['times'],times):
                    raise ValueError('changed raw time grid')
                signal = transform({k:raw[k] for k in ('values','available')},mapping)
                curves, updates = predict(net,signal,slow,policies)
                validate_receipts(updates,len(times))
            if set(curves) != set(NAMES):
                raise ValueError('all ten curves required')
            for name, values in curves.items():
                if values.shape != times.shape or not np.isfinite(values).all() or (values<0).any() or (values>1).any():
                    raise ValueError('invalid probability curve')
                if (name in ('identical','origin__mean_feedback','mature__mean_feedback')
                        and not np.array_equal(values,curves['cold'])):
                    raise AssertionError('zero-action control differs from cold')
            error, disagreements = check_cold_archive(curves['cold'],cold,threshold)
            stem = f'{i+1:04d}_{sid}'
            path = output/'shots'/f'{stem}.npz'
            with path.open('xb') as handle:
                np.savez_compressed(handle,times=times,**curves)
                handle.flush()
                os.fsync(handle.fileno())
            seal = dict(shot=sid,index=i+1,sha256=sha(path),updates=updates,
                        archive_seal_sha256=old_seal,archive_max_abs_error=error,
                        archive_crossing_disagreements=disagreements,startup_archived=i<20)
            sp = output/'shots'/f'{stem}_seal.json'
            _durable_json(sp,seal)
            gate.seal(sid,seal['sha256'])
            receipts.append({k:v for k,v in seal.items() if k!='updates'} | {'seal_sha256':sha(sp)})
            if (i+1)%10==0:
                print(dict(shots=i+1,total=360,suffix_outcomes_read=0),flush=True)
        for i,row in enumerate(receipts):
            stem=f"{i+1:04d}_{row['shot']}"
            if sha(output/'shots'/f'{stem}.npz') != row['sha256'] or sha(output/'shots'/f'{stem}_seal.json') != row['seal_sha256']:
                raise AssertionError('sealed artifact changed')
        barrier=output/'all_predictions_sealed.json'
        _durable_json(barrier,dict(count=len(receipts),seals=gate.seals))
    gate.release(barrier)
    return receipts


def run(output):
    if output.exists() or output.parent != io23.ROOT/'probe_outputs':
        raise ValueError('new direct worktree output required')
    configure()
    if sha(ARCHIVE/'complete.json') != ARCHIVE_SHA:
        raise ValueError('archive changed')
    archive=read(ARCHIVE/'protocol.json')
    ids, reference=archive['order'],Path(archive['reference'])
    for name,digest in archive['reference_hashes'].items():
        if sha(reference/name)!=digest:
            raise ValueError('cold reference changed')
    mapping, raw_hashes = read(reference/'input_map.json'),read(reference/'input_hashes.json')
    if mapping['calibration_ids']!=ids[:20] or len(ids)!=360 or len(set(ids))!=360:
        raise ValueError('fixed order and cold20 required')
    base,parent=load_parent(DEFAULT_PARENT)
    net=MatrixAttention(base.cuda()).eval()
    frozen={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    slow=torch.load(reference/'models/cold.pt',map_location='cuda',weights_only=True)['slow']
    slow_before=slow.clone()
    threshold=float(parent['threshold'])
    if threshold!=archive['threshold']:
        raise ValueError('threshold changed')
    output.mkdir()
    (output/'shots').mkdir()
    files=code_files()
    hashes={str(p.relative_to(io23.ROOT.resolve())):sha(p) for p in sorted(files)}
    protocol=dict(scope='SUPPLEMENTARY_FULL360_AFTER_FAILED_COMPONENT_SCREENS',order=ids,curves=NAMES,
        variants=VARIANTS,reference=str(reference),reference_hashes=archive['reference_hashes'],
        archive=str(ARCHIVE),archive_complete_sha256=ARCHIVE_SHA,source_sha256=PARENT_SHA256,
        threshold=threshold,lead_ms=30.,lead_comparison='strict greater than',warmup=20,
        source_updates=0,inter_updates_after_cold20=0,alarm_updates=0,independent_confirmation=False,
        component_screen='FAILED',promoted=False,steps=256,batch=24,seed=131,lr=.001,eta=.1,radius=.1,
        first_origin=127,horizon=25,first_affected='maturity+1',reset_each_shot=True,
        replace_each_feedback=True,all_predictions_before_evaluation_outcomes=True,
        raw_startup_predictions_reused=True,code_sha256=hashes,torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name())
    _durable_json(output/'protocol.json',protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            z.write(path,str(path.relative_to(io23.ROOT.resolve())))
    started=time.monotonic()
    policies=fit_policies(net,slow,ids[:20],output)
    fixed={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    gate=GlobalOutcomeGate(ids,io23.read_outcome)
    receipts=seal_predictions(output,ids,gate,read(ARCHIVE/'all_predictions_sealed.json'),raw_hashes,
                              mapping,net,slow,policies,threshold)
    rows={n:[] for n in NAMES}
    counts={n:0 for n in NAMES}
    for i,sid in enumerate(ids):
        stem=f'{i+1:04d}_{sid}'
        path=output/'shots'/f'{stem}.npz'
        if sha(path)!=receipts[i]['sha256']:
            raise ValueError('sealed prediction changed before evaluation')
        with np.load(path,allow_pickle=False) as saved:
            flag,down=gate.read_outcome('EAST',sid)
            _,valid=io23.labels_at(saved['times'],flag,down,125.)
            for n in NAMES:
                rows[n].append(dict(shot=sid,times=saved['times'][valid],scores=saved[n][valid],
                                    is_disrupt=flag,down=down))
        for item in read(output/'shots'/f'{stem}_seal.json')['updates']:
            for n in NAMES:
                counts[n]+=int(item['state_norm'][n]>0 and item['apply_count']>0)
    results,predictions,paired,controls=report_predictions(rows,threshold)
    if (sha(DEFAULT_PARENT)!=PARENT_SHA256 or not torch.equal(slow,slow_before)
            or any(not torch.equal(v,net.state_dict()[k].cpu()) for k,v in frozen.items())):
        raise AssertionError('frozen source or slow changed')
    for name,digest in hashes.items():
        if sha(io23.ROOT/name)!=digest:
            raise AssertionError('code changed during run')
    for name,digest in fixed.items():
        if sha(output/name)!=digest:
            raise AssertionError('fitted artifact changed')
    for name,digest in archive['reference_hashes'].items():
        if sha(reference/name)!=digest:
            raise AssertionError('fixed parent changed')
    for name,value in [('per_shot.json',predictions),('receipts.json',receipts),('accesses.json',gate.accesses)]:
        _durable_json(output/name,value)
    complete=dict(complete=True,results=results,comparisons=paired,real_vs_controls=controls,
        operational_status=operational_status(results),applied_updates=counts,
        seconds=time.monotonic()-started,source_unchanged=True,component_screen='FAILED',promoted=False,
        scientific_goal_achieved=False,independent_confirmation=False,
        fpr_caps={n:{str(cap):results[n]['full360']['fpr']<=cap for cap in (.10,.15,.20)} for n in NAMES},
        archive_max_abs_error=max(r['archive_max_abs_error'] for r in receipts),
        artifact_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    _durable_json(output/'complete.json',complete)
    print({n:r['full360'] for n,r in results.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
