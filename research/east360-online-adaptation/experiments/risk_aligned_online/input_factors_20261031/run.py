"""Complete fixed stream: isolated input changes, no neural or alarm updates."""
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
from ..attention_basis_20261029.learning import MatrixAttention
from ..causal_input import apply_input_contract
from ..context_vector_20261030.run_full import GlobalOutcomeGate, load_archived_shot
from ..maturity_jacobian_20261030.run import configure, read
from ..operational_transport_20261030.run import check_cold_archive, evaluate
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .core import NAMES, fit_maps
from .integrity import verify_archived_controls, verify_local_seals
from .stream import predict

ARCHIVE = io23.ROOT/'probe_outputs/anchored_full360_diagnostic_v1'
ARCHIVE_SHA = '6ac657ecfcd2e7c283854a070cf1dc6647c2bf58981bd10639a4835ef8327b6a'
PLAN = io23.ROOT/'INPUT_FACTORS_PLAN_20261031.md'


def flip_types(predictions):
    result = {}
    for name, rows in predictions.items():
        detail = dict(gained_tp=[],lost_tp=[],removed_fp=[],added_fp=[])
        for a,b in zip(predictions['cold'],rows,strict=True):
            assert a['shot']==b['shot'] and a['is_disrupt']==b['is_disrupt']
            if a['is_disrupt']:
                if b['timely'] and not a['timely']:
                    detail['gained_tp'].append(a['shot'])
                if a['timely'] and not b['timely']:
                    detail['lost_tp'].append(a['shot'])
            else:
                if a['alarm'] and not b['alarm']:
                    detail['removed_fp'].append(a['shot'])
                if b['alarm'] and not a['alarm']:
                    detail['added_fp'].append(a['shot'])
        result[name] = detail
    return result


def run(output):
    if output.exists() or output.parent!=io23.ROOT/'probe_outputs':
        raise ValueError('fresh direct worktree output required')
    configure()
    verify_archived_controls(ARCHIVE,ARCHIVE_SHA)
    previous = read(ARCHIVE/'protocol.json')
    ids, reference = previous['order'],Path(previous['reference'])
    for name,digest in previous['reference_hashes'].items():
        if sha(reference/name)!=digest:
            raise ValueError('reference changed: '+name)
    mapping = read(reference/'input_map.json')
    raw_hashes = read(reference/'input_hashes.json')
    maps = fit_maps(mapping,ids)
    base,parent = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda()).eval().requires_grad_(False)
    before = {k:v.cpu().clone() for k,v in net.state_dict().items()}
    slow = torch.load(reference/'models/cold.pt',map_location='cuda',weights_only=True)['slow']
    slow_before = slow.clone()
    threshold = float(parent['threshold'])
    if threshold!=previous['threshold']:
        raise ValueError('threshold changed')
    output.mkdir()
    (output/'shots').mkdir()
    files = {PLAN, *Path(__file__).parent.glob('*.py'),
             io23.ROOT/'tests/test_input_factors_20261031.py',
             io23.ROOT/'tests/test_input_factor_integrity.py'}
    files.update(Path(m.__file__) for n,m in list(sys.modules.items())
                 if n.startswith('experiments.risk_aligned_online.') and getattr(m,'__file__',None))
    code_hashes = {str(p.relative_to(io23.ROOT)):sha(p) for p in sorted(files)}
    protocol = dict(scope='DEVELOPMENT_ONLY_INPUT_SENSITIVITY_COMPLETE360',order=ids,
        archive=str(ARCHIVE),archive_complete_sha256=ARCHIVE_SHA,
        reference=str(reference),reference_hashes=previous['reference_hashes'],
        source_sha256=PARENT_SHA256,threshold=threshold,arms=NAMES,warmup=20,
        transformations=maps,first_affected_shot_index=21,baseline='frozen cold20',
        neural_updates=0,alarm_updates=0,independent_confirmation=False,
        physical_calibration_certified=False,promoted=False,code_sha256=code_hashes,
        timeout_seconds=1800,integrity_version=2,
        torch=str(torch.__version__),gpu=torch.cuda.get_device_name())
    _durable_json(output/'protocol.json',protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.write(p,str(p.relative_to(io23.ROOT)))
    gate = GlobalOutcomeGate(ids,io23.read_outcome)
    barrier = read(ARCHIVE/'all_predictions_sealed.json')
    receipts = []
    started = time.monotonic()

    def deny(*args,**kwargs):
        raise AssertionError('no outcomes while generating input interventions')

    with patch.object(io23,'read_outcome',side_effect=deny), \
            patch.object(io23,'labels_at',side_effect=deny), \
            patch.object(fixed_test_protocol,'load_test_contract',side_effect=deny):
        for i,sid in enumerate(ids):
            if time.monotonic()-started>1800:
                raise TimeoutError('input diagnostic exceeded declared 1800 seconds')
            if sha(io23.path_for('EAST',sid))!=raw_hashes[str(sid)]:
                raise ValueError('raw input changed')
            times,cold,archive_seal = load_archived_shot(ARCHIVE,i,sid,barrier)
            if i<20:
                curves = {n:cold.copy() for n in NAMES}
            else:
                raw = apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                if not np.array_equal(times,raw['times']):
                    raise ValueError('changed signal clock')
                signal = transform({k:raw[k] for k in ('values','available')},mapping)
                curves = predict(net,signal,slow,maps)
            error,disagreements = check_cold_archive(curves['cold'],cold,threshold)
            stem = f'{i+1:04d}_{sid}'
            path = output/'shots'/f'{stem}.npz'
            with path.open('xb') as f:
                np.savez_compressed(f,times=times,**curves)
                f.flush()
                os.fsync(f.fileno())
            seal = dict(shot=sid,index=i+1,sha256=sha(path),archive_seal_sha256=archive_seal,
                        archive_max_abs_error=error,archive_crossing_disagreements=disagreements,
                        neural_updates=0,outcomes_used=False,startup_archived=i<20)
            seal_path = output/'shots'/f'{stem}_seal.json'
            _durable_json(seal_path,seal)
            gate.seal(sid,seal['sha256'])
            receipts.append(dict(seal,seal_sha256=sha(seal_path)))
            if (i+1)%10==0:
                progress = dict(shots=i+1,total=360,seconds=time.monotonic()-started,outcomes_read=0)
                _durable_json(output/'progress.json',progress)
                print(progress,flush=True)
        verify_local_seals(output,receipts)
        barrier_path = output/'all_predictions_sealed.json'
        _durable_json(barrier_path,dict(count=360,seals=gate.seals,
                                      seal_sha256=[r['seal_sha256'] for r in receipts]))
    gate.release(barrier_path)
    rows = {n:[] for n in NAMES}
    for i,sid in enumerate(ids):
        path = output/'shots'/f'{i+1:04d}_{sid}.npz'
        if sha(path)!=receipts[i]['sha256']:
            raise ValueError('sealed curve changed')
        with np.load(path,allow_pickle=False) as a:
            flag,down = gate.read_outcome('EAST',sid)
            _,valid = io23.labels_at(a['times'],flag,down,125.)
            for n in NAMES:
                rows[n].append(dict(shot=sid,times=a['times'][valid],scores=a[n][valid],
                                    is_disrupt=flag,down=down))
    results,predictions,paired = evaluate(rows,threshold)
    for value in results.values():
        value['last90'] = value.pop('high_proxy90')
    if (not torch.equal(slow,slow_before) or sha(DEFAULT_PARENT)!=PARENT_SHA256
            or any(not torch.equal(v,net.state_dict()[k].cpu()) for k,v in before.items())):
        raise AssertionError('frozen parent changed')
    for name,digest in code_hashes.items():
        if sha(io23.ROOT/name)!=digest:
            raise AssertionError('executed code changed')
    for name,digest in previous['reference_hashes'].items():
        if sha(reference/name)!=digest:
            raise AssertionError('reference artifact changed')
    verify_archived_controls(ARCHIVE,ARCHIVE_SHA)
    verify_local_seals(output,receipts)
    _durable_json(output/'per_shot.json',predictions)
    _durable_json(output/'receipts.json',receipts)
    _durable_json(output/'accesses.json',gate.accesses)
    done = dict(complete=True,results=results,paired=paired,flips=flip_types(predictions),
        seconds=time.monotonic()-started,frozen_source_and_cold_unchanged=True,
        archive_max_abs_error=max(r['archive_max_abs_error'] for r in receipts),
        scientific_goal_achieved=False,neural_updates=0,physical_calibration_certified=False,
        promoted=False,independent_confirmation=False,
        artifact_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    _durable_json(output/'complete.json',done)
    print({n:r['full360'] for n,r in results.items()},flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
