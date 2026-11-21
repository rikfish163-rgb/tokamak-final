"""Three frozen-design intra-shot arms, same authorized twenty-shot parent."""
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
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .learning import ARMS, OMITTED, estimate_importance, importance_weights, predict

REFERENCE = io23.ROOT/'experiments/risk_aligned_online/runs/matched_preparation20_test360_20261023'
PLAN = io23.ROOT/'docs/research/source_event_rebuild_20261022/任务敏感参数与诊断目标_事前约定_20261023.md'
NAMES = ('static', 'ordinary')+ARMS
READ = lambda p: json.loads(p.read_text())


def load_model(device='cuda'):
    base, info = load_parent(DEFAULT_PARENT)
    model = PairedFeedback(base, np.ones(23)).to(device)
    state = torch.load(REFERENCE/'conditional_moments.pt', map_location=device, weights_only=True)
    model.load_state_dict(state['model']); model.set_stage('frozen')
    return model, float(info['threshold'])


def main(output):
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2); torch.manual_seed(131); started = time.monotonic()
    rp, rv = READ(REFERENCE/'protocol.json'), READ(REFERENCE/'verification.json')
    assert rv['status'] == 'VERIFIED' and sha(REFERENCE/'completed.json') == rv['completed_sha256']
    for file, digest in rp['code_sha256'].items():
        assert sha(io23.ROOT/file) == digest, file
    parent_path = REFERENCE/'conditional_moments.pt'
    assert sha(parent_path) == READ(REFERENCE/'preparation_receipts.json')['conditional_moments']['checkpoint_sha256']
    contract = load_test_contract('EAST'); assert contract == rp['contract']
    ids = rp['selected_ids']; assert len(ids) == 20 and not set(ids)&set(contract['shot_ids'])
    model, threshold = load_model(); original = state_cpu(model)
    assert model.cfg.window == 128 and model.cfg.horizon == 25 and threshold == rp['threshold']
    mapping = READ(REFERENCE/'input_map.json')
    output.mkdir(parents=True); (output/'shots').mkdir()
    paths = {io23.ROOT/f for f in rp['code_sha256']}
    paths.update(Path(__file__).parent.glob('*.py'))
    paths.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_task_geometry.py')
    paths = sorted(paths)
    with zipfile.ZipFile(output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, str(path.relative_to(io23.ROOT)))
    protocol = dict(source_checkpoint=str(DEFAULT_PARENT), source_sha256=PARENT_SHA256,
        parent=str(parent_path), parent_sha256=sha(parent_path), reference=str(REFERENCE),
        reference_hashes={f:sha(REFERENCE/f) for f in ('protocol.json','completed.json','verification.json',
            'input_map.json','selection.json','prediction_receipts.json')},
        code_sha256={str(p.relative_to(io23.ROOT)):sha(p) for p in paths},
        plan_path=str(PLAN), plan_sha256=sha(PLAN), contract=contract, threshold=threshold,
        selected_ids=ids, parameter_scope=model.encoder_names(), arms=list(ARMS), prediction_names=list(NAMES),
        labels='EAST125ms historical targets; strict TEST lead>30ms',
        calibration='32 endpoints/shot; disruptive16positive+16early; repeated if short; equal class masses',
        empirical_importance='single-endpoint BCE gradients squared, NOT square of mean gradient',
        update='weighted direction matched to current FULL forecast gradient norm then cap1; SGD.001',
        omitted_targets=[io23.RAW23[c] for c in OMITTED], omitted_inputs=[],
        no_source_training=True, no_preparation_retraining=True, no_inter_shot_update=True,
        no_alarm_change=True, independent_confirmation=False,
        chronology='historical TRAIN preparation, not first20 shots of chronological commissioning',
        outcome_gate='only20 TRAIN outcomes before all new360x3 predictions sealed; identities previously exposed')
    _durable_json(output/'protocol.json', protocol)
    outcome = io23.read_outcome; access = []
    def guarded(device, sid):
        if device != 'EAST' or sid not in ids:
            raise AssertionError('unauthorized outcome before all prediction seals')
        access.append(dict(device=device,shot=int(sid)))
        return outcome(device,sid)
    with patch.object(io23, 'read_outcome', side_effect=guarded):
        records = []
        for sid in ids:
            s = simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),
                                           'EAST',{})
            s.update(transform({k:s[k] for k in ('values','available')}, mapping)); records.append(s)
        fisher, calibration = estimate_importance(model, records, device='cuda')
        importance, shuffled = importance_weights(fisher)
        weights = dict(importance=importance,shuffled=shuffled)
        torch.save({key:{k:v.cpu() for k,v in value.items()} for key,value in
                    dict(fisher=fisher, **weights).items()}, output/'geometry.pt')
        _durable_json(output/'calibration.json', calibration)
        print(json.dumps(dict(stage='calibration',endpoints=len(calibration),seconds=time.monotonic()-started)),flush=True)
        del records
        refseals = {r['shot']:r for r in READ(REFERENCE/'prediction_receipts.json')}; seals = []
        for index,sid in enumerate(contract['shot_ids']):
            shot = apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal = transform({k:shot[k] for k in ('values','available')}, mapping)
            predictions,updates = predict(model,signal,weights)
            refpath = REFERENCE/'shots'/f'{sid}.npz'
            assert sha(refpath) == refseals[sid]['prediction_sha256']
            with np.load(refpath,allow_pickle=False) as z:
                np.testing.assert_array_equal(shot['times'],z['times'])
                predictions.update(static=z['conditional_moments_static'].copy(),ordinary=z['conditional_moments_intra'].copy())
            for arm in ARMS:
                np.testing.assert_array_equal(predictions[arm][:256],predictions['static'][:256])
            path,up = output/'shots'/f'{sid}.npz',output/'shots'/f'{sid}_updates.json'
            np.savez_compressed(path,times=shot['times'],**predictions); _durable_json(up,updates)
            seals.append(dict(shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up),reference_sha256=sha(refpath)))
            if (index+1)%30 == 0:
                progress = dict(stage='stream',shots=index+1,seconds=time.monotonic()-started)
                _durable_json(output/'progress.json',progress); print(json.dumps(progress),flush=True)
        _durable_json(output/'prediction_receipts.json',seals)
        _durable_json(output/'pre_eval_label_access.json',access)
    # Outcomes are read only after ALL new prediction curves have been sealed.
    rows = {a:[] for a in NAMES}
    for sid in contract['shot_ids']:
        shot = simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(output/'shots'/f'{sid}.npz',allow_pickle=False) as z:
            for a in NAMES:
                rows[a].append(dict(shot=sid,times=shot['times'][shot['valid']],scores=z[a][shot['valid']],
                                    is_disrupt=shot['is_disrupt'],down=shot['down']))
    results = {a:score_fixed_test('EAST',r,threshold) for a,r in rows.items()}
    assert_unchanged(model,original); assert sha(DEFAULT_PARENT) == PARENT_SHA256
    assert sha(parent_path) == protocol['parent_sha256']
    _durable_json(output/'per_shot.json',results)
    completed = dict(status='COMPLETED',seconds=time.monotonic()-started,
        parent_bitwise_unchanged=True,temporary_updates_discarded=True,geometry_sha256=sha(output/'geometry.pt'),
        calibration_sha256=sha(output/'calibration.json'),
        metrics={a:dict(**public(r),very_early_over3000=sum(v['timely'] and v['lead_ms']>3000
                                                        for v in r['predictions'])) for a,r in results.items()})
    _durable_json(output/'completed.json',completed); print(json.dumps(completed),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('output',type=Path)
    main(parser.parse_args().output)
