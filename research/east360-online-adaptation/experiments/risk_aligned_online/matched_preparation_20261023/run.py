"""Change historical control selection only; reuse existing preparation exactly."""
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
from ..deepcoral_prefix_20261022.run import predict_pairs, state_cpu, assert_unchanged
from ..fixed_test_protocol import load_test_contract, score_fixed_test
from ..labeled_alignment_20261023.run import ARMS, STEPS, FORECAST_STEPS, historical_pool, prepare
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent, load_source_train
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .selection import matched_controls

REFERENCE = io23.ROOT/'experiments/risk_aligned_online/runs/labeled_alignment20_test360_20261023'
PLAN = io23.ROOT/'docs/research/source_event_rebuild_20261022/邻近时期安全对照_事前约定_20261023.md'
NAMES = ['raw_source']+[f'{a}_{m}' for a in ARMS for m in ('static', 'intra')]


def main(output):
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2); torch.manual_seed(131); started = time.monotonic()
    read = lambda p: json.loads(p.read_text())
    rp, rv = read(REFERENCE/'protocol.json'), read(REFERENCE/'verification.json')
    assert rv['status'] == 'VERIFIED' and sha(REFERENCE/'completed.json') == rv['completed_sha256']
    for filename, digest in rp['code_sha256'].items():
        assert sha(io23.ROOT/filename) == digest, filename
    contract = load_test_contract('EAST'); assert contract == rp['contract']
    old = read(REFERENCE/'selection.json')
    pool, metadata = historical_pool(contract['shot_ids'])
    assert pool == old['audit']['accepted'] and metadata['excluded'] == old['audit']['excluded']
    cases = [r['shot'] for r in old['selected'] if r['is_disrupt']]
    selected, pairs = matched_controls(pool, cases, test_ids=contract['shot_ids'])
    ids = [r['shot'] for r in selected]
    metadata.update(selection='fixed original4 disruptive; nearest unused safe2before2after per case',
                    signal_and_gradient_budget='selected20 only in this candidate')
    base, info = load_parent(DEFAULT_PARENT)
    parent = PairedFeedback(base, np.ones(23)).cuda(); parent.set_stage('frozen')
    initial = state_cpu(parent); threshold = float(info['threshold']); assert threshold == rp['threshold']
    output.mkdir(parents=True); (output/'shots').mkdir()
    _durable_json(output/'selection.json', dict(audit=metadata, selected=selected, pairs=pairs,
        old_selected_ids=rp['selected_ids'], development_union_ids=sorted(set(ids)|set(rp['selected_ids']))))
    paths = [io23.ROOT/p for p in rp['code_sha256']]+list(Path(__file__).parent.glob('*.py'))
    paths.append(io23.ROOT/'experiments/risk_aligned_online/tests/test_matched_preparation.py')
    hashes = {str(p.relative_to(io23.ROOT)): sha(p) for p in paths}
    with zipfile.ZipFile(output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, str(path.relative_to(io23.ROOT)))
    protocol = dict(source_sha256=PARENT_SHA256, source_checkpoint=str(DEFAULT_PARENT),
        threshold=threshold, contract=contract, arms=ARMS, prediction_names=NAMES,
        budget=20, selected_ids=ids, fixed_disruptive_ids=cases, reference=str(REFERENCE),
        reference_hashes={n:sha(REFERENCE/n) for n in ('protocol.json','completed.json','verification.json',
            'selection.json','prediction_receipts.json','risk_training_manifest.jsonl','forecast_manifest.jsonl')},
        plan_path=str(PLAN), plan_sha256=sha(PLAN), code_sha256=hashes,
        steps=STEPS, forecast_steps=FORECAST_STEPS, parameter_scope=parent.encoder_names(),
        preparation_function='unchanged labeled_alignment_20261023.run.prepare',
        metadata_selection_budget=88, model_fitting_shots=20,
        old_and_new_signal_identity_union=len(set(ids)|set(rp['selected_ids'])),
        chronology='retrospective role-disjoint calibration, NOT chronological commissioning',
        labels='source max25ms AdvancedTime; EAST fixed125ms; strict TEST lead>30ms',
        independent_confirmation=False, intra='unchanged H25x4 arrived signals,SGD.001,clip1,every128,first256,reset eachshot',
        outcome_gate='TRAIN only before all360x8 new curves sealed; class split manifests already known',
        no_alarm_change=True, no_inter_shot_update=True)
    _durable_json(output/'protocol.json', protocol)
    original_outcome = io23.read_outcome; source_ids = set(io23.source_roles()['train']); access = []
    def guarded(device, sid):
        allowed = source_ids if device == 'J-TEXT' else set(ids) if device == 'EAST' else set()
        if sid not in allowed:
            raise AssertionError('nontraining outcome before prediction seals')
        access.append(dict(device=device, shot=int(sid)))
        return original_outcome(device, sid)
    with patch.object(io23, 'read_outcome', side_effect=guarded):
        source, _ = load_source_train()
        models, mapping = prepare(parent, selected, output, source); del source
        # Same source replay despite different target sampling and statistics.
        for filename, length in (('risk_training_manifest.jsonl', STEPS), ('forecast_manifest.jsonl', FORECAST_STEPS)):
            old_rows = [json.loads(s) for s in (REFERENCE/filename).read_text().splitlines()]
            new_rows = [json.loads(s) for s in (output/filename).read_text().splitlines()]
            assert len(old_rows) == len(new_rows) == length
            assert all(a['source'] == b['source'] for a, b in zip(old_rows, new_rows))
        frozen = {a:state_cpu(m) for a,m in models.items()}; seals = []
        refseals = {r['shot']:r for r in read(REFERENCE/'prediction_receipts.json')}
        for index, sid in enumerate(contract['shot_ids']):
            shot = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
            predictions, updates = predict_pairs(models, transform({k:shot[k] for k in ('values','available')}, mapping))
            refpath = REFERENCE/'shots'/f'{sid}.npz'
            assert sha(refpath) == refseals[sid]['prediction_sha256']
            with np.load(refpath, allow_pickle=False) as z:
                np.testing.assert_array_equal(shot['times'], z['times'])
                predictions['raw_source'] = z['raw_source'].copy()
            for arm in ARMS:
                np.testing.assert_array_equal(predictions[arm+'_static'][:256], predictions[arm+'_intra'][:256])
            path, up = output/'shots'/f'{sid}.npz', output/'shots'/f'{sid}_updates.json'
            np.savez_compressed(path, times=shot['times'], **predictions); _durable_json(up, updates)
            seals.append(dict(shot=sid, prediction_sha256=sha(path), updates_sha256=sha(up), reference_sha256=sha(refpath)))
            if (index+1) % 30 == 0:
                progress = dict(stage='stream', shots=index+1, seconds=time.monotonic()-started)
                _durable_json(output/'progress.json', progress); print(json.dumps(progress), flush=True)
        _durable_json(output/'prediction_receipts.json', seals)
        _durable_json(output/'pre_eval_label_access.json', access)
    rows = {a:[] for a in NAMES}
    for sid in contract['shot_ids']:
        shot = simple_data.attach_outcome(simple_data.read_shot('EAST', sid, rate_contract='nominal'), 'EAST', {})
        with np.load(output/'shots'/f'{sid}.npz', allow_pickle=False) as z:
            for arm in NAMES:
                rows[arm].append(dict(shot=sid, times=shot['times'][shot['valid']], scores=z[arm][shot['valid']],
                    is_disrupt=shot['is_disrupt'], down=shot['down']))
    results = {a:score_fixed_test('EAST', r, threshold) for a,r in rows.items()}
    for a,m in models.items():
        assert_unchanged(m, frozen[a])
    assert_unchanged(parent, initial); assert sha(DEFAULT_PARENT) == PARENT_SHA256
    _durable_json(output/'per_shot.json', results)
    completed = dict(status='COMPLETED', seconds=time.monotonic()-started, parent_bitwise_unchanged=True,
        temporary_updates_discarded=True, metrics={a:dict(**public(r),
            very_early_over3000=sum(v['timely'] and v['lead_ms'] > 3000 for v in r['predictions'])) for a,r in results.items()})
    _durable_json(output/'completed.json', completed); print(json.dumps(completed), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
