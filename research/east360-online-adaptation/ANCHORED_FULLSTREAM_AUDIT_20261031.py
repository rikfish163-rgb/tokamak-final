#!/usr/bin/env python3
"""Read-only, independent NumPy audit of the completed anchored full stream.

Only sealed predictions, JSON metadata, an explicitly supplied outcome JSON,
the immutable source checkpoint bytes, and the saved code snapshot are read.
No experiment module, HDF5, model, or GPU is opened. JSON goes to stdout.

The outcome JSON is a list of {"shot": int, "is_disrupt": bool, "down": number
or null} objects, or an object with that list under the "outcomes" key. It must
contain exactly the locked 360 shot IDs; safe shots have null down.
"""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np


VARIANTS = ('origin', 'mature')
ARMS = ('real', 'reverse', 'mean_feedback', 'donor')
NAMES = ('cold', 'identical', *(v + '__' + a for v in VARIANTS for a in ARMS))
SECTIONS = {'full360': slice(None), 'cold20': slice(0, 20),
            'suffix340': slice(20, None), 'last90': slice(-90, None)}
THRESHOLD = 0.7258946895599365
SOURCE_SHA256 = 'c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02'
ARCHIVE_SHA256 = '57dbd7085fba526fb78acee65d14f1033bafc1e3e25d0d54dd7041413e641a35'
ARCHIVE_PROTOCOL_SHA256 = '0b65901186e7960ca08ba023be9a52934907ad4b4800fa59bc41965206f8565f'
ARCHIVE_BARRIER_SHA256 = 'a7fd8e1fd9ffbef3b12ecdeda02259a53bc4441f1df5b0cafe537f97309bac7b'
SOURCE_PATH = 'experiments/risk_aligned_online/runs/gradient_aligned512_20260929/source.pt'
REFERENCE_PATH = 'experiments/risk_aligned_online/runs/lifecycle_context8360_service_20261028'
RUNS_RELATIVE = Path('experiments/risk_aligned_online/runs')
SHARED_RUNS_ROOT = Path('/media/hetaisheng/044A81D94A81C83E/Fuison/experiments/risk_aligned_online/runs')
REFERENCE_FILES = {'completed.json', 'input_hashes.json', 'input_map.json',
                   'models/cold.pt', 'order.json', 'protocol.json'}


class AuditError(ValueError):
    """An artifact or independently computed result failed verification."""


class FileStore:
    """Minimal read-only filesystem adapter; tests can supply an in-memory one."""

    def exists(self, path):
        return Path(path).is_file()

    def read_bytes(self, path):
        return Path(path).read_bytes()

    def sha256(self, path):
        value = hashlib.sha256()
        with Path(path).open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                value.update(chunk)
        return value.hexdigest()

    def files(self, directory):
        return {path.name for path in Path(directory).iterdir() if path.is_file()}


def require(condition, message):
    if not condition:
        raise AuditError(message)


def safe_child(base, name):
    """Constrain manifest paths and categorically exclude HDF5 inputs."""
    require(isinstance(name, str) and bool(name), 'invalid artifact path')
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts and
            relative.suffix.lower() not in ('.h5', '.hdf5'), 'unsafe or HDF5 artifact path')
    base = Path(base).resolve()
    result = (base / relative).resolve()
    require(result.is_relative_to(base) and result != base and
            result.suffix.lower() not in ('.h5', '.hdf5'), 'artifact escapes its root or is HDF5')
    return result


def contracted_artifact(root, name):
    """Allow only the two locked artifacts through the known shared runs link."""
    require(name in (SOURCE_PATH, REFERENCE_PATH), 'not a contracted source or reference path')
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts and
            relative.is_relative_to(RUNS_RELATIVE), 'invalid contracted runs path')
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if result.is_relative_to(root):
        return safe_child(root, name)
    shared = SHARED_RUNS_ROOT.resolve()
    expected = (shared / relative.relative_to(RUNS_RELATIVE)).resolve()
    require((root / RUNS_RELATIVE).resolve() == shared and result == expected and
            result.is_relative_to(shared) and result.suffix.lower() not in ('.h5', '.hdf5'),
            'unapproved shared runs link or artifact escape')
    return result


def checked_bytes(store, path, expected=None):
    data = store.read_bytes(path)
    if expected is not None:
        require(hashlib.sha256(data).hexdigest() == expected, 'read artifact hash differs: ' + str(path))
    return data


def read_json(store, path, expected=None):
    return json.loads(checked_bytes(store, path, expected))


def read_npz(store, path, expected=None):
    with np.load(io.BytesIO(checked_bytes(store, path, expected)), allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def compare(actual, expected, where):
    """Compare full schemas and values, allowing only tiny roundoff in floats."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), where + ': keys differ')
        for key in expected:
            compare(actual[key], expected[key], where + '/' + str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), where + ': length differs')
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            compare(a, b, where + '/' + str(index))
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(type(actual) is type(expected) and actual == expected, where + ': value differs')
    elif isinstance(expected, int):
        require(type(actual) is int and actual == expected, where + ': integer value differs')
    elif isinstance(expected, float):
        require(not isinstance(actual, bool) and isinstance(actual, (int, float)) and
                math.isfinite(actual) and math.isfinite(expected) and
                math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8),
                where + ': numeric value differs')
    else:
        raise AuditError(where + ': unsupported comparison value')


def score_shot(shot, times, scores, is_disrupt, down, threshold):
    """Recompute the first preterminal crossing and the strict >30 ms rule."""
    times, scores = np.asarray(times), np.asarray(scores)
    require(times.ndim == 1 and len(times) > 0 and scores.shape == times.shape and
            np.isfinite(times).all() and np.isfinite(scores).all() and
            (np.diff(times) > 0).all() and ((scores >= 0) & (scores <= 1)).all(),
            'invalid time grid or probability curve')
    require(type(is_disrupt) is bool and math.isfinite(threshold), 'invalid outcome or threshold')
    if is_disrupt:
        require(isinstance(down, (float, int)) and math.isfinite(down),
                'finite disruption timestamp required')
        valid = (down - times) * 1000. >= -1e-6
    else:
        valid = np.ones(len(times), dtype=bool)
    hits = np.flatnonzero(valid & (scores >= threshold))
    first = float(times[hits[0]]) if len(hits) else None
    lead = (float(round((down - times[hits[0]]) * 1000., 9))
            if is_disrupt and first is not None else None)
    return dict(shot=int(shot), is_disrupt=is_disrupt, alarm=first is not None, lead_ms=lead,
                timely=lead is not None and lead > 30., first_alarm_time_seconds=first)


def summarize(records, threshold):
    """Report each shot once; FN combines missing and tardy disruptive alarms."""
    nd = sum(row['is_disrupt'] for row in records)
    ns = len(records) - nd
    tp = sum(row['timely'] for row in records)
    fp = sum(row['alarm'] and not row['is_disrupt'] for row in records)
    leads = [row['lead_ms'] for row in records if row['timely']]
    tpr, fpr = (tp / nd if nd else None), (fp / ns if ns else None)
    return dict(tp=tp, fp=fp, n_disrupt=nd, n_safe=ns, tpr=tpr, fpr=fpr,
                J=tpr - fpr if nd and ns else None, threshold=float(threshold),
                median_lead_ms=float(np.median(leads)) if leads else None,
                early_over3000=sum(lead > 3000. for lead in leads), fn=nd - tp, tn=ns - fp,
                tardy=sum(row['is_disrupt'] and row['alarm'] and not row['timely']
                          for row in records),
                missed=sum(row['is_disrupt'] and not row['alarm'] for row in records))


def paired_changes(reference, candidate):
    require(len(reference) == len(candidate), 'paired shot counts differ')
    rescued, harmed = [], []
    for old, new in zip(reference, candidate, strict=True):
        require(old['shot'] == new['shot'] and old['is_disrupt'] == new['is_disrupt'],
                'paired shot identity or outcome differs')
        was_good = old['timely'] if old['is_disrupt'] else not old['alarm']
        is_good = new['timely'] if new['is_disrupt'] else not new['alarm']
        if is_good and not was_good:
            rescued.append(old['shot'])
        if was_good and not is_good:
            harmed.append(old['shot'])
    return dict(rescued=rescued, harmed=harmed)


def check_shot_records(shot, times, curves, saved, outcome, threshold):
    """Recompute every arm using externally supplied outcome and actual down."""
    require(set(curves) == set(saved) == set(NAMES), 'per-shot branch set differs')
    require(outcome['shot'] == shot, 'authoritative outcome shot differs')
    records = {}
    for name in NAMES:
        records[name] = score_shot(shot, times, curves[name], outcome['is_disrupt'],
                                   outcome['down'], threshold)
        compare(records[name], saved[name], f'per_shot/{shot}/{name}')
        for field in ('lead_ms', 'first_alarm_time_seconds'):
            require(records[name][field] == saved[name][field],
                    f'per_shot/{shot}/{name}/{field}: value differs')
    return records


def validate_outcomes(payload, ids):
    """Require one well-formed independent outcome for each locked shot."""
    rows = payload.get('outcomes') if isinstance(payload, dict) else payload
    require(isinstance(rows, list) and len(rows) == 360, '360 authoritative outcomes required')
    result = {}
    for row in rows:
        require(isinstance(row, dict) and {'shot', 'is_disrupt', 'down'} <= set(row) and
                type(row['shot']) is int and type(row['is_disrupt']) is bool,
                'invalid authoritative outcome row')
        down = row['down']
        require((row['is_disrupt'] and isinstance(down, (float, int)) and
                 not isinstance(down, bool) and math.isfinite(down)) or
                (not row['is_disrupt'] and down is None), 'invalid authoritative down time')
        require(row['shot'] not in result, 'duplicate authoritative outcome shot')
        result[row['shot']] = row
    require(set(result) == set(ids) and sum(row['is_disrupt'] for row in rows) == 180,
            'authoritative outcomes must match the locked 180/180 cohort')
    return result


def validate_updates(updates, curves, *, startup):
    """Check receipt clocks and every interval known to have zero action."""
    require(set(curves) == set(NAMES), 'curve branch set differs')
    length, cold = len(curves['cold']), curves['cold']
    counts = {name: 0 for name in NAMES}
    for name in ('identical', 'origin__mean_feedback', 'mature__mean_feedback'):
        require(np.array_equal(curves[name], cold), 'zero reference curve differs from cold')
    if startup:
        require(updates == [] and all(np.array_equal(curves[name], cold) for name in NAMES),
                'archived startup curves or updates differ')
        return counts
    require(all(np.array_equal(curves[name][:153], cold[:153]) for name in NAMES),
            'per-shot zero reset or pre-maturity scores differ')
    events = list(range(152, length, 25))
    require(len(updates) == len(events), 'feedback event count differs')
    for event, row in zip(events, updates, strict=True):
        require((row['origin'], row['maturity'], row['first_affected'], row['apply_count']) ==
                (event - 25, event, event + 1, min(25, length - event - 1)),
                'feedback clock or apply_count differs')
        norms, cosines = row['state_norm'], row['cos_real']
        require(set(norms) == set(cosines) == set(NAMES), 'state receipt branch set differs')
        require(type(row['feedback_supported']) is bool, 'invalid feedback support flag')
        require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and
                    math.isfinite(v) and 0 <= v <= .100001 for v in norms.values()),
                'state norm exceeds radius or is nonfinite')
        require(isinstance(row['max_bound'], (int, float)) and
                math.isfinite(row['max_bound']) and
                math.isclose(row['max_bound'], max(norms.values()), rel_tol=0, abs_tol=1e-8),
                'state norm maximum differs')
        for name in ('cold', 'identical', 'origin__mean_feedback', 'mature__mean_feedback'):
            require(norms[name] == 0, 'zero reference state has nonzero norm')
        require(row['feedback_supported'] or not any(norms.values()),
                'unsupported feedback must produce zero replacement')
        if 'donor_index' in row:
            require(set(row['donor_index']) == set(VARIANTS) and
                    all(type(i) is int and 0 <= i < 2707 for i in row['donor_index'].values()),
                    'donor index is outside the declared cold20 opportunity pool')
        for variant in VARIANTS:
            real = norms[variant + '__real']
            for arm in ('reverse', 'donor'):
                control = norms[variant + '__' + arm]
                require(control == 0 or math.isclose(control, real, rel_tol=2e-5, abs_tol=0.),
                        'nonzero control norm is not matched to real')
        for name in NAMES:
            variant = name.split('__')[0]
            real = norms.get(variant + '__real', 0.)
            cosine = cosines[name]
            if norms[name] == 0 or real == 0:
                require(cosine is None, 'zero state has a defined cosine')
            else:
                require(isinstance(cosine, (float, int)) and not isinstance(cosine, bool) and
                        math.isfinite(cosine) and abs(cosine) <= 1.00001,
                        'invalid cosine relative to real')
                if name.endswith('__real'):
                    require(math.isclose(cosine, 1., abs_tol=1e-6), 'real self-cosine differs')
            if norms[name] == 0:
                end = event + 1 + row['apply_count']
                require(np.array_equal(curves[name][event + 1:end], cold[event + 1:end]),
                        'zero replacement does not reproduce cold scores')
            counts[name] += int(norms[name] > 0 and row['apply_count'] > 0)
    return counts


def audit(output, *, root, outcomes, store=None, include_rows=False):
    """Audit an already-completed run using explicit outcomes, never raw HDF5."""
    store = FileStore() if store is None else store
    root, output = Path(root).resolve(), Path(output).resolve()
    outcomes_arg = Path(outcomes).expanduser()
    outcomes_path = outcomes_arg.resolve()
    require(outcomes_arg.suffix.lower() == '.json' and outcomes_path.suffix.lower() == '.json',
            'authoritative outcomes must be an explicit JSON file')
    require(output.parent == (root / 'probe_outputs').resolve(), 'output must be a direct run folder')
    complete_path = output / 'complete.json'
    require(store.exists(complete_path), 'complete.json does not exist; audit must wait for completion')
    complete_bytes = checked_bytes(store, complete_path)  # before any saved outcome access
    complete_hash = hashlib.sha256(complete_bytes).hexdigest()
    complete = json.loads(complete_bytes)
    require(complete.get('complete') is True, 'run is not complete')
    require(complete.get('source_unchanged') is True and complete.get('component_screen') == 'FAILED'
            and complete.get('promoted') is False and complete.get('scientific_goal_achieved') is False
            and complete.get('independent_confirmation') is False, 'completion identity differs')
    manifest = complete['artifact_sha256']
    required = {'protocol.json', 'code_snapshot.zip', 'origin.pt', 'mature.pt',
                'origin_training.json', 'mature_training.json', 'all_predictions_sealed.json',
                'receipts.json', 'accesses.json', 'per_shot.json'}
    require(not outcomes_path.is_relative_to(output) or outcomes_path.parent == output,
            'outcomes sidecar cannot be inside the sealed shots directory')
    sidecar = {outcomes_path.name} if outcomes_path.parent == output else set()
    require(not sidecar.intersection(set(manifest) | {'complete.json'}),
            'outcomes sidecar cannot replace a sealed artifact')
    require(isinstance(manifest, dict) and required <= set(manifest) and
            set(manifest) == store.files(output) - {'complete.json'} - sidecar,
            'complete artifact set differs')
    for name, expected in manifest.items():
        path = safe_child(output, name)
        require(path.parent == output and store.sha256(path) == expected,
                'artifact hash differs: ' + name)
    protocol = read_json(store, output / 'protocol.json', manifest['protocol.json'])
    identity = dict(scope='SUPPLEMENTARY_FULL360_AFTER_FAILED_COMPONENT_SCREENS',
                    curves=list(NAMES), variants=list(VARIANTS), threshold=THRESHOLD,
                    source_sha256=SOURCE_SHA256, archive_complete_sha256=ARCHIVE_SHA256,
                    lead_ms=30., lead_comparison='strict greater than', warmup=20,
                    source_updates=0, inter_updates_after_cold20=0, alarm_updates=0,
                    independent_confirmation=False, component_screen='FAILED', promoted=False,
                    steps=256, batch=24, seed=131, lr=.001, eta=.1, radius=.1,
                    first_origin=127, horizon=25, first_affected='maturity+1',
                    reset_each_shot=True, replace_each_feedback=True,
                    all_predictions_before_evaluation_outcomes=True,
                    raw_startup_predictions_reused=True)
    for key, expected in identity.items():
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            require(type(protocol[key]) in (int, float) and protocol[key] == expected,
                    'protocol/' + key + ': fixed value differs')
        else:
            compare(protocol[key], expected, 'protocol/' + key)
    ids = protocol['order']
    require(isinstance(ids, list) and len(ids) == 360 and len(set(ids)) == 360 and
            all(type(sid) is int and sid > 0 for sid in ids), 'invalid full360 order')
    require(store.sha256(contracted_artifact(root, SOURCE_PATH)) == SOURCE_SHA256,
            'qualified source hash differs')
    archive = (root / 'probe_outputs/arrival_reference_east360_v1').resolve()
    require(Path(protocol['archive']).resolve() == archive, 'archive identity differs')
    require(store.sha256(archive / 'complete.json') == ARCHIVE_SHA256, 'archive complete hash differs')
    require(store.sha256(archive / 'protocol.json') == ARCHIVE_PROTOCOL_SHA256,
            'archive protocol hash differs')
    require(store.sha256(archive / 'all_predictions_sealed.json') == ARCHIVE_BARRIER_SHA256,
            'archive barrier hash differs')
    old = read_json(store, archive / 'protocol.json', ARCHIVE_PROTOCOL_SHA256)
    require(old['order'] == ids and old['threshold'] == THRESHOLD and
            old['reference'] == protocol['reference'] == REFERENCE_PATH and
            old['reference_hashes'] == protocol['reference_hashes'], 'archive/reference identity differs')
    reference = contracted_artifact(root, protocol['reference'])
    require(set(protocol['reference_hashes']) == REFERENCE_FILES, 'reference manifest differs')
    for name, expected in protocol['reference_hashes'].items():
        require(store.sha256(safe_child(reference, name)) == expected, 'reference hash differs: ' + name)
    require([row['shot'] for row in read_json(store, reference / 'order.json',
                                             protocol['reference_hashes']['order.json'])] == ids,
            'reference shot order differs')
    require(read_json(store, reference / 'input_map.json',
                      protocol['reference_hashes']['input_map.json'])['calibration_ids'] == ids[:20],
            'reference cold20 mapping differs')
    code_hashes = protocol['code_sha256']
    require(isinstance(code_hashes, dict) and bool(code_hashes), 'missing source code hashes')
    snapshot_bytes = checked_bytes(store, output / 'code_snapshot.zip', manifest['code_snapshot.zip'])
    with zipfile.ZipFile(io.BytesIO(snapshot_bytes)) as snapshot:
        members = snapshot.namelist()
        require(len(members) == len(set(members)) and set(members) == set(code_hashes),
                'source snapshot member set differs')
        for name, expected in code_hashes.items():
            path = safe_child(root, name)
            require(store.sha256(path) == expected and
                    hashlib.sha256(snapshot.read(name)).hexdigest() == expected,
                    'live or snapshot source hash differs: ' + name)

    barrier = read_json(store, output / 'all_predictions_sealed.json',
                        manifest['all_predictions_sealed.json'])
    old_barrier = read_json(store, archive / 'all_predictions_sealed.json', ARCHIVE_BARRIER_SHA256)
    receipts = read_json(store, output / 'receipts.json', manifest['receipts.json'])
    accesses = read_json(store, output / 'accesses.json', manifest['accesses.json'])
    require(barrier['count'] == old_barrier['count'] == len(receipts) == 360 and
            len(barrier['seals']) == len(old_barrier['seals']) == len(accesses) == 360,
            'barrier or receipt count differs')
    require(accesses == [dict(shot=sid, sealed_count=360) for sid in ids],
            'outcome access ledger is not globally sealed')
    filenames = {f'{i:04d}_{sid}{suffix}' for i, sid in enumerate(ids, 1)
                 for suffix in ('.npz', '_seal.json')}
    require(store.files(output / 'shots') == filenames and store.files(archive / 'shots') == filenames,
            'full360 shot/seal file set differs')
    seals = []
    for i, sid in enumerate(ids, 1):
        stem = f'{i:04d}_{sid}'
        path, seal_path = output / 'shots' / (stem + '.npz'), output / 'shots' / (stem + '_seal.json')
        old_path = archive / 'shots' / (stem + '.npz')
        old_seal_path = archive / 'shots' / (stem + '_seal.json')
        seal = read_json(store, seal_path, receipts[i - 1]['seal_sha256'])
        old_seal = read_json(store, old_seal_path, seal['archive_seal_sha256'])
        actual, old_actual = store.sha256(path), store.sha256(old_path)
        require(seal['shot'] == old_seal['shot'] == sid and
                seal['index'] == old_seal['index'] == i and
                seal['sha256'] == actual and old_seal['sha256'] == old_actual,
                f'shot {sid} prediction seal or hash differs')
        require(barrier['seals'][i - 1] == dict(shot=sid, sha256=actual) and
                old_barrier['seals'][i - 1] == dict(shot=sid, sha256=old_actual),
                f'shot {sid} barrier hash differs')
        require(seal['archive_seal_sha256'] == store.sha256(old_seal_path),
                f'shot {sid} archived seal hash differs')
        expected_receipt = {k: v for k, v in seal.items() if k != 'updates'} | {
            'seal_sha256': store.sha256(seal_path)}
        require(receipts[i - 1] == expected_receipt and seal['startup_archived'] is (i <= 20),
                f'shot {sid} receipt or startup identity differs')
        seals.append(seal)

    # Outcomes are opened only after completion, full hash checks and the 360-shot barrier.
    outcome_bytes = checked_bytes(store, outcomes_path)
    outcome_hash = hashlib.sha256(outcome_bytes).hexdigest()
    truth = validate_outcomes(json.loads(outcome_bytes), ids)
    saved = read_json(store, output / 'per_shot.json', manifest['per_shot.json'])
    require(set(saved) == set(NAMES) and all(len(saved[name]) == 360 for name in NAMES),
            'saved per-shot branch/row count differs')
    predictions = {name: [] for name in NAMES}
    counts = {name: 0 for name in NAMES}
    max_error = 0.
    for i, sid in enumerate(ids, 1):
        stem = f'{i:04d}_{sid}'
        shot = read_npz(store, output / 'shots' / (stem + '.npz'), seals[i - 1]['sha256'])
        old_shot = read_npz(store, archive / 'shots' / (stem + '.npz'),
                            old_barrier['seals'][i - 1]['sha256'])
        require(set(shot) == {'times', *NAMES} and {'times', 'cold'} <= set(old_shot),
                f'shot {sid} prediction arrays differ')
        times, curves = shot['times'], {name: shot[name] for name in NAMES}
        require(times.ndim == 1 and len(times) > 0 and np.isfinite(times).all() and
                (np.diff(times) > 0).all() and np.array_equal(times, old_shot['times']),
                f'shot {sid} time grid differs')
        require(all(v.shape == times.shape and np.isfinite(v).all() and
                    ((v >= 0) & (v <= 1)).all() for v in curves.values()),
                f'shot {sid} invalid score curve')
        require(np.array_equal(curves['cold'], old_shot['cold']), f'shot {sid} archived cold differs')
        error = float(np.max(np.abs(curves['cold'] - old_shot['cold'])))
        disagreements = int(np.count_nonzero((curves['cold'] >= THRESHOLD) !=
                                             (old_shot['cold'] >= THRESHOLD)))
        require(error == seals[i - 1]['archive_max_abs_error'] and disagreements ==
                seals[i - 1]['archive_crossing_disagreements'], f'shot {sid} archive comparison differs')
        max_error = max(max_error, error)
        action_counts = validate_updates(seals[i - 1]['updates'], curves, startup=i <= 20)
        for name in NAMES:
            counts[name] += action_counts[name]
        shot_saved = {name: saved[name][i - 1] for name in NAMES}
        records = check_shot_records(sid, times, curves, shot_saved, truth[sid], THRESHOLD)
        for name in NAMES:
            predictions[name].append(records[name])
    require(sum(row['is_disrupt'] for row in predictions['cold']) == 180,
            'saved outcomes are not 180 disruptive and 180 safe')
    compare(counts, complete['applied_updates'], 'applied_updates')
    compare(max_error, complete['archive_max_abs_error'], 'archive_max_abs_error')

    metrics = {name: {section: summarize(rows[part], THRESHOLD)
                      for section, part in SECTIONS.items()} for name, rows in predictions.items()}
    derived = {'fn', 'tn', 'tardy', 'missed'}
    compact = {name: {section: {k: v for k, v in row.items() if k not in derived}
                      for section, row in groups.items()} for name, groups in metrics.items()}
    compare(compact, complete['results'], 'metrics')
    paired = {section: {name: paired_changes(predictions['cold'][part], predictions[name][part])
                        for name in NAMES} for section, part in SECTIONS.items()}
    compare(paired, complete['comparisons'], 'comparisons')
    controls = {v: {a: {section: paired_changes(predictions[v + '__' + a][part],
                                                predictions[v + '__real'][part])
                        for section, part in SECTIONS.items()} for a in ARMS[1:]} for v in VARIANTS}
    compare(controls, complete['real_vs_controls'], 'real_vs_controls')
    cold = metrics['cold']['full360']
    status = {}
    for variant in VARIANTS:
        real = metrics[variant + '__real']['full360']
        status[variant] = dict(
            operational_improvement=(real['J'] > cold['J'] and real['tpr'] >= cold['tpr'] and
                                     real['fpr'] <= .20),
            real_exceeds_controls=all(real['J'] > metrics[variant + '__' + arm]['full360']['J']
                                      for arm in ('reverse', 'donor')),
            original_component_screen='FAILED', independent_confirmation=False)
    compare(status, complete['operational_status'], 'operational_status')
    caps = {name: {str(cap): metrics[name]['full360']['fpr'] <= cap for cap in (.1, .15, .2)}
            for name in NAMES}
    compare(caps, complete['fpr_caps'], 'fpr_caps')
    require(store.sha256(complete_path) == complete_hash, 'completion metadata changed during audit')
    require(store.sha256(outcomes_path) == outcome_hash, 'authoritative outcomes changed during audit')
    report = dict(verified=True, checks=dict(shots=360, archived_shots=360,
                  top_level_hashes=len(manifest), source_files=len(code_hashes),
                  reference_files=len(REFERENCE_FILES), applied_updates=counts,
                  outcomes_path=str(outcomes_path), outcomes_sha256=outcome_hash),
                  metrics=metrics, comparisons=paired,
                  real_vs_controls=controls, operational_status=status, fpr_caps=caps,
                  limitations=[
                      'Outcome JSON provenance and raw HDF5 hashes must be checked by its creator.',
                      'Receipt and zero-curve checks cannot prove unsaved gradient tensors or actual reset operations.',
                      'This development-exposed diagnostic is not independent confirmation.'])
    if include_rows:
        report['per_shot'] = predictions
    return report


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=root / 'probe_outputs/anchored_full360_diagnostic_v1')
    parser.add_argument('--outcomes', type=Path, required=True,
                        help='explicit authoritative outcome JSON, created after completion')
    parser.add_argument('--rows', action='store_true', help='include all recomputed per-shot rows')
    args = parser.parse_args()
    try:
        result = audit(args.output, root=root, outcomes=args.outcomes, include_rows=args.rows)
    except (AuditError, OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        print(json.dumps(dict(verified=False, error=str(exc)), allow_nan=False), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
