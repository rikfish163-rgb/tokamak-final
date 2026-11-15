"""Independently recount the completed context-vector diagnostic on CPU.

This checks saved curves, seals, outcomes and runner attestations. It does not
replay the network, establish physical mechanisms, or promote a candidate.
"""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import zipfile

import numpy as np

from .. import io23


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'probe_outputs/context_vector_full360_diagnostic_v1'
PREVIOUS = ROOT / 'probe_outputs/arrival_reference_east360_v1'
POLICY = ROOT / 'probe_outputs/context_vector_factorial_v1'
PARENT = ROOT / 'experiments/risk_aligned_online/runs/gradient_aligned512_20260929/source.pt'
PARENT_SHA = 'c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02'
PREVIOUS_SHA = '57dbd7085fba526fb78acee65d14f1033bafc1e3e25d0d54dd7041413e641a35'
MANIFEST_SHA = '657cab542bd987f9c97c3cb88867a9cc828373b32e6ee58d450451461c6ec825'
VARIANTS = ('raw_fixed', 'scaled_fixed', 'raw_calibrated', 'scaled_calibrated')
ACTIONS = ('real', 'reverse', 'mean_feedback', 'constant', 'risk_axis')
NAMES = ('cold', 'identical', *(v + '__' + a for v in VARIANTS for a in ACTIONS))
SECTIONS = {'full360': slice(None), 'cold20': slice(0, 20),
            'suffix340': slice(20, None), 'last90': slice(-90, None)}
METRIC_KEYS = ('tp', 'fp', 'n_disrupt', 'n_safe', 'tpr', 'fpr', 'J', 'threshold',
               'median_lead_ms', 'early_over3000')


def read_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def shot_decision(shot, times, scores, is_disrupt, down, threshold):
    """First threshold crossing, with the runner's preterminal tolerance."""
    times, scores = np.asarray(times), np.asarray(scores)
    require(times.ndim == 1 and scores.shape == times.shape and len(times) > 0,
            'invalid decision curve shape')
    require(np.isfinite(times).all() and np.isfinite(scores).all()
            and (scores >= 0).all() and (scores <= 1).all()
            and (np.diff(times) > 0).all() and math.isfinite(threshold),
            'invalid decision curve values')
    if is_disrupt:
        require(down is not None and math.isfinite(down), 'invalid disruption time')
        valid = (down - times) * 1000 >= -1e-6
    else:
        valid = np.ones(len(times), dtype=bool)
    hits = np.flatnonzero(valid & (scores >= threshold))
    first = float(times[hits[0]]) if len(hits) else None
    lead = (float(round((down - times[hits[0]]) * 1000., 9))
            if is_disrupt and first is not None else None)
    return dict(shot=int(shot), is_disrupt=bool(is_disrupt), alarm=first is not None,
                lead_ms=lead, timely=bool(lead is not None and lead > 30.),
                first_alarm_time_seconds=first)


def summarize(records, threshold):
    nd = sum(row['is_disrupt'] for row in records)
    ns = len(records) - nd
    tp = sum(row['timely'] for row in records)
    fp = sum(row['alarm'] and not row['is_disrupt'] for row in records)
    tpr = tp / nd if nd else None
    fpr = fp / ns if ns else None
    leads = [row['lead_ms'] for row in records if row['timely']]
    return dict(tp=tp, fp=fp, n_disrupt=nd, n_safe=ns, tpr=tpr, fpr=fpr,
                J=tpr - fpr if nd and ns else None, threshold=float(threshold),
                median_lead_ms=float(np.median(leads)) if leads else None,
                early_over3000=sum(lead > 3000 for lead in leads))


def equal_value(actual, saved):
    if isinstance(actual, bool) or isinstance(saved, bool):
        return type(actual) is type(saved) and actual == saved
    if isinstance(actual, int):
        return type(saved) is int and actual == saved
    if isinstance(actual, (int, float)) and isinstance(saved, (int, float)):
        return math.isfinite(actual) and math.isfinite(saved) and math.isclose(
            actual, saved, rel_tol=0., abs_tol=1e-9)
    return actual == saved


def compare_metrics(actual, saved):
    require(set(saved) == set(METRIC_KEYS), 'saved metric fields differ')
    for name in METRIC_KEYS:
        require(equal_value(actual[name], saved[name]),
                f'metric {name} differs: recomputed={actual[name]!r}, saved={saved[name]!r}')


def paired_changes(baseline, candidate):
    require(len(baseline) == len(candidate), 'paired lengths differ')
    rescued, harmed = [], []
    for old, new in zip(baseline, candidate, strict=True):
        require(old['shot'] == new['shot'] and old['is_disrupt'] == new['is_disrupt'],
                'paired shot or outcome differs')
        old_good = old['timely'] if old['is_disrupt'] else not old['alarm']
        new_good = new['timely'] if new['is_disrupt'] else not new['alarm']
        if new_good and not old_good:
            rescued.append(old['shot'])
        if old_good and not new_good:
            harmed.append(old['shot'])
    return dict(rescued=rescued, harmed=harmed)


def verify_curve_contract(times, curves, archive_times, archive_cold, updates, startup):
    """Check exact archive/control equality, reset prefix, cadence and support."""
    require(set(curves) == set(NAMES), '22 curve names required')
    times, archive_times, archive_cold = map(np.asarray, (times, archive_times, archive_cold))
    require(times.ndim == 1 and len(times) > 0 and np.isfinite(times).all()
            and (np.diff(times) > 0).all(), 'invalid time axis')
    for name, values in curves.items():
        values = np.asarray(values)
        require(values.shape == times.shape and np.isfinite(values).all()
                and (values >= 0).all() and (values <= 1).all(), f'invalid curve {name}')
    cold = curves['cold']
    require(np.array_equal(times, archive_times) and np.array_equal(cold, archive_cold),
            'cold is not exactly archived')
    require(np.array_equal(cold, curves['identical']), 'cold and identical differ')
    prefix = len(times) if startup else min(153, len(times))
    for name, values in curves.items():
        require(np.array_equal(values[:prefix], cold[:prefix]), f'causal prefix differs: {name}')
    events = [] if startup else list(range(152, len(times), 25))
    require(len(updates) == len(events), 'receipt cadence count differs')
    for event, receipt in zip(events, updates, strict=True):
        require((receipt['origin'], receipt['maturity'], receipt['first_affected'], receipt['apply_count'])
                == (event - 25, event, event + 1, min(25, len(times) - event - 1)),
                'receipt timing or cadence differs')
        norms, cosines = receipt['state_norm'], receipt['cos_real']
        require(set(norms) == set(NAMES) and set(cosines) == set(NAMES), 'receipt arm keys differ')
        require(type(receipt['feedback_supported']) is bool, 'receipt support flag missing')
        require(all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0
                    for x in norms.values()), 'receipt action norm invalid')
        require(norms['cold'] == norms['identical'] == 0
                and math.isfinite(receipt['max_bound'])
                and math.isclose(receipt['max_bound'], max(norms.values()), abs_tol=1e-6),
                'receipt zero controls or max norm differ')
        for name, value in cosines.items():
            require(value is None or (math.isfinite(value) and abs(value) <= 1 + 1e-5),
                    f'receipt cosine invalid: {name}')
            require(norms[name] != 0 or value is None, f'zero direction cosine invalid: {name}')
        if not receipt['feedback_supported']:
            require(not any(norms.values()), 'unsupported feedback did not reset action to zero')
            start, stop = event + 1, event + 1 + receipt['apply_count']
            for name, values in curves.items():
                require(np.array_equal(values[start:stop], cold[start:stop]),
                        f'unsupported feedback reset curve differs: {name}')


def require_completed_barrier(directory):
    """Fail closed before accessing labels or result artifacts without a seal."""
    directory = Path(directory)
    complete_path = directory / 'complete.json'
    if not complete_path.is_file():
        raise FileNotFoundError(f'complete.json required before audit: {complete_path}')
    complete = read_json(complete_path)
    require(complete.get('complete') is True, 'completed run required')
    artifacts = complete['artifact_sha256']
    expected = {'protocol.json', 'code_snapshot.zip', 'all_predictions_sealed.json',
                'per_shot.json', 'receipts.json', 'accesses.json'}
    require(set(artifacts) == expected, 'completed artifact list differs')
    for name in ('protocol.json', 'all_predictions_sealed.json'):
        require(sha(directory / name) == artifacts[name], f'completed artifact changed: {name}')
    protocol = read_json(directory / 'protocol.json')
    barrier = read_json(directory / 'all_predictions_sealed.json')
    ids = protocol['order']
    require(len(ids) == len(set(ids)) == 360 and all(type(s) is int and s > 0 for s in ids),
            'ordered unique full360 required')
    require(barrier.get('count') == 360 and len(barrier.get('seals', [])) == 360
            and len(barrier.get('receipt_sha256', [])) == 360
            and [s['shot'] for s in barrier['seals']] == ids, 'global360 seal identities differ')
    for digest in [x['sha256'] for x in barrier['seals']] + barrier['receipt_sha256']:
        require(isinstance(digest, str) and len(digest) == 64
                and all(c in '0123456789abcdef' for c in digest), 'invalid global seal hash')
    return complete, protocol, barrier


def verify_provenance(directory, complete, protocol):
    require(protocol['scope'] == complete['scope'] == 'DIAGNOSTIC_FULL360_AFTER_FAILED_COMPONENT_GATE',
            'diagnostic scope differs')
    require(protocol['curves'] == list(NAMES) and protocol['variants'] == list(VARIANTS)
            and protocol['update_arms'] == list(ACTIONS), 'preset arms differ')
    failed = {v: 'FAILED' for v in VARIANTS}
    require(protocol['component_gate_status'] == complete['component_gate_status'] == failed
            and protocol['promoted'] is False and complete['promoted'] is False
            and complete['scientific_goal_achieved'] is False
            and protocol['independent_confirmation'] is False
            and complete['independent_confirmation'] is False, 'failed gates or interpretation changed')
    for key in ('reset_each_shot', 'replace_each_feedback', 'no_feedback_replaces_with_zero',
                'startup_reused_from_sealed_archive', 'outcomes_after_all_predictions'):
        require(protocol[key] is True, f'protocol {key} not asserted')
    require(protocol['warmup'] == 20 and protocol['first_origin'] == 127
            and protocol['horizon'] == protocol['update_cadence'] == 25
            and protocol['first_affected'] == 'maturity + 1'
            and protocol['lead_ms'] == 30. and protocol['lead_comparison'] == 'strict greater than'
            and protocol['source_updates'] == protocol['inter_updates_after_cold20']
            == protocol['alarm_updates'] == 0, 'locked operational protocol differs')
    require(protocol['source_sha256'] == PARENT_SHA and sha(PARENT) == PARENT_SHA
            and complete['source_unchanged'] is True and complete['fixed_cold_unchanged'] is True
            and complete['source_parameter_updates'] == 0, 'parent/source evidence differs')
    artifacts = complete['artifact_sha256']
    for name in ('code_snapshot.zip', 'receipts.json', 'accesses.json'):
        require(sha(directory / name) == artifacts[name], f'completed artifact changed: {name}')
    with zipfile.ZipFile(directory / 'code_snapshot.zip') as archive:
        names = protocol['code_sha256']
        require(set(archive.namelist()) == set(names), 'code snapshot entries differ')
        for name, digest in names.items():
            path = (ROOT / name).resolve()
            require(path.is_relative_to(ROOT.resolve()) and sha(path) == digest
                    and hashlib.sha256(archive.read(name)).hexdigest() == digest,
                    f'code or snapshot changed: {name}')
    require(protocol['plan_sha256'] == names['CONTEXT_VECTOR_FULLSTREAM_DIAGNOSTIC_PLAN_20261030.md'],
            'plan digest differs')
    require(Path(protocol['previous']).resolve() == PREVIOUS.resolve()
            and protocol['previous_complete_sha256'] == PREVIOUS_SHA
            and sha(PREVIOUS / 'complete.json') == PREVIOUS_SHA, 'fixed archive completion changed')
    for name, digest in protocol['previous_hashes'].items():
        require(sha(PREVIOUS / name) == digest, f'archive metadata changed: {name}')
    require(set(protocol['previous_hashes']) == {'protocol.json', 'all_predictions_sealed.json'},
            'archive metadata hashes missing')
    previous = read_json(PREVIOUS / 'protocol.json')
    old_barrier = read_json(PREVIOUS / 'all_predictions_sealed.json')
    reference = Path(protocol['reference'])
    require(previous['order'] == protocol['order'] and previous['source_sha256'] == PARENT_SHA
            and previous['threshold'] == protocol['threshold']
            and (ROOT / previous['reference']).resolve() == reference.resolve()
            and previous['reference_hashes'] == protocol['reference_hashes'],
            'archive/reference protocol differs')
    require(old_barrier['count'] == 360 and len(old_barrier['seals']) == 360
            and [s['shot'] for s in old_barrier['seals']] == protocol['order'],
            'archived global360 seal differs')
    for name, digest in protocol['reference_hashes'].items():
        require(sha(reference / name) == digest, f'cold reference changed: {name}')
    require([row['shot'] for row in read_json(reference / 'order.json')] == protocol['order']
            and read_json(reference / 'input_map.json')['calibration_ids'] == protocol['order'][:20],
            'reference order or cold20 mapping differs')
    require(Path(protocol['policy']).resolve() == POLICY.resolve(), 'policy location differs')
    for name, digest in protocol['policy_hashes'].items():
        require(sha(POLICY / name) == digest, f'policy provenance changed: {name}')
    policy = read_json(POLICY / 'complete.json')
    require(policy['complete'] is True and policy['source_unchanged'] is True
            and policy['full_stream_run'] is False and policy['suffix_labels_read'] == 0
            and policy['selected_for_full_stream'] is None and set(policy['variants']) == set(VARIANTS)
            and all(policy['variants'][v]['full_stream_gate'] is False for v in VARIANTS),
            'underlying component failure gates differ')
    return reference, old_barrier, {v: policy['variants'][v]['reasons'] for v in VARIANTS}


def verify_all_seals(directory, protocol, barrier, old_barrier):
    """Verify every identity/hash before accessing per-shot performance or outcomes."""
    receipts = read_json(directory / 'receipts.json')
    require(len(receipts) == 360, '360 receipt summaries required')
    for i, sid in enumerate(protocol['order']):
        name = f'{i + 1:04d}_{sid}'
        shot, seal_path = directory / 'shots' / (name + '.npz'), directory / 'shots' / (name + '_seal.json')
        old_shot = PREVIOUS / 'shots' / (name + '.npz')
        old_seal_path = PREVIOUS / 'shots' / (name + '_seal.json')
        seal_hash, old_seal_hash = sha(seal_path), sha(old_seal_path)
        seal, old_seal = read_json(seal_path), read_json(old_seal_path)
        digest, old_digest = sha(shot), sha(old_shot)
        require(seal['shot'] == sid and seal['index'] == i + 1 and seal['sha256'] == digest
                and barrier['seals'][i] == {'shot': sid, 'sha256': digest}
                and barrier['receipt_sha256'][i] == seal_hash, f'new shot/receipt seal differs: {sid}')
        require(old_seal['shot'] == sid and old_seal['index'] == i + 1
                and old_seal['sha256'] == old_digest
                and old_barrier['seals'][i] == {'shot': sid, 'sha256': old_digest}
                and seal['archive_seal_sha256'] == old_seal_hash, f'archive shot seal differs: {sid}')
        summary = {k: v for k, v in seal.items() if k != 'updates'}
        summary['seal_sha256'] = seal_hash
        require(summary == receipts[i] and seal['outcomes_used_in_prediction'] is False
                and seal['startup_archived'] is (i < 20)
                and seal['archive_max_abs_error'] == 0
                and seal['archive_crossing_disagreements'] == 0, f'receipt identity differs: {sid}')
        if (i + 1) % 60 == 0:
            print(f'verified shot and archive seals: {i + 1}/360', flush=True)
    accesses = read_json(directory / 'accesses.json')
    require(accesses == [dict(shot=s, sealed_count=360) for s in protocol['order']],
            'outcome-access receipt differs from global barrier')
    return receipts


def verified_npz(path, digest):
    """Hash the exact bytes decoded, so a changed file cannot be silently used."""
    data = Path(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == digest, f'sealed curve changed: {path}')
    with np.load(io.BytesIO(data), allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def audit(directory):
    directory = Path(directory)
    complete, protocol, barrier = require_completed_barrier(directory)
    completion_sha = sha(directory / 'complete.json')
    reference, old_barrier, failed_reasons = verify_provenance(directory, complete, protocol)
    receipts = verify_all_seals(directory, protocol, barrier, old_barrier)
    require(sha(directory / 'per_shot.json') == complete['artifact_sha256']['per_shot.json'],
            'saved per-shot decisions changed')
    declared = read_json(directory / 'per_shot.json')
    require(set(declared) == set(NAMES) and all(len(declared[n]) == 360 for n in NAMES),
            'saved per-shot arm coverage differs')
    require(sha(io23.EAST_COHORT) == MANIFEST_SHA, 'locked manifest changed')
    manifest = read_json(io23.EAST_COHORT)['test']
    disruptive, safe = set(manifest['disruption_shots']), set(manifest['non_disruption_shots'])
    require(len(disruptive) == len(safe) == 180 and not disruptive & safe
            and disruptive | safe == set(protocol['order']), 'locked class identities differ')
    raw_hashes = read_json(reference / 'input_hashes.json')
    records = {name: [] for name in NAMES}
    update_counts = {name: 0 for name in NAMES}
    applied_counts = {name: 0 for name in NAMES}
    point_counts = {name: 0 for name in NAMES}
    max_norm, unsupported_events, unsupported_points = 0., 0, 0
    threshold = float(protocol['threshold'])
    for i, sid in enumerate(protocol['order']):
        name = f'{i + 1:04d}_{sid}'
        path = directory / 'shots' / (name + '.npz')
        seal_path = directory / 'shots' / (name + '_seal.json')
        old_path = PREVIOUS / 'shots' / (name + '.npz')
        require(sha(seal_path) == receipts[i]['seal_sha256'], f'receipt changed during audit: {sid}')
        seal = read_json(seal_path)
        current = verified_npz(path, barrier['seals'][i]['sha256'])
        old = verified_npz(old_path, old_barrier['seals'][i]['sha256'])
        require(set(current) == {'times', *NAMES}, f'curve archive fields differ: {sid}')
        curves = {arm: current[arm] for arm in NAMES}
        verify_curve_contract(current['times'], curves, old['times'], old['cold'], seal['updates'], i < 20)
        require(sha(io23.path_for('EAST', sid)) == raw_hashes[str(sid)], f'raw input changed: {sid}')
        # These are the first actual HDF5 outcome reads, after complete + all 360 seals.
        flag, down = io23.read_outcome('EAST', sid)
        require(flag == (sid in disruptive), f'HDF5/manifest outcome differs: {sid}')
        for arm in NAMES:
            record = shot_decision(sid, current['times'], curves[arm], flag, down, threshold)
            saved = declared[arm][i]
            require(set(saved) == set(record) and all(equal_value(record[k], saved[k]) for k in record),
                    f'per-shot decision differs: {sid} {arm}')
            records[arm].append(record)
        for item in seal['updates']:
            max_norm = max(max_norm, item['max_bound'])
            if not item['feedback_supported']:
                unsupported_events += 1
                unsupported_points += item['apply_count']
            for arm in NAMES:
                active = item['state_norm'][arm] > 0
                update_counts[arm] += int(active)
                applied_counts[arm] += int(active and item['apply_count'] > 0)
                point_counts[arm] += item['apply_count'] if active else 0
        if (i + 1) % 60 == 0:
            print(f'recounted completed shots: {i + 1}/360', flush=True)
    require(update_counts == complete['update_counts'] and applied_counts == complete['applied_update_counts']
            and point_counts == complete['applied_point_counts']
            and equal_value(max_norm, complete['max_state_norm']), 'update totals differ')
    require(complete['archive_max_abs_error'] == 0 and complete['archive_crossing_disagreements'] == 0,
            'completed archive comparison differs')
    results = {arm: {part: summarize(rows[cut], threshold) for part, cut in SECTIONS.items()}
               for arm, rows in records.items()}
    require(set(complete['results']) == set(NAMES), 'completed result arms differ')
    for arm in NAMES:
        require(set(complete['results'][arm]) == set(SECTIONS), f'completed sections differ: {arm}')
        for part in SECTIONS:
            try:
                compare_metrics(results[arm][part], complete['results'][arm][part])
            except ValueError as exc:
                raise ValueError(f'{arm}/{part}: {exc}') from exc
    paired = {part: {arm: paired_changes(records['cold'][cut], records[arm][cut]) for arm in NAMES}
              for part, cut in SECTIONS.items()}
    controls = {v: {control: {part: paired_changes(records[v + '__' + control][cut],
                                                    records[v + '__real'][cut])
                             for part, cut in SECTIONS.items()}
                    for control in ACTIONS[1:]} for v in VARIANTS}
    require(paired == complete['comparisons'] and controls == complete['real_vs_controls'],
            'paired rescue/harm lists differ')
    caps = {arm: {part: {str(cap): None if m['fpr'] is None else m['fpr'] <= cap
                         for cap in (.10, .15, .20)} for part, m in values.items()}
            for arm, values in results.items()}
    require(caps == complete['fpr_cap_status'], 'fixed FPR cap statuses differ')
    require(sha(directory / 'complete.json') == completion_sha, 'completion changed during audit')
    require(sha(PARENT) == PARENT_SHA, 'parent changed during audit')
    confusion = {arm: {part: dict(tp=m['tp'], fn=m['n_disrupt'] - m['tp'],
                                  fp=m['fp'], tn=m['n_safe'] - m['fp'])
                       for part, m in values.items()} for arm, values in results.items()}
    identities = [dict(index=i + 1, shot=sid, prediction_sha256=barrier['seals'][i]['sha256'],
                       receipt_sha256=receipts[i]['seal_sha256'],
                       archive_prediction_sha256=old_barrier['seals'][i]['sha256'],
                       archive_receipt_sha256=receipts[i]['archive_seal_sha256'],
                       raw_input_sha256=raw_hashes[str(sid)])
                  for i, sid in enumerate(protocol['order'])]
    return dict(audit_passed=True, scope='independent CPU recount of saved diagnostic curves; not a GPU replay',
                run=str(directory), complete_sha256=completion_sha,
                global_barrier_sha256=complete['artifact_sha256']['all_predictions_sealed.json'],
                source_evidence=dict(parent_sha256=PARENT_SHA, current_parent_sha256=sha(PARENT),
                                     runner_attests_source_unchanged=complete['source_unchanged'],
                                     runner_attests_fixed_cold_unchanged=complete['fixed_cold_unchanged'],
                                     code_snapshot_sha256=complete['artifact_sha256'].get('code_snapshot.zip'),
                                     code_files_verified=len(protocol.get('code_sha256', {})),
                                     reference_files_verified=len(protocol.get('reference_hashes', {})),
                                     policy_files_verified=len(protocol.get('policy_hashes', {})),
                                     caveat='in-memory immutability and prediction-time access are runner attestations, not independently replayed'),
                checks=dict(shots=360, arms=len(NAMES), sections=list(SECTIONS),
                            shot_and_receipt_hashes_verified=360, archived_hashes_verified=360,
                            raw_input_hashes_verified=360, exact_cold_archive=360,
                            cold_identical_equal=360, zero_state_prefix_equal=360,
                            unsupported_events=unsupported_events, unsupported_applied_points=unsupported_points,
                            per_shot_decisions_compared=360 * len(NAMES),
                            outcome_access_records_all_after_global_seal=True),
                component_gate_status=complete['component_gate_status'], component_failed_reasons=failed_reasons,
                promoted=False, scientific_goal_achieved=False, independent_confirmation=False,
                seal_identities=identities, results=results, confusion_counts=confusion, comparisons=paired,
                real_vs_controls=controls, fpr_cap_status=caps,
                update_counts=update_counts, applied_update_counts=applied_counts,
                applied_point_counts=point_counts, max_state_norm=max_norm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path,
                        help='new .json file directly inside the completed diagnostic directory')
    args = parser.parse_args()
    output = args.output.absolute()
    require(output.parent.resolve() == RUN.resolve() and output.suffix == '.json',
            'output must be a JSON file directly inside the completed diagnostic directory')
    if os.path.lexists(output):
        raise FileExistsError(f'will not overwrite audit: {output}')
    report = audit(RUN)
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    with output.open('x', encoding='utf-8') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(f"CPU audit passed; 360 shots, {len(NAMES)} arms; report: {output}")


if __name__ == '__main__':
    main()
