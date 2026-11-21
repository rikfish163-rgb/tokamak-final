"""Census of existing source event supervision, not a learned event detector.

Only TRAIN/VAL identities enter the census. The supplied CSV itself also
contains historical test annotations; no claim of independent labels is made.
Gaps and event ends remain unknown, never recovery or non-disruptive targets.
No signals, model checkpoints, EAST records or test caches are loaded.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .event_prefix_probe import DEFAULT_LABELS, LABEL_SHA256, assign_events, read_events
from .simple_train import sha
from .simple_online import _durable_json


def summarize(inventory, events, metadata):
    ids = [r['shot'] for r in inventory]
    if len(set(ids)) != len(ids) or any(r['role'] not in ('train', 'val') for r in inventory):
        raise ValueError('unique source TRAIN/VAL identities required; TEST is out of scope')
    result = {}
    for role in ('train', 'val'):
        selected = [r for r in inventory if r['role'] == role]
        family_intervals = {c: dict(D=0, S=0) for c in 'ABCDEF'}
        family_shots = {c: dict(D=0, S=0) for c in 'ABCDEF'}
        transitions = {c: Counter() for c in ('D', 'S')}
        statuses, lengths, duration_ms = Counter(), Counter(), []
        rows = []
        gaps = descending = equal = pairs = 0
        for r in selected:
            sid = r['shot']
            es = events.get(sid, [])
            m = metadata.get(sid)
            status = 'not_in_label_file' if m is None else m['status']
            if m is not None and m['outcome'] != int(r['disruptive']):
                raise ValueError(f'outcome disagreement for {sid}')
            if status not in ('event', 'no_event_detected', 'excluded_short_record', 'not_in_label_file'):
                raise ValueError('undeclared annotation status')
            if bool(es) != (status == 'event'):
                raise ValueError('annotation status/interval mismatch')
            if es and (es[0].index != 1 or any(e.shot != sid for e in es)):
                raise ValueError('event identity/index mismatch')
            assign_events(np.array([0.]), es)  # reuse interval validation only
            statuses[status] += 1
            lengths[len(es)] += 1
            label = 'D' if r['disruptive'] else 'S'
            letters = [chr(ord('A') + e.family) for e in es]
            for c in letters:
                family_intervals[c][label] += 1
            for c in set(letters):
                family_shots[c][label] += 1
            gap_seconds = 0.
            for a, b in zip(es, es[1:]):
                # Match the label reader's nanosecond decimal boundary contract.
                gap = round(b.start - a.end, 9)
                gap_seconds += gap
                gaps += int(gap > 0)
                descending += int(b.family < a.family)
                equal += int(b.family == a.family)
                pairs += 1
                transitions[label][f'{chr(65+a.family)}>{chr(65+b.family)}'] += 1
            duration_ms.extend(round((e.end-e.start)*1000., 6) for e in es)
            rows.append(dict(shot=sid, outcome=label, status=status, event_count=len(es),
                             sequence='>'.join(letters),
                             unannotated_gap_seconds=round(gap_seconds, 9)))
        annotated = sum(n for k, n in lengths.items() if k > 0)
        multiple = sum(n for k, n in lengths.items() if k > 1)
        result[role] = dict(
            shots=len(selected), shots_with_events=annotated, multi_event_shots=multiple,
            multi_event_fraction_of_annotated_shots=multiple/annotated if annotated else None,
            event_count=sum(k*n for k, n in lengths.items()),
            events_per_shot_counts={str(k): n for k, n in sorted(lengths.items())},
            status_counts=dict(statuses), family_intervals=family_intervals,
            family_shots=family_shots, transition_counts={k: dict(v) for k, v in transitions.items()},
            adjacent_event_pairs=pairs, adjacent_with_unannotated_gap=gaps,
            alphabet_descending_pairs=descending, alphabet_equal_pairs=equal,
            alphabet_ascending_pairs=pairs-descending-equal,
            duration_ms=dict(zip(('min', 'median', 'max'),
                                map(float, np.quantile(duration_ms, [0, .5, 1])))) if duration_ms else None,
            recovery_labels_available=False, rows=rows)
    return result


def run(out):
    if out.exists():
        raise FileExistsError(out)
    inventory_path = Path(__file__).parent/'runs/source_task_memory_v2_20261026/inventory.json'
    description = DEFAULT_LABELS.parent/'字段说明.txt'
    label_hash = sha(DEFAULT_LABELS)
    if label_hash != LABEL_SHA256:
        raise ValueError('supplied labels changed; review their semantics before repeating census')
    inventory = json.loads(inventory_path.read_text())
    if Counter(r['role'] for r in inventory) != dict(train=1353, val=160):
        raise ValueError('existing source1353/160 inventory required')
    events, metadata = read_events(DEFAULT_LABELS)
    result = summarize(inventory, events, metadata)
    out.parent.mkdir(parents=True, exist_ok=True)
    _durable_json(out, dict(
        scope='OFFLINE_SOURCE_EVENT_CENSUS_DEVELOPMENT_ONLY_NOT_PREDICTOR_RESULTS',
        input_paths=dict(labels=str(DEFAULT_LABELS), inventory=str(inventory_path), description=str(description)),
        hashes=dict(labels=label_hash, inventory=sha(inventory_path), description=sha(description),
                    implementation=sha(Path(__file__)),
                    event_parser=sha(Path(__file__).with_name('event_prefix_probe.py'))),
        annotations_use_full_record_and_overlap_historical_test=True,
        signal_files_loaded=0, model_checkpoints_loaded=0, model_updates=0,
        interpretation='A-F are candidate families, not ordered hazard stages. Adjacent rows need not form a complete physical chain.',
        summary=result))
    print(json.dumps({k: {a: b for a, b in v.items() if a != 'rows'} for k, v in result.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
