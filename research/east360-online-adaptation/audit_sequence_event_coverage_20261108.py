"""Post-hoc event coverage of source training episodes on current HDF clocks."""
import json
import os
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.risk_aligned_online import event_prefix_probe as event_probe, io23, simple_data
from experiments.risk_aligned_online.source_event_tasks_20261102.data import load_events

ROOT = Path(__file__).resolve().parent
FIT = ROOT / 'probe_outputs/risk_teacher_sequence_source_20261108_v1'
OUTPUT = FIT / 'event_coverage_verified.json'
GROUPS = ('0_to_0', '0_to_1', '1_to_1')
FAMILIES = 'ABCDEF'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def flags(ids):
    known = [i != -1 for i in ids]
    return dict(first_origin_known=known[0], query_known=known[4],
                same_known_first_last_origin=known[0] and known[3] and ids[0] == ids[3],
                same_known_first_query=known[0] and known[4] and ids[0] == ids[4],
                all_origins_query_one_known_event=all(known) and len(set(ids)) == 1)


def event_label(family, event_id):
    family, event_id = int(family), int(event_id)
    require((family == -1 and event_id == -1) or (0 <= family < 6 and event_id > 0),
            'invalid family/event ID pair')
    return dict(family=FAMILIES[family] if family >= 0 else -1,
                event_id=event_id, known=event_id != -1)


def summary(rows):
    query = Counter(r['query_event']['family'] for r in rows)
    result = dict(sequences=len(rows), unique_shots=len({r['shot'] for r in rows}))
    for key in flags([-1] * 5):
        result[key] = sum(r[key] for r in rows)
    result['query_family_counts'] = {family: query[family] for family in FAMILIES}
    result['query_unknown'] = query[-1]
    require(sum(result['query_family_counts'].values()) == result['query_known']
            and result['query_known'] + result['query_unknown'] == len(rows),
            'query coverage does not reconcile')
    require(result['all_origins_query_one_known_event'] <= result['same_known_first_query']
            <= min(result['first_origin_known'], result['query_known'])
            and result['all_origins_query_one_known_event'] <= result['same_known_first_last_origin'],
            'event coverage nesting does not reconcile')
    return result


def main():
    for key, value in dict(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
                           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1').items():
        require(os.environ.get(key) == value, f'required environment: {key}={value!r}')
    require(not OUTPUT.exists(), f'refusing to overwrite {OUTPUT}')
    fit = json.loads((FIT / 'fit.json').read_text())
    protocol = json.loads((FIT / 'protocol.json').read_text())
    history = fit['history']
    require(len(history) == protocol['source_steps'] == 512 and protocol['sequence_steps'] == 4,
            'expected the existing 512 four-step history records')
    train = simple_data.source_roles()['train']
    require(protocol['source_train_ids'] == train, 'protocol and canonical TRAIN lists differ')
    shot_ids = {r['shot'] for r in history}
    require(shot_ids <= set(train), 'history contains a non-TRAIN shot')
    for r in history:
        o, q, t = r['origins'], r['query'], r['label_time']
        require(r['group'] in GROUPS and len(o) == 4
                and all(type(x) is int for x in [r['shot'], *o, q, t])
                and o[0] >= 0 and all(o[i+1] - o[i] == 32 for i in range(3))
                and q == o[-1] + 26 and t == q + 25, 'invalid history sequence')

    # Preserve the original parser's metadata checks. CSV outcome metadata is
    # validated but not used to assign intervals or replace HDF5 outcomes.
    events, _ = load_events()

    # Check same family is insufficient and unknown endpoints never match.
    sample = [event_probe.Event(1, 1, 0, 0., 1.), event_probe.Event(1, 2, 0, 1., 2.)]
    family, ids = event_probe.assign_events(np.array([.1, .2, .3, 1.1, 2.1]), sample)
    require(family.tolist() == [0, 0, 0, 0, -1] and ids.tolist() == [1, 1, 1, 2, -1]
            and not flags(ids.tolist())['same_known_first_last_origin']
            and not flags(ids.tolist())['same_known_first_query']
            and not flags([-1]*5)['all_origins_query_one_known_event']
            and flags([1]*5)['all_origins_query_one_known_event'], 'synthetic event checks failed')

    reads, clocks, read_signals = [], {}, io23.read_signals
    def read_train(device, sid, *args, **kwargs):
        require(device == 'J-TEXT' and sid in shot_ids and sid not in reads,
                'signal read outside unique source TRAIN history IDs')
        reads.append(sid)
        return read_signals(device, sid, *args, **kwargs)
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('outcome access forbidden')) as outcome, \
         patch.object(simple_data, 'attach_outcome', side_effect=RuntimeError('outcome attach forbidden')) as attach, \
         patch.object(io23, 'read_signals', side_effect=read_train):
        for sid in sorted(shot_ids):
            times = np.asarray(simple_data.read_shot('J-TEXT', sid)['times'], dtype=np.float64)
            families, event_ids = event_probe.assign_events(times, events.get(sid, []))
            clocks[sid] = times, families, event_ids
    require(set(reads) == shot_ids and len(reads) == len(shot_ids)
            and outcome.call_count == attach.call_count == 0, 'signal/outcome access mismatch')

    records = []
    for r in history:
        times, family, event_ids = clocks[r['shot']]
        require(r['label_time'] < len(times), 'history endpoint outside current diagnostic clock')
        indices = [*r['origins'], r['query']]
        labels = [event_label(family[i], event_ids[i]) for i in indices]
        records.append(dict(shot=r['shot'], group=r['group'], origins=r['origins'], query=r['query'],
                            label_time=r['label_time'], origin_times_s=[float(times[i]) for i in r['origins']],
                            query_time_s=float(times[r['query']]), label_time_s=float(times[r['label_time']]),
                            origin_events=labels[:4], query_event=labels[4],
                            **flags([label['event_id'] for label in labels])))
    groups = {g: summary([r for r in records if r['group'] == g]) for g in GROUPS}
    total = summary(records)
    require(sum(v['sequences'] for v in groups.values()) == total['sequences'] == len(history),
            'group counts do not reconcile')
    for key in (*flags([-1] * 5), 'query_unknown'):
        require(sum(v[key] for v in groups.values()) == total[key], f'{key} total mismatch')
    for family in FAMILIES:
        require(sum(v['query_family_counts'][family] for v in groups.values())
                == total['query_family_counts'][family], f'family {family} total mismatch')
    starts = [float(clocks[sid][0][0]) for sid in reads]
    result = dict(status='COMPLETE', scope='Post-hoc annotation coverage on CURRENT source HDF clocks only',
                  limitations=['Not proof raw input bytes are unchanged since fit',
                               'Families A-F are nominal annotations, not physical phase/danger labels',
                               'Unknown is -1, never normal; no risk or performance claim',
                               'CSV outcome metadata validated by original parser, not used in coverage or HDF5 outcomes'],
                  sources=dict(fit=str(FIT/'fit.json'), protocol=str(FIT/'protocol.json'),
                               event_csv=str(event_probe.DEFAULT_LABELS), event_csv_existing_hash_verified=True,
                               diagnostic_clock="simple_data.read_shot('J-TEXT', sid)['times']"),
                  access=dict(canonical_train_membership=True, source_signal_shots=sorted(reads),
                              source_val_test_signal_reads=0, east_signal_reads=0,
                              read_outcome_calls=outcome.call_count, attach_outcome_calls=attach.call_count,
                              clock_start_s_min=min(starts), clock_start_s_max=max(starts),
                              nonzero_clock_starts=sum(x != 0 for x in starts)),
                  total=total, groups=groups, records=records,
                  supersedes=str(FIT/'event_coverage.json'))
    previous = FIT/'event_coverage.json'
    if previous.is_file():
        old = json.loads(previous.read_text())
        require(old['groups'] == groups and old['total'] == total,
                'restoring original metadata validation changed coverage; investigate')
        result['coverage_unchanged_after_parser_restore'] = True
    serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with OUTPUT.open('x', encoding='utf-8') as handle:
        handle.write(serialized)
    print('synthetic event checks: PASS; group/query counts: PASS')
    print(f'sequences={len(records)} unique_train_signal_shots={len(reads)} '
          f'outcome_calls={outcome.call_count} attach_calls={attach.call_count}')
    print(json.dumps(groups, indent=2))
    print(f'wrote {OUTPUT}')


if __name__ == '__main__':
    main()
