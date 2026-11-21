"""Read frozen event catalogs and prepare the collaborator's flat label table."""
from pathlib import Path
from collections import Counter, defaultdict
import ast
import hashlib
import json

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
PAPER = OUT.parents[1]
LINE = PAPER.parents[1]
ROOT = next(p for p in LINE.parents if (p / 'WORKSPACE_GOVERNANCE.md').exists())
RERUN = LINE / 'FFTMIR_V2FFT_RERUN'
RUNS = RERUN / 'comparison_m2m4/runs'
sources = {}

def read(path):
    raw = path.read_bytes()
    sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

split = read(ROOT / 'mil_event_chain_mutitrain_v2fft_ap_serial/configs/sanity_split.json')
metadata = read(PAPER / 'analysis/chain_prefix_20260922/endpoint_metadata.json')['records']
short_doc = read(RERUN / 'event_execution/runs/single_ap_v1/event_table.json')
excluded = set(map(int, short_doc['per_round'][0]['excluded_short_dis']))
assert len(excluded) == 9
core_path = RERUN / 'event_execution/code/event_core.py'
core_text = core_path.read_text(encoding='utf-8')
sources[core_path.relative_to(ROOT).as_posix()] = hashlib.sha256(core_path.read_bytes()).hexdigest()
definitions = [n for n in ast.parse(core_text).body
               if isinstance(n, ast.FunctionDef) and n.name in ('premerge', 'finalize_families')]
assert len(definitions) == 2
namespace = {'defaultdict': defaultdict}
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(core_path), 'exec'), namespace)

header = ['shot_no','cohort','disruption_label','event_index','event_class',
          'event_start_s','event_end_s','annotation_status']
rows, trace, counts = [], [], {}
for cohort, split_key, run in [('training_pool','training_pool','single_ap'),
                                ('sanity','sanity_set','single_ap__heldout')]:
    ids = set(map(int, split[split_key]))
    folder = RUNS / run
    events = read(folder / 'events/event_table.json')['events']
    assignments = read(folder / 'vocab/assignments.json')
    families = {e['event_id']: e['family'] for e in assignments}
    assert len(families) == len(assignments) == len(events)
    main = namespace['finalize_families'](events, families)['main_events']
    metrics = read(folder / 'metrics/chain_metrics.json')
    assert len(main) == metrics['summary']['mainline_segments']['total']
    by_shot = defaultdict(list)
    for event in main:
        by_shot[event['shot_no']].append(event)
    assert len(by_shot) == metrics['summary']['event_bearing_shots']['total']
    # Match every frozen chain's type and start sequence, not just totals.
    for chain in metrics['chains']:
        actual = by_shot[chain['shot_no']]
        assert [e['family'] for e in actual] == chain['families']
        assert [e['start_token'] for e in actual] == chain['starts']
    if cohort == 'training_pool':
        frozen = read(RERUN / 'cluster_observation_v1/representation_v1/solutions/h_mean_own12/kmeans/k6/chains.json')
        expected = [(c['shot_no'], e['family'], e['start_token'], e['end_token'])
                    for c in frozen for e in c['segments']]
        actual = [(e['shot_no'],e['family'],e['start_token'],e['end_token']) for e in main]
        assert actual == expected
    eligible = set(int(s) for s in metadata if int(s) in ids)
    assert ids - eligible == (excluded if cohort == 'training_pool' else set())
    stat = Counter()
    for shot in sorted(ids):
        if shot in excluded:
            assert cohort == 'training_pool' and shot not in by_shot
            rows.append([shot,cohort,1,None,None,None,None,'excluded_short_record'])
            stat['excluded_shots'] += 1
            continue
        m = metadata[str(shot)]
        assert not m['issues'] and not m['missing_features']
        assert m['archived_label'] == m['is_disrupt']
        y = m['is_disrupt']
        if shot not in by_shot:
            rows.append([shot,cohort,y,None,None,None,None,'no_event_detected'])
            stat['no_event_shots'] += 1
            continue
        stat['event_bearing_shots'] += 1
        previous_end = -1
        for index, event in enumerate(by_shot[shot],1):
            start,end = event['start_token'],event['end_token']
            assert 0 <= start < end <= m['n_tokens_exact']
            assert start >= previous_end
            previous_end = end
            assert event['family'] in 'ABCDEF'
            t0 = m['common_start_time_s'] + start * 50 / m['common_sample_rate_hz']
            t1 = m['common_start_time_s'] + end * 50 / m['common_sample_rate_hz']
            assert t1 <= m['input_end_exclusive_s'] + 1e-10
            rows.append([shot,cohort,y,index,event['family'],round(t0,9),round(t1,9),'event'])
            trace.append(dict(shot_no=shot,cohort=cohort,event_index=index,
                              start_token=start,end_token=end,family=event['family'],
                              source_event_ids=list(dict.fromkeys(s['source_event_id'] for s in event['source_segments']))))
            stat['events'] += 1
    stat['registered_shots'] = len(ids)
    stat['eligible_shots'] = len(eligible)
    counts[cohort] = dict(stat)

assert len(rows)==1975 and len({r[0] for r in rows})==1816
assert sum(r[-1]=='event' for r in rows)==732
payload=dict(header=header,rows=rows,summary=counts)
(HERE/'labels_payload.json').write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
(HERE/'event_provenance.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')
(HERE/'source_manifest.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
(HERE/'validation.json').write_text(json.dumps(dict(summary=counts,row_count=len(rows),unique_shots=1816,
   source_checks='Frozen counts, all class/start sequences, full pool intervals, monotone nonoverlapping ranges, timing bounds, all registered shots',
   excluded_shots=sorted(excluded)),indent=2),encoding='utf-8')
print(json.dumps(counts,ensure_ascii=False))
