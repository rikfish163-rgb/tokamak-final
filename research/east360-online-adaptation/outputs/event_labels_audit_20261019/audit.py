"""Read-only audit of supplied offline event labels against local J-TEXT HDF5.

Writes derived tables and descriptive statistics only in this analysis folder.
Does not execute supplied code, train a predictor, or alter labels/checkpoints.
"""
from pathlib import Path
from collections import Counter
from decimal import Decimal
import csv
import hashlib
import json

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT = HERE / 'input/event_labels_20261018'


def quantiles(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'n': 0}
    return dict(n=int(len(a)), **{k: float(v) for k, v in zip(
        ['min', 'q25', 'median', 'q75', 'max'], np.quantile(a, [0, .25, .5, .75, 1]))})


def records(df):
    return json.loads(df.to_json(orient='records'))


def run():
    path = INPUT / 'JTEXT_event_labels.csv'
    df = pd.read_csv(path, encoding='utf-8-sig')
    shots = df.drop_duplicates('shot_no').copy()
    events = df.loc[df.annotation_status.eq('event')].copy()
    events['duration_ms'] = (events.event_end_s - events.event_start_s) * 1000
    checks = {}
    checks['duplicate_rows'] = int(df.duplicated().sum())
    checks['duplicate_event_keys'] = int(events.duplicated(['shot_no', 'event_index']).sum())
    checks['inconsistent_shot_labels'] = int((df.groupby('shot_no').disruption_label.nunique() > 1).sum())
    checks['inconsistent_shot_cohorts'] = int((df.groupby('shot_no').cohort.nunique() > 1).sum())
    checks['invalid_intervals'] = int((events.event_end_s <= events.event_start_s).sum())
    checks['invalid_classes'] = int((~events.event_class.isin(list('ABCDEF'))).sum())
    blank_fields = ['event_index', 'event_class', 'event_start_s', 'event_end_s']
    checks['populated_non_event_fields'] = int(df.loc[~df.annotation_status.eq('event'), blank_fields].notna().sum().sum())
    overlaps = []
    for sid, g in events.groupby('shot_no'):
        g = g.sort_values('event_index')
        if (np.diff(g.event_index.to_numpy()) != 1).any() or g.event_index.iloc[0] != 1:
            overlaps.append({'shot': int(sid), 'issue': 'event_index'})
        if len(g) > 1 and (g.event_start_s.to_numpy()[1:] < g.event_end_s.to_numpy()[:-1] - 1e-10).any():
            overlaps.append({'shot': int(sid), 'issue': 'overlap'})
    checks['ordering_or_overlap_errors'] = overlaps
    provenance = json.loads((INPUT / 'work/event_provenance.json').read_text())
    trace = {(int(r['shot_no']), int(r['event_index'])): r for r in provenance}
    timing_errors = []
    with path.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r['annotation_status'] != 'event':
                continue
            t = trace[(int(r['shot_no']), int(r['event_index']))]
            for field, token in [('event_start_s', 'start_token'), ('event_end_s', 'end_token')]:
                if Decimal(r[field]) != Decimal('0.15') + Decimal(t[token]) * Decimal('0.005'):
                    timing_errors.append([r['shot_no'], field])
            if r['event_class'] != t['family']:
                timing_errors.append([r['shot_no'], 'family'])
    checks['provenance_mismatches'] = timing_errors
    checks['provenance_unique_keys'] = len(trace)
    payload = json.loads((INPUT / 'work/labels_payload.json').read_text())
    p = pd.DataFrame(payload['rows'], columns=payload['header'])
    checks['csv_payload_equal'] = bool(df.fillna('<missing>').equals(p.fillna('<missing>')))
    # Compare serialized values as types inferred by pandas can differ on blanks.
    checks['csv_payload_values_equal'] = bool((df.fillna('<missing>').to_numpy() == p.fillna('<missing>').to_numpy()).all())

    roles = json.loads((ROOT / 'artifacts/canonical/jtext_source_bundle_p131_v1/index.json').read_text())['split_shot_ids']
    role_by_shot = {int(s): role for role, ids in roles.items() for s in ids}
    advanced = json.loads((ROOT / 'shot_list/J-TEXT/AdvancedTime_J-TEXT.json').read_text())
    h5_index = {int(p.stem): p for p in (ROOT / 'data/J-TEXT/unified_hdf5').glob('*/*.hdf5') if p.stem.isdigit()}
    linked = []
    for r in shots.itertuples():
        sid = int(r.shot_no)
        item = dict(shot_no=sid, cohort=r.cohort, csv_label=int(r.disruption_label),
                    status=r.annotation_status, canonical_role=role_by_shot.get(sid, 'outside_canonical'),
                    hdf5_found=sid in h5_index)
        if sid in h5_index:
            with h5py.File(h5_index[sid], 'r') as f:
                y = int(np.asarray(f['meta/IsDisrupt'][()]).reshape(-1)[0])
                down_array = np.asarray(f['meta/DownTime'][()]).reshape(-1)
                down = float(down_array[0]) if len(down_array) else np.nan
                ds = f['data/ip']
                start = float(ds.attrs['StartTime'])
                rate = float(ds.attrs['SampleRate'])
                n = len(ds)
                item.update(hdf5_label=y, label_agrees=y == r.disruption_label,
                            down_s=down, record_start_s=start, sample_rate_hz=rate,
                            sample_count=n, record_end_exclusive_s=start+n/rate,
                            horizon_ms=max(25., float(advanced.get(str(sid), 25.))))
        linked.append(item)
    link = pd.DataFrame(linked)
    link.to_csv(HERE / 'shot_linkage.csv', index=False)
    e = events.merge(link.drop(columns=['cohort']), on='shot_no', validate='many_to_one')
    e['start_lead_ms'] = np.where(e.hdf5_label.eq(1), (e.down_s-e.event_start_s)*1000, np.nan)
    e['end_lead_ms'] = np.where(e.hdf5_label.eq(1), (e.down_s-e.event_end_s)*1000, np.nan)
    e['outside_hdf5_ms'] = (np.maximum(e.record_start_s-e.event_start_s, 0) +
                               np.maximum(e.event_end_s-e.record_end_exclusive_s, 0))*1000
    e['beyond_down_ms'] = np.where(e.hdf5_label.eq(1), np.maximum(e.event_end_s-e.down_s, 0)*1000, 0)
    e['source_positive_overlap_ms'] = np.where(e.hdf5_label.eq(1), np.maximum(0,
        np.minimum(e.event_end_s, e.down_s)-np.maximum(e.event_start_s, e.down_s-e.horizon_ms/1000))*1000, 0)
    e['outside_source_positive_ms'] = e.duration_ms-e.source_positive_overlap_ms
    e.to_csv(HERE / 'events_joined.csv', index=False)
    # Retrospective description in the EXISTING 23-channel coordinate system.
    # This is not a reproduction of the external event generator's extraction.
    # Whole-shot medians/IQRs are permissible for this offline audit only.
    profile_rows = []
    for sid, g in e[e.hdf5_found].groupby('shot_no'):
        with h5py.File(h5_index[int(sid)], 'r') as f:
            first_ds = f['data/ip']
            times = float(first_ds.attrs['StartTime']) + np.arange(len(first_ds))/float(first_ds.attrs['SampleRate'])
            for name, ds in f['data'].items():
                values = np.asarray(ds[:], float)
                finite = np.isfinite(values)
                if len(values) != len(times) or not finite.any():
                    continue
                q25, center, q75 = np.quantile(values[finite], [.25, .5, .75])
                scale = q75-q25
                if scale <= np.finfo(float).eps*max(abs(center), 1):
                    continue
                for r in g.itertuples():
                    selected = finite & (times >= r.event_start_s-1e-10) & (times < r.event_end_s-1e-10)
                    before = finite & (times >= r.event_start_s-.05-1e-10) & (times < r.event_start_s-1e-10)
                    if not selected.any():
                        continue
                    event_median = np.median(values[selected])
                    profile_rows.append(dict(shot_no=int(sid), event_index=int(r.event_index),
                        event_class=r.event_class, outcome=int(r.disruption_label), feature=name,
                        relative_to_whole_shot=(event_median-center)/scale,
                        relative_to_previous_50ms=(event_median-np.median(values[before]))/scale if before.sum() >= 10 else np.nan,
                        preceding_points=int(before.sum())))
    profiles = pd.DataFrame(profile_rows)
    profiles.to_csv(HERE / 'event_signal_profiles.csv', index=False)
    by_shot = profiles.groupby(['event_class','shot_no','feature'])[
        ['relative_to_whole_shot','relative_to_previous_50ms']].mean()
    profile_summary = by_shot.groupby(['event_class','feature']).median().reset_index()
    profile_summary.to_csv(HERE / 'family_signal_profile_medians.csv', index=False)
    families = []
    for fam, g in e.groupby('event_class'):
        u = g.drop_duplicates('shot_no')
        disruptive = g.loc[g.disruption_label.eq(1)]
        families.append(dict(family=fam, events=len(g), shots=len(u),
                             disruptive_shots=int(u.disruption_label.sum()),
                             safe_shots=int((u.disruption_label == 0).sum()),
                             pool_events=int((g.cohort == 'training_pool').sum()),
                             sanity_events=int((g.cohort == 'sanity').sum()),
                             duration_median_ms=float(g.duration_ms.median()),
                             start_lead_median_ms=float(disruptive.start_lead_ms.median()),
                             end_lead_median_ms=float(disruptive.end_lead_ms.median()),
                             shot_min=int(g.shot_no.min()), shot_max=int(g.shot_no.max())))
    family = pd.DataFrame(families)
    family.to_csv(HERE / 'family_summary.csv', index=False)

    chains, transitions = [], []
    for sid, g in e.groupby('shot_no'):
        g = g.sort_values('event_index')
        families_seq = list(g.event_class)
        chains.append(dict(shot_no=int(sid), label=int(g.disruption_label.iloc[0]),
                           cohort=g.cohort.iloc[0], sequence='>'.join(families_seq),
                           event_count=len(g), distinct_classes=len(set(families_seq))))
        for a, b in zip(g.to_dict('records'), g.to_dict('records')[1:]):
            transitions.append(dict(shot_no=int(sid), label=int(a['disruption_label']),
                                    cohort=a['cohort'], source=a['event_class'], target=b['event_class'],
                                    gap_ms=(b['event_start_s']-a['event_end_s'])*1000))
    chain = pd.DataFrame(chains)
    trans = pd.DataFrame(transitions)
    chain.to_csv(HERE / 'shot_sequences.csv', index=False)
    trans.to_csv(HERE / 'transitions.csv', index=False)
    seq_count = chain.groupby(['sequence', 'label']).size().unstack(fill_value=0)
    seq_count.to_csv(HERE / 'sequence_counts.csv')

    # One fixed diagnostic: can shot number alone predict event class on sanity?
    # No diagnostic signals or outcome enter this shallow tree. It is NOT a
    # disruption model; high accuracy exposes possible campaign confounding.
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
    train = e.cohort.eq('training_pool')
    test = e.cohort.eq('sanity')
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=20261019)
    weights = 1/e.loc[train].groupby('shot_no').shot_no.transform('size')
    tree.fit(e.loc[train, ['shot_no']], e.loc[train, 'event_class'], sample_weight=weights)
    pred = tree.predict(e.loc[test, ['shot_no']])
    shot_only = dict(description='Depth-3 tree; train on training_pool, test on sanity; sole input=shot number. Descriptive confounding check, not prediction evidence.',
                     sanity_event_accuracy=float(accuracy_score(e.loc[test, 'event_class'], pred)),
                     sanity_balanced_accuracy=float(balanced_accuracy_score(e.loc[test, 'event_class'], pred)),
                     majority_accuracy=float((e.loc[test, 'event_class'] == e.loc[train, 'event_class'].mode().iloc[0]).mean()),
                     confusion_matrix=confusion_matrix(e.loc[test, 'event_class'], pred, labels=list('ABCDEF')).tolist(),
                     thresholds=[float(x) for x in tree.tree_.threshold if x != -2])
    pd.DataFrame({'shot_no':e.loc[test, 'shot_no'], 'true_family':e.loc[test, 'event_class'],
                  'shot_number_only_prediction':pred}).to_csv(HERE / 'shot_number_only_diagnostic.csv', index=False)

    test_source = json.loads((ROOT / 'experiments/risk_aligned_online/runs/gradient_aligned512_20260929/test220.json').read_text())['jtext']
    predictions = pd.DataFrame(test_source['predictions']).rename(columns={'shot':'shot_no'})
    predictions['category'] = np.select([predictions.is_disrupt & predictions.timely,
                                         predictions.is_disrupt & ~predictions.timely,
                                         ~predictions.is_disrupt & predictions.alarm], ['TP','FN_or_late','FP'], default='TN')
    predictions = predictions.merge(link[['shot_no', 'cohort', 'status']], on='shot_no', how='left', validate='one_to_one')
    family_per_shot = e.groupby('shot_no').event_class.apply(lambda x: '|'.join(sorted(set(x))))
    predictions['families'] = predictions.shot_no.map(family_per_shot)
    predictions.to_csv(HERE / 'source_test220_errors_joined.csv', index=False)

    good = link[link.hdf5_found]
    d_events = e[e.disruption_label.eq(1) & e.hdf5_found]
    first = d_events.sort_values('event_index').drop_duplicates('shot_no')
    summary = dict(
        archive_sha256=hashlib.sha256(Path('/home/hetaisheng/下载/event_labels_20261018.rar').read_bytes()).hexdigest(),
        csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        rows=len(df), shots=len(shots), events=len(events), event_bearing_shots=events.shot_no.nunique(),
        labels=shots.disruption_label.value_counts().sort_index().to_dict(),
        cohort_by_label=records(shots.groupby(['cohort','disruption_label']).size().rename('shots').reset_index()),
        status_by_label=records(shots.groupby(['annotation_status','disruption_label']).size().rename('shots').reset_index()),
        validation=checks,
        hdf5_link=dict(found=int(link.hdf5_found.sum()), missing=link.loc[~link.hdf5_found,'shot_no'].tolist(),
                       outcome_mismatches=records(good[~good.label_agrees.astype(bool)]),
                       source_clock_start=quantiles(good.record_start_s),
                       source_rate_hz=quantiles(good.sample_rate_hz),
                       events_outside_record=records(e[e.outside_hdf5_ms > 1e-6][['shot_no','event_index','outside_hdf5_ms','event_start_s','event_end_s','record_end_exclusive_s']]),
                       events_beyond_disruption=records(e[e.beyond_down_ms > 1e-6][['shot_no','event_index','beyond_down_ms']]),
                       local_hdf5_count=len(h5_index), local_without_labels=len(set(h5_index)-set(shots.shot_no))),
        split_overlap=records(link.groupby(['canonical_role','cohort']).size().rename('shots').reset_index()),
        canonical_without_labels={k: sorted(set(v)-set(shots.shot_no)) for k,v in roles.items()},
        families=records(family),
        linked_families=records(e[e.hdf5_found].groupby(['event_class','disruption_label']).agg(
            events=('shot_no','size'), shots=('shot_no','nunique')).reset_index()),
        unlinked_families=records(e[~e.hdf5_found].groupby('event_class').agg(
            events=('shot_no','size'), shots=('shot_no','nunique')).reset_index()),
        timing=dict(duration_ms=quantiles(e.duration_ms), first_disruptive_event_start_lead_ms=quantiles(first.start_lead_ms),
                    scope='Lead times and label-window comparisons use only matched local disruptive HDF5 records.',
                    disruptive_event_starts_before_source_positive=int((d_events.start_lead_ms > d_events.horizon_ms+1e-6).sum()),
                    disruptive_event_total=len(d_events),
                    event_duration_fraction_outside_source_positive=float(d_events.outside_source_positive_ms.sum()/d_events.duration_ms.sum()),
                    disruptive_event_duration_ms=float(d_events.duration_ms.sum()),
                    source_positive_overlap_ms=float(d_events.source_positive_overlap_ms.sum()),
                    starts_at_165ms=int(np.isclose(e.event_start_s, .165).sum()),
                    disruptive_event_end_within_5ms_of_down=int(d_events.end_lead_ms.between(0,5).sum()),
                    ends_within_5ms_of_record_end=int(((e.record_end_exclusive_s-e.event_end_s)*1000).between(-1e-6,5+1e-6).sum()),
                    first_event_lead_gt30_shots=int((first.start_lead_ms > 30).sum()),
                    first_event_lead_gt10_shots=int((first.start_lead_ms > 10).sum())),
        sequences=dict(event_count_distribution=chain.event_count.value_counts().sort_index().to_dict(),
                       shots_with_multiple_classes=int((chain.distinct_classes > 1).sum()),
                       distinct_sequences=int(chain.sequence.nunique()),
                       transitions=len(trans), same_class_transitions=int(trans.source.eq(trans.target).sum()),
                       cross_class_transitions=int(trans.source.ne(trans.target).sum()),
                       transition_counts=records(trans.groupby(['source','target','label']).size().rename('count').reset_index())),
        shot_number_only=shot_only,
        source_test_errors=records(predictions.groupby(['category','status'], dropna=False).size().rename('shots').reset_index()),
        source_test_error_cases=records(predictions[predictions.category.isin(['FP','FN_or_late'])]),
    )
    (HERE / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['source_test_error_cases','validation','hdf5_link','sequences']}, ensure_ascii=False, indent=2))
    print('HDF5:', json.dumps(summary['hdf5_link'], ensure_ascii=False))
    print('VALIDATION:', json.dumps(checks, ensure_ascii=False))
    print('SEQUENCES:', json.dumps(summary['sequences'], ensure_ascii=False))


if __name__ == '__main__':
    run()
