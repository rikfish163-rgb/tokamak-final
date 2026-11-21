"""Read-only annotation census; never supplies labels to a learner."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from .. import io23
from ..event_prefix_probe import read_events,DEFAULT_LABELS,LABEL_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from .run import REFERENCE


def main(output):
    if output.exists():raise FileExistsError(output)
    assert sha(DEFAULT_LABELS)==LABEL_SHA256
    events,metadata=read_events(DEFAULT_LABELS)
    mapping_path=REFERENCE/'input_map.json'
    selected=json.loads(mapping_path.read_text())['calibration_ids']
    provenance=io23.ROOT/'experiments/risk_aligned_online/runs/feedback_clock20_test360_20261023/alignment_provenance.json'
    lost=json.loads(provenance.read_text())['lost_timely']
    assert len(selected)==20 and len(lost)==29 and not set(selected)&set(lost)
    groups=dict(selected20=selected,lost29=lost)
    roles=io23.source_roles();assigned=set().union(*(set(v) for v in roles.values()))
    source={}
    for role,ids in dict(roles,outside=sorted(set(metadata)-assigned)).items():
        source[role]=dict(metadata_shots=len(set(ids)&set(metadata)),event_shots=len(set(ids)&set(events)),
                          intervals=sum(len(events.get(s,[])) for s in ids))
    supplement=dict(path=str(DEFAULT_LABELS),sha256=LABEL_SHA256,total_shots=len(metadata),
        event_shots=len(events),intervals=sum(map(len,events.values())),source_roles=source,
        groups={g:dict(metadata_overlap=sorted(set(ids)&set(metadata)),
                       event_overlap=sorted(set(ids)&set(events))) for g,ids in groups.items()})
    legacy={}
    for name in ('east_shot_onset_labels.csv','east_shot_onset_labels_advtime.csv'):
        path=io23.ROOT/name
        with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        info=dict(path=str(path),sha256=sha(path),total_rows=len(rows),
            total_shots=len({int(r['shot_no']) for r in rows}),groups={})
        for group,ids in groups.items():
            matched=[r for r in rows if int(r['shot_no']) in ids];found={int(r['shot_no']) for r in matched}
            info['groups'][group]=dict(shots=len(found),rows=len(matched),missing=sorted(set(ids)-found),
                split_counts=dict(Counter(r['split'] for r in matched)),
                value_counts=dict(Counter(r['adaptive_onset_ms'] for r in matched)),
                confidence_counts=dict(Counter(r['confidence'] for r in matched)),records=matched)
        info['distinct_event_templates']=sorted({(r['event'],r['pathway'],r['adaptive_onset_ms'],r['severity']) for r in rows})
        legacy[name]=info
    out=dict(scope='POSTHOC ANNOTATION INVENTORY, NO TRAINING',selected20=selected,lost29=lost,
        input_map_sha256=sha(mapping_path),provenance_sha256=sha(provenance),code_sha256=sha(Path(__file__)),
        supplement=supplement,legacy=legacy,
        interpretation='supplement is JTEXT offline discovered events; missing means unknown; legacy EAST templates are not event truth')
    _durable_json(output,out)
    print(json.dumps(dict(source_roles=source,supplement_groups=supplement['groups'],legacy={
        n:{g:{k:v for k,v in r.items() if k!='records'} for g,r in info['groups'].items()} for n,info in legacy.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();main(a.output)
