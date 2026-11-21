"""Audit whether TRAIN weak intervals actually supervise event transitions."""
from collections import Counter
import json

import numpy as np

from .. import io23
from ..event_prefix_probe import DEFAULT_LABELS,LABEL_SHA256,read_events
from ..simple_train import sha
from ..simple_online import _durable_json


def main():
    if sha(DEFAULT_LABELS)!=LABEL_SHA256: raise AssertionError('event file changed')
    grouped,metadata=read_events(DEFAULT_LABELS)
    ids=io23.source_roles()['train']; hist=Counter(); families=Counter(); transitions=[]
    class_outcome={c:{'safe':0,'disruptive':0} for c in 'ABCDEF'}
    annotated=[]
    for sid in ids:
        events=grouped.get(sid,[]);hist[len(events)]+=1
        if not events: continue
        d,_=io23.read_outcome('J-TEXT',sid);annotated.append((sid,d))
        for e in events:
            family=chr(65+e.family);families[family]+=1
            class_outcome[family]['disruptive' if d else 'safe']+=1
        for a,b in zip(events,events[1:]):
            transitions.append(dict(shot=sid,is_disrupt=d,source=chr(65+a.family),
                target=chr(65+b.family),gap_ms=float(round((b.start-a.end)*1000,9))))
    pairs=Counter((r['source'],r['target']) for r in transitions)
    contiguous=[r for r in transitions if abs(r['gap_ms'])<=1e-6]
    different=[r for r in transitions if r['source']!=r['target']]
    result=dict(status='VERIFIED',partition='J-TEXT TRAIN1353 only',event_sha256=LABEL_SHA256,
        shots_by_number_of_events=dict(sorted(hist.items())),events_by_family=dict(sorted(families.items())),
        annotated_shots=len(annotated),annotated_disruptive=sum(d for _,d in annotated),
        annotated_safe=sum(not d for _,d in annotated),
        events_by_family_and_shot_outcome=class_outcome,
        consecutive_annotated_pairs=len(transitions),different_family_pairs=len(different),
        contiguous_pairs=len(contiguous),contiguous_different_family_pairs=sum(r['source']!=r['target'] for r in contiguous),
        pair_counts={f'{a}->{b}':n for (a,b),n in sorted(pairs.items())},
        gap_ms_quantiles=np.quantile([r['gap_ms'] for r in transitions],[0,.25,.5,.75,1]).tolist(),
        pairs=transitions,
        interpretation='A-F is weak categorical supervision; unknown gaps cannot be labeled recovery/normal, and same-family intervals do not establish a continuous cause chain.')
    output=io23.ROOT/'outputs/source_event_rebuild_audit_20261022/context_event_transition_readiness.json'
    if output.exists(): raise FileExistsError(output)
    _durable_json(output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'},ensure_ascii=False))


if __name__=='__main__':main()
