"""Read-only retrospective error provenance; no fitted policy or new prediction."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..simple_online import _durable_json
from ..simple_train import sha
from ..target_feedback_meta_20261023.audit import recount


def category(row):
    if row['is_disrupt']:
        return 'timely' if row['timely'] else 'late' if row['alarm'] else 'never'
    return 'fp' if row['alarm'] else 'tn'


def main(reference,output):
    if output.exists():raise FileExistsError(output)
    read=lambda p:json.loads(p.read_text())
    verify=read(reference/'verification.json');complete=read(reference/'completed.json')
    assert verify['status']=='VERIFIED'
    # Verification schema is specific to this completed alignment experiment.
    assert sha(reference/'completed.json')==verify['completed_sha256']
    full=read(reference/'per_shot.json');p=read(reference/'protocol.json')
    raw={r['shot']:r for r in full['raw_source']['predictions']}
    mapped={r['shot']:r for r in full['input_only_static']['predictions']}
    seals={r['shot']:r for r in read(reference/'prediction_receipts.json')}
    assert set(raw)==set(mapped)==set(p['contract']['shot_ids'])
    transitions={};rows=[];channel=io23.RAW23.index('ne_nG')
    for sid in p['contract']['shot_ids']:
        path=reference/'shots'/f'{sid}.npz';assert sha(path)==seals[sid]['prediction_sha256']
        d,down=io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as z:
            for key,saved in (('raw_source',raw[sid]),('input_only_static',mapped[sid])):
                assert recount(sid,z['times'],z[key],d,down,p['threshold'])==saved
        a,b=category(raw[sid]),category(mapped[sid]);transitions.setdefault(a+'->'+b,[]).append(sid)
        s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        # Causal first128 observations, not full-shot median or outcome crop.
        take=s['available'][:128,channel];values=s['values'][:128,channel][take]
        value=float(np.median(values)) if len(values) else None
        rows.append(dict(shot=sid,is_disrupt=d,raw_category=a,mapped_category=b,
            prefix_count=min(128,len(s['values'])),density_observations=int(take.sum()),
            stored_density_prefix_median=value,raw_lead_ms=raw[sid]['lead_ms'],mapped_lead_ms=mapped[sid]['lead_ms']))
    for a in ('raw_source','input_only_static'):
        rs=raw.values() if a=='raw_source' else mapped.values();rs=list(rs);m=complete['metrics'][a]
        assert sum(r['timely'] for r in rs)==m['tp']
        assert sum(not r['is_disrupt'] and r['alarm'] for r in rs)==m['fp']
    groups={}
    for label in ('timely','late','never','fp','tn'):
        rs=[r for r in rows if r['mapped_category']==label]
        values=[r['stored_density_prefix_median'] for r in rs if r['stored_density_prefix_median'] is not None]
        groups[label]=dict(shots=len(rs),density_available=len(values),
            density_prefix_quantiles=dict(zip(('p10','median','p90'),map(float,np.quantile(values,[.1,.5,.9])))) if values else None)
    lost=[r['shot'] for r in rows if r['raw_category']=='timely' and r['mapped_category']!='timely']
    gained=[r['shot'] for r in rows if r['raw_category']!='timely' and r['mapped_category']=='timely']
    assert len(gained)-len(lost)==complete['metrics']['input_only_static']['tp']-complete['metrics']['raw_source']['tp']
    result=dict(scope='POSTHOC error provenance and descriptive prefix statistics; no policy learned, no physical unit certification',
        completed_sha256=sha(reference/'completed.json'),verification_sha256=sha(reference/'verification.json'),
        code_sha256=sha(Path(__file__)),compared_traces=len(rows)*2,
        transition_counts={k:len(v) for k,v in transitions.items()},transition_shots=transitions,
        lost_timely=lost,gained_timely=gained,prefix_groups=groups,rows=rows,
        predictive_claim=False,causal_mechanism_identified=False,models_predictions_modified=False)
    _durable_json(output,result)
    print(json.dumps({k:result[k] for k in ('transition_counts','lost_timely','gained_timely','prefix_groups')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('reference',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();main(args.reference,args.output)
