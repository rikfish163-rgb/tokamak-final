"""Descriptive diagnostics after all outcomes are revealed; never select a model."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..simple_online import _durable_json
from ..simple_train import sha
from .run import REFERENCE,ARMS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    root=p.parse_args().directory
    verified=json.loads((root/'verification.json').read_text());assert verified['status']=='VERIFIED'
    full=json.loads((root/'per_shot.json').read_text());fit=json.loads((root/'alignment.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    previous=json.loads((REFERENCE/'per_shot.json').read_text())
    parent=protocol['parent_threshold'];timing={}
    for arm in (*ARMS,'raw_reference','density_gain_n20_reference'):
        own=arm in ARMS;key=arm if own else ('static' if arm=='raw_reference' else 'density_gain_n20')
        records=full[arm]['predictions'] if own else previous[key]['predictions']
        directory=root if own else REFERENCE
        offsets=[];cases=[]
        for r in records:
            if r['is_disrupt'] or not r['alarm']:continue
            with np.load(directory/'shots'/f"{r['shot']}.npz",allow_pickle=False) as trace:
                th=r['threshold'] if own else parent
                first=int(np.flatnonzero(trace[key]>=th)[0])
                elapsed=float((trace['times'][first]-trace['times'][0])*1000.)
                offsets.append(elapsed);cases.append(dict(shot=r['shot'],sample=first,elapsed_from_record_start_ms=elapsed))
        timing[arm]=dict(fp_count=len(offsets),
            first_128ms=sum(x<128 for x in offsets),first_500ms=sum(x<500 for x in offsets),
            first_1000ms=sum(x<1000 for x in offsets),median_elapsed_ms=float(np.median(offsets)),cases=cases)
    names=fit['channels'];cs=np.array(fit['source_cov']);ct=np.array(fit['target_cov']);a=np.array(fit['matrix'])
    pairs=sorted(((abs(cs[i,j]-ct[i,j]),i,j) for i in range(23) for j in range(i)),reverse=True)[:10]
    shifts=[dict(first=names[i],second=names[j],source=float(cs[i,j]),target=float(ct[i,j]),absolute_change=float(d)) for d,i,j in pairs]
    mixed=sorted(((abs(a[i,j]),i,j) for i in range(23) for j in range(23) if i!=j),reverse=True)[:10]
    moments=[dict(channel=n,source_mean=fit['source']['mean'][i],source_std=fit['source']['scale'][i],
                  target_mean=fit['target']['mean'][i],target_std=fit['target']['scale'][i],active=fit['active'][i]) for i,n in enumerate(names)]
    result=dict(scope='POSTHOC DESCRIPTIVE; no thresholds or weights written back',
        verification_sha256=sha(root/'verification.json'),elapsed_origin='stored HDF5 record start, NOT plasma discharge start',
        false_alarm_timing=timing,largest_standardized_covariance_changes=shifts,
        largest_source_coordinate_mixtures=[dict(input=names[i],output=names[j],coefficient=float(a[i,j])) for _,i,j in mixed],
        raw_file_moments_not_certified_units=moments,
        source_mean_rows_per_record=fit['source']['rows']/len(fit['source_ids']),
        target_mean_rows_per_record=fit['target']['rows']/len(fit['target_ids']),
        diagnostic_code_sha256=sha(Path(__file__)))
    _durable_json(root/'diagnostic.json',result)
    print(json.dumps({k:{a:b for a,b in v.items() if a!='cases'} for k,v in timing.items()}),flush=True)


if __name__=='__main__':main()
