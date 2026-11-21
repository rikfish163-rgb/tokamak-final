"""Fast POST-HOC capped-FPR audit; not an online threshold selection policy."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import representation_analysis as analysis


def peak_caps(rows,caps=(.10,.15,.20)):
    positive=[];negative=[]
    for row in rows:
        times,scores=analysis._row_arrays(row)
        if row['is_disrupt']:
            scores=scores[np.round((row['down']-times)*1000.,9)>30.]
            positive.append(float(scores.max()) if len(scores) else -float('inf'))
        else:
            negative.append(float(scores.max()) if len(scores) else -float('inf'))
    if not positive or not negative:
        raise ValueError('both shot classes required for capped-FPR diagnostic')
    positive=np.sort(positive);negative=np.sort(negative)
    candidates=np.unique(np.r_[positive,negative,np.nextafter(1.,2.)])
    candidates=candidates[np.isfinite(candidates)]
    tp=len(positive)-np.searchsorted(positive,candidates,side='left')
    fp=len(negative)-np.searchsorted(negative,candidates,side='left')
    result={}
    for cap in caps:
        if not np.isfinite(cap) or not 0<=cap<=1:
            raise ValueError('FPR cap must be in [0,1]')
        feasible=np.flatnonzero(fp/len(negative)<=cap)
        index=max(feasible,key=lambda i:(int(tp[i]),-int(fp[i]),float(candidates[i])))
        tpr=float(tp[index]/len(positive));fpr=float(fp[index]/len(negative))
        result[f'{cap:.2f}']=dict(threshold=float(candidates[index]),tp=int(tp[index]),fp=int(fp[index]),
            n_disrupt=len(positive),n_safe=len(negative),tpr=tpr,fpr=fpr,J=tpr-fpr,fpr_cap=float(cap))
    return dict(kind='TEST_DRIVEN_ORACLE_DIAGNOSTIC',not_online_predeployed=True,caps=result,
        interpretation='uses all revealed labels; no deployed threshold or independent validation claim')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paired',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    paired=json.loads(args.paired.read_text())
    if not paired['verification']['full2953_verified'] or not paired['verification']['summary_metrics_verified']:
        raise ValueError('completed and verified paired full2953 report required')
    results={}
    for arm,path in paired['runs'].items():
        directory=Path(path)
        rows=analysis._outcome_rows(analysis._load_sealed_arm(directory),analysis.simple_data.attach_outcome)
        if len(rows)!=2953:raise ValueError('full roster required')
        summary=json.loads((directory/'summary.json').read_text())
        metrics=analysis.metrics_for_rows(rows,paired['threshold'])
        analysis._verify_summary(summary,summary['metrics'],metrics,2953)
        results[arm]=peak_caps(rows)
        print(json.dumps(dict(arm=arm,**results[arm])),flush=True)
        del rows
    analysis._write_json_once(args.output,dict(paired_sha256=analysis._sha256(args.paired),
        sealed_metrics_verified=True,source_sha256=paired['source_sha256'],results=results,
        interpretation='strict >30ms; fixed emitted curves only; post-hoc oracle, not online gains at a prechosen cap'))


if __name__=='__main__':main()
