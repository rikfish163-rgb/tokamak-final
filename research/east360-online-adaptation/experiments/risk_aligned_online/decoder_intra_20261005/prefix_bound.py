"""Post-hoc irreversible-prefix alarm bound; never a target selection policy."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import simple_data
from ..representation_analysis import (_load_sealed_arm,_outcome_rows,_verify_summary,metrics_for_rows)
from ..simple_online import _durable_json,_hash


def count_prefix(rows,threshold,first_effective_index):
    if type(first_effective_index) is not int or first_effective_index<1 or not 0<=threshold<=1:
        raise ValueError('positive first effective endpoint index and valid threshold required')
    safe=[r for r in rows if not r['is_disrupt']]
    if not safe:raise ValueError('nonempty safe denominator required')
    shots=[];margins=[]
    for r in safe:
        scores=np.asarray(r['scores'],float)[:first_effective_index]
        if not len(scores) or not np.isfinite(scores).all():
            raise ValueError('nonempty finite sealed prefix required')
        margin=float(scores.max()-threshold)
        if margin>=0:
            shots.append(int(r['shot']));margins.append(margin)
    return dict(first_effective_index=first_effective_index,safe_count=len(safe),
        prefix_fp=len(shots),fpr_lower_bound=len(shots)/len(safe),shots=shots,
        minimum_positive_margin=min(margins) if margins else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--static-arm',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=json.loads((args.static_arm/'summary.json').read_text())
    traces=_load_sealed_arm(args.static_arm)
    rows=_outcome_rows(traces,simple_data.attach_outcome)
    threshold=result['metrics']['threshold'];measured=metrics_for_rows(rows,threshold)
    _verify_summary(result,result['metrics'],measured,len(traces))
    # Safe validity is all endpoints in the current io23.labels_at contract.
    # H25 issued at index0 matures at25; prediction25 is sealed first. The
    # earliest change is index26. These are SAMPLE indices, not universal ms.
    bound=count_prefix(rows,threshold,26)
    payload=dict(evidence='POSTHOC_STRUCTURAL_DIAGNOSTIC',threshold=threshold,roster_count=len(rows),
        summary_sha256=_hash(args.static_arm/'summary.json'),sealed_metrics_verified=True,
        current_H25_contract=bound,
        sensitivity=[count_prefix(rows,threshold,n) for n in (1,10,52,128)],
        assumptions=['same source classifier and input, unchanged threshold and any-crossing first alarm',
                     'no inter-shot parameter change; per-shot reset; no update before endpoint26',
                     'no retraction of an already emitted alarm; no alarm warm-up/veto'],
        interpretation='conditional FPR lower bound for unchanged pre-update outputs, not a bound on all architectures or intra+inter',
        available_additional_fp={str(cap):int(np.floor(cap*bound['safe_count']))-bound['prefix_fp'] for cap in (.20,.15,.10)},
        code_sha256=_hash(Path(__file__)))
    _durable_json(args.output,payload)
    print(json.dumps({k:v for k,v in payload.items() if k not in ('sensitivity','current_H25_contract')}))
    print(json.dumps({k:v for k,v in bound.items() if k!='shots'}))


if __name__=='__main__':main()
