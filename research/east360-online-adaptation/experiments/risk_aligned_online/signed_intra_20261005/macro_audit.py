"""Post-run audit of the bounded macro experiments; never trains a new rule."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import simple_data
from ..causal_input import apply_input_contract
from ..simple_online import _durable_json, _hash
from .finite_probe import whole_cost
from .macro_credit import fit_rule
from .macro_experiment import statistics


def paired_prefix(h, p, qh, qp, sh, sp, threshold):
    """H/P may differ only after the second decision; no early-alarm refund."""
    if h['rule_file_sha256'] != p['rule_file_sha256'] or h['history_end'] != p['history_end']:
        raise ValueError('H/P rule history differs')
    dh, dp = h['decisions'], p['decisions']
    if len(dh) != len(dp):
        raise ValueError('H/P decision clocks differ')
    if dh and (dh[0]['action'] != dp[0]['action'] or dh[0]['end'] != dp[0]['end']):
        raise ValueError('H/P first decision differs')
    effect = dh[1]['effect'] if len(dh)>1 else None
    if len(dh)>1 and (dh[1]['end'] != dp[1]['end'] or effect != dp[1]['effect']):
        raise ValueError('H/P second decision clocks differ')
    end = len(qh) if effect is None else effect
    error = float(np.max(np.abs(sh[:end]-sp[:end]))) if end else 0.
    if (not np.array_equal(qh[:end],qp[:end]) or error>3e-6
            or not np.array_equal(sh[:end]>=threshold,sp[:end]>=threshold)):
        raise ValueError('H/P changed immutable common prefix')
    changed = effect is not None and not np.array_equal(qh[effect:],qp[effect:])
    return dict(effect_2=effect, prefix_max_abs=error, later_q_differs=changed,
                h_actions=[d['action'] for d in dh],p_actions=[d['action'] for d in dp])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--guarded',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(4)
    tables=[json.loads(f.read_text()) for f in sorted((args.original/'tables').glob('*.json'))]
    if len(tables)!=32: raise ValueError('exact first32 historical tables required')
    fitted,logs=fit_rule(tables,torch.zeros(3,117,dtype=torch.float64))
    saved=torch.load(args.original/'history_rule.pt',map_location='cpu',weights_only=True)
    if not torch.equal(fitted,saved['rule']): raise ValueError('original three-round fit no longer reproducible')
    summary=json.loads((args.guarded/'summary.json').read_text())
    protocol=summary['protocol']; threshold=protocol['threshold']; records=summary['future_records']
    if not protocol['historical_guard'] or not summary['future_run']:
        raise ValueError('completed guarded experiment required')
    rows=[]; verified=0; witnesses=[]
    for offset,sid in enumerate(protocol['future_ids']):
        index=offset+33; stem=f'{index:04d}_{sid}'
        signals=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        shot=simple_data.attach_outcome(signals,'EAST',{})  # post-run audit only
        traces={}; events={}; qs={}; predictions={}
        expected_rule=(args.guarded/'history_rule.pt' if index==33 else args.guarded/f'rule_after_{index-1}.pt')
        for arm in ('S','H','P','G'):
            folder=args.guarded/arm
            receipt=json.loads((folder/f'{stem}_seal.json').read_text())
            if receipt['threshold']!=threshold or _hash(folder/receipt['npz'])!=receipt['sha256']:
                raise ValueError('future seal changed')
            with np.load(folder/receipt['npz'],allow_pickle=False) as data:
                if not np.array_equal(data['times'],shot['times']): raise ValueError('time grid mismatch')
                traces[arm]=data['scores'].copy()
            events[arm]=json.loads((folder/f'{stem}_events.json').read_text())
            if events[arm]['history_end']!=index-1 or events[arm]['rule_file_sha256']!=_hash(expected_rule):
                raise ValueError('future rule version mismatch')
            with np.load(folder/f'{stem}_q.npz',allow_pickle=False) as data: qs[arm]=data['q'].copy()
            _,_,predictions[arm]=whole_cost(shot,traces[arm],threshold)
            if predictions[arm]!=records[arm][offset]['prediction']:
                raise ValueError('summary does not match sealed scores')
            verified+=1
        prefix=paired_prefix(events['H'],events['P'],qs['H'],qs['P'],traces['H'],traces['P'],threshold)
        valid=shot['valid'].copy()
        if shot['is_disrupt']: valid &= shot['times']<=shot['down']
        hits=np.flatnonzero(valid & (traces['H']>=threshold))
        first=int(hits[0]) if len(hits) else None
        witness=bool(shot['is_disrupt'] and predictions['H']['timely']
            and not predictions['S']['timely'] and not predictions['P']['timely']
            and prefix['later_q_differs'] and first>=prefix['effect_2'])
        if witness: witnesses.append(sid)
        rows.append(dict(index=index,shot=sid,disruptive=bool(shot['is_disrupt']),
            **prefix,first_alarm_h_index=first,new_danger_witness=witness,predictions=predictions))
    metrics={arm:statistics(records[arm]) for arm in records}
    if metrics!=summary['future']: raise ValueError('future metric mismatch')
    paired={}
    for arm in ('S','P','G'):
        change={name:[] for name in ('timely_gained','timely_lost','fp_removed','fp_added')}
        for r in rows:
            h,b=r['predictions']['H'],r['predictions'][arm]
            if r['disruptive']:
                if h['timely'] and not b['timely']: change['timely_gained'].append(r['shot'])
                if b['timely'] and not h['timely']: change['timely_lost'].append(r['shot'])
            else:
                if b['alarm'] and not h['alarm']: change['fp_removed'].append(r['shot'])
                if h['alarm'] and not b['alarm']: change['fp_added'].append(r['shot'])
        paired[arm]=change
    accepted=sum(r['accepted'] for u in summary['updates'] for r in u['fit_log'])
    result=dict(original_fit_bitwise_reproduced=True,original_fit_logs=logs,
        verified_future_seals=verified,future_metrics=metrics,paired_vs_H=paired,
        future_rule_proposals=sum(len(u['fit_log']) for u in summary['updates']),
        future_accepted_rule_changes=accepted,new_danger_witnesses=witnesses,rows=rows,
        original_summary_sha256=_hash(args.original/'summary.json'),
        guarded_summary_sha256=_hash(args.guarded/'summary.json'),development_only=True)
    _durable_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','original_fit_logs')}))


if __name__=='__main__': main()
