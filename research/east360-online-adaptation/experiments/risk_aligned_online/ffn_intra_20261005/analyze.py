"""Recompute paired metrics from sealed traces; no model/threshold mutation."""
import argparse
import json
from pathlib import Path

from .. import simple_data
from ..representation_analysis import (_load_sealed_arm,_outcome_rows,_verify_summary,
    metrics_for_rows,paired_flips,threshold_diagnostic)
from ..simple_online import _durable_json


def analyze(run):
    run=Path(run)
    protocol=json.loads((run/'protocol.json').read_text())
    order=json.loads((run/'order.json').read_text())
    limit=protocol['args']['limit']
    expected=[r['shot'] for r in (order[:limit] if limit else order)]
    arms={};rows={};traces={}
    for arm in protocol['args']['arms']:
        result=json.loads((run/arm/'summary.json').read_text())
        traces[arm]=_load_sealed_arm(run/arm)
        if [r['shot'] for r in traces[arm]]!=expected:
            raise ValueError('observed order does not match predeclared prefix')
        rows[arm]=_outcome_rows(traces[arm],simple_data.attach_outcome)
        m=metrics_for_rows(rows[arm],protocol['threshold'])
        _verify_summary(result,result['metrics'],m,len(traces[arm]))
        audits=[json.loads(f.read_text()) for f in sorted((run/arm).glob('*_audit.json'))]
        events=[e for a in audits for e in a['events']]
        arms[arm]=dict(metrics=m,updates=result['accepted_intra_updates'],
            seconds=result['elapsed_seconds'],
            forecast_loss_improved=sum(e['loss_after']<e['loss_before'] for e in events if e['accepted']),
            at_radius=sum(e['delta_norm']>=protocol['radius']*(1-1e-5) for e in events),
            early_above_3s=sum(p['timely'] and p['lead_ms']>3000 for p in m['predictions']),
            stages={name:metrics_for_rows(rows[arm][lo:hi],protocol['threshold'])
                    for name,lo,hi in (('first30',0,30),('31to120',30,120))},
            threshold_scan=threshold_diagnostic(rows[arm]))
    flips={f'{b}_vs_{a}':paired_flips(arms[a]['metrics']['predictions'],arms[b]['metrics']['predictions'],
                                    before_arm=a,after_arm=b)
           for a,b in (('static','fixed'),('static','learned'),('fixed','learned')) if a in arms and b in arms}
    return dict(evidence='DEVELOPMENT_PREFIX' if limit else 'DEVELOPMENT_FULL_STREAM',
        observed_count=len(expected),roster_count=len(order),sealed_metrics_verified=True,
        arms=arms,paired_flips=flips,interpretation='single development run; no independent validation/significance claim')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();output=args.run/'audited_comparison.json'
    if output.exists():raise FileExistsError(output)
    report=analyze(args.run);_durable_json(output,report)
    print(json.dumps({a:{k:v for k,v in r['metrics'].items() if k!='predictions'}
                      for a,r in report['arms'].items()},indent=2))
    print(json.dumps(report['paired_flips'],indent=2))


if __name__=='__main__':main()
