"""Describe completed traces; never fit or overwrite a learner/threshold.

Matched-FPR curves use all TEST outcomes AFTER prediction sealing. They are
oracle diagnostics of ranking, not operational or independent test results.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import io23
from ..operating_curve_audit import operating_curve, operational_maxima
from ..simple_online import _durable_json
from ..simple_train import sha
from .run import ARMS


def updated_region_masks(disrupt,lead,valid):
    # Always copy: changing the diagnostic subregion must not crop the main
    # whole-shot safety mask shared by the immutable prediction records.
    masks={'safe':valid.copy() if not disrupt else np.zeros(len(valid),bool),
           'disrupt_early':valid & (lead>125) if disrupt else np.zeros(len(valid),bool),
           'disrupt_timely_window':valid & (lead>30) & (lead<=125) if disrupt else np.zeros(len(valid),bool)}
    for mask in masks.values(): mask[:256]=False
    return masks


def main(root):
    verified=json.loads((root/'verification.json').read_text())
    completed=json.loads((root/'completed.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    assert verified['status']=='VERIFIED' and sha(root/'completed.json')==verified['completed_sha256']
    rows={name:[] for name in completed['metrics']}
    prediction_changes={arm:dict(safe=[],disrupt_early=[],disrupt_timely_window=[]) for arm in ARMS}
    update_losses={arm:dict(steps=0,loss_reduced=0) for arm in ARMS}
    for sid in protocol['contract']['shot_ids']:
        d,down=io23.read_outcome('EAST',sid)
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
            times=trace['times']; lead=np.round((down-times)*1000,9) if d else None
            valid=lead>=-1e-6 if d else np.ones(len(times),dtype=bool)
            for name in rows:
                rows[name].append(dict(shot=sid,is_disrupt=d,down=down,times=times.copy(),
                    valid=valid,scores=trace[name].copy()))
            for arm in ARMS:
                # Exclude the untouched initial256 points; summarize each shot
                # first, so long shots do not automatically dominate this audit.
                masks=updated_region_masks(d,lead,valid)
                for key,mask in masks.items():
                    if mask.any():
                        delta=trace[f'{arm}_intra'][mask]-trace[f'{arm}_static'][mask]
                        prediction_changes[arm][key].append(dict(shot=sid,mean=float(delta.mean()),
                            maximum=float(delta.max()),minimum=float(delta.min())))
        updates=json.loads((root/'shots'/f'{sid}_updates.json').read_text())
        for arm in ARMS:
            for u in updates[arm]:
                if u['gradient_steps']:
                    update_losses[arm]['steps']+=u['gradient_steps']
                    update_losses[arm]['loss_reduced']+=u['loss_after']<u['loss_before']
    operating={}
    for name,records in rows.items():
        curve=operating_curve(records,protocol['threshold']);fixed=curve['fixed_source_threshold']
        assert all(fixed[k]==completed['metrics'][name][k] for k in ('tp','fp','tpr','fpr'))
        maxima=operational_maxima(records)
        d,s=maxima['positive'][:,None],maxima['negative'][None,:]
        operating[name]=dict(timely_maximum_auc=float(((d>s)+.5*(d==s)).mean()),
                            **{k:v for k,v in curve.items() if k!='curve'})
    summary={arm:{key:dict(shots=len(rs),median_shot_mean_probability_change=float(np.median([r['mean'] for r in rs])),
                          shots_mean_increased=sum(r['mean']>0 for r in rs))
                   for key,rs in groups.items()} for arm,groups in prediction_changes.items()}
    result=dict(scope='POSTHOC DEVELOPMENT DIAGNOSTIC; FULL TEST LABELS USED FOR ORACLE FPR CURVES',
        changes_to_models_predictions_thresholds=False,not_a_deployment_result=True,
        verification_sha256=sha(root/'verification.json'),code_sha256=sha(Path(__file__)),
        operating=operating,update_losses=update_losses,probability_changes=summary,
        per_shot_probability_changes=prediction_changes)
    _durable_json(root/'posthoc_diagnostic.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='per_shot_probability_changes'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    main(p.parse_args().root)
