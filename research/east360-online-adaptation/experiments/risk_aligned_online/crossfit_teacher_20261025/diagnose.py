"""Post-hoc explanation of a failed candidate; does not select new settings."""
import argparse
from pathlib import Path
import numpy as np
import torch

from ..io23 import RAW23
from ..simple_online import _durable_json
from ..simple_train import sha
from ..finite_conditional_loss_20261025.learning import load_policy
from .fit import READ


def summarize(root):
    done=READ(root/'completed.json');audit=READ(root/'audit.json')
    assert audit['status']=='passed' and audit['completed_sha256']==sha(root/'completed.json')
    selection=READ(root/'context_selection.json')
    log=READ(root/'training'/f'context_{selection["epoch"]:02d}.json')
    policy=load_policy(READ(root/'policies.json')['context'],device='cpu')
    receipts=READ(root/'cache_receipts.json');cosines=[];weights=[]
    for row in receipts:
        c=policy(torch.tensor(row['context'],dtype=torch.double),torch.tensor(row['observed'])).detach().numpy()
        gram=np.asarray(row['channel_gram']);one=np.ones(23)
        norm2,ordinary2=float(c@gram@c),float(one@gram@one)
        if norm2<=0 or ordinary2<=0:raise AssertionError('archived active historical gradient is not nonzero')
        cosine=float(c@gram@one/np.sqrt(norm2*ordinary2))
        if not -1.000001<=cosine<=1.000001:raise AssertionError('invalid Gram-derived cosine')
        cosines.append(cosine);weights.append(c)
    w=np.asarray(weights);baselines=READ(root/'historical_baselines.json')
    result=dict(scope='POST_HOC_DIAGNOSTIC_ONLY_NO_SELECTION_OR_NEW360',
        completed_sha256=sha(root/'completed.json'),audit_sha256=sha(root/'audit.json'),
        context_epoch=selection['epoch'],strength=log['strength'],
        historical_counts=log['history_gate'],
        historical_disruptive_losses=[dict(shot=r['shot'],before=baselines[str(r['shot'])]['loss'],
            after=r['loss'],delta=r['delta'],static_prefix_winner=r['static_prefix_winner'])
            for r in log['per_shot'] if r['flag']],
        zero_credit_shots=[r['shot'] for r in log['per_shot'] if r['static_prefix_winner']],
        cached_gradient_blocks=len(receipts),winning_blocks_with_policy_gradient=sum(not r['static_prefix_winner'] for r in log['per_shot']),
        gradient_cosine_to_uniform=dict(quantile_levels=[0,.25,.5,.75,1],
            values=np.quantile(cosines,[0,.25,.5,.75,1]).tolist(),above_point99=int((np.asarray(cosines)>.99).sum())),
        channel_coefficients=[dict(name=name,mean=float(w[:,i].mean()),minimum=float(w[:,i].min()),maximum=float(w[:,i].max()))
            for i,name in enumerate(RAW23)],
        calibration_disruption_loss_change=done['results']['crossfit_context']['bag_disrupt_loss']-done['results']['off']['bag_disrupt_loss'],
        calibration_balanced_loss_gap_to_uniform=done['results']['crossfit_context']['bag_balanced_loss']-done['results']['uniform']['bag_balanced_loss'],
        calibration_balanced_loss_gap_to_permuted=done['results']['crossfit_context']['bag_balanced_loss']-done['results']['permuted']['bag_balanced_loss'],
        evidence_limit='Directional similarity is not proof of identical updates or a physical mechanism; policy context averages arrived residuals, but channel gradients can retain timing information.',
        code_sha256=sha(Path(__file__)))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    result=summarize(args.run);_durable_json(args.run/'posthoc_diagnosis.json',result);print(result)
