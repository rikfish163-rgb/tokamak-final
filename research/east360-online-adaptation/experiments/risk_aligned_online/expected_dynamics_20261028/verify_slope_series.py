"""Independent arithmetic/provenance audit of the completed diagnostic series."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from .. import io23
from ..context_memory_20261028.checkpoint import read
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha


def run():
    root=io23.ROOT/'probe_outputs'
    out=root/'slope_series_verification.json'
    if out.exists():
        raise FileExistsError(out)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    source=root/'anchored_forecast_factorial_v1';sp=read(source,'protocol.json');sd=read(source,'complete.json')
    assert sd['complete']
    orders=[]
    for name,saved in sd['results'].items():
        assert sha(source/name/'forecast_epoch10.pt')==saved['checkpoint_sha256']
        history=read(source/name,'training.json')
        assert [r['epoch'] for r in history]==list(range(1,11))
        assert all(r['steps']==r['nonzero_gradient_steps'] for r in history)
        orders.append([r['order_sha256'] for r in history])
        v=read(source/name,'val_epoch10.json')
        assert [r['shot'] for r in v['rows']]==sp['val_ids']
        for g in ('all','D','S'):
            sub=[r for r in v['rows'] if g=='all' or r['disruptive']==(g=='D')]
            for k in ('prediction','persistence'):
                assert np.isclose(np.mean([r[k] for r in sub]),saved['summary'][g][k],rtol=1e-12)
    assert all(v==orders[0] for v in orders)
    with zipfile.ZipFile(source/'code_snapshot.zip') as z:
        for f,h in sp['code_sha256'].items():
            assert hashlib.sha256(z.read(f)).hexdigest()==h
    verified={}
    for name in ('anchored_forecast_east360_v1','matured_slope_probe_v1','matured_slope_release_v1'):
        d=root/name;p=read(d,'protocol.json');done=read(d,'complete.json')
        assert done['complete'] and not done['gate']['eligible']
        assert len(list((d/'seals').glob('*.json')))==360
        rows=done['rows']
        groups=rows if isinstance(rows,dict) else {'single':rows}
        for arm,rr in groups.items():
            assert [r['shot'] for r in rr]==p['order']
            assert len(set(r['shot'] for r in rr))==360 and sum(r['disruptive'] for r in rr)==180
            for section,part in (('full360',rr),('cold20',rr[:20]),('suffix340',rr[20:])):
                for g in ('all','D','S'):
                    sub=[r for r in part if g=='all' or r['disruptive']==(g=='D')]
                    target=done['results'][arm][section][g] if isinstance(rows,dict) else done['results'][section][g]
                    fields=('prediction','persistence') if isinstance(rows,dict) else tuple(sub[0]['errors'])
                    for k in fields:
                        raw=[r[k] if isinstance(rows,dict) else r['errors'][k] for r in sub]
                        expected=target[k] if isinstance(rows,dict) else target['errors'][k]
                        assert np.isclose(np.mean(raw),expected,rtol=1e-12,atol=1e-14)
        verified[name]=dict(complete_sha256=sha(d/'complete.json'),seals=360,arms=len(groups))
    attr=read(root/'matured_slope_probe_v1','failure_attribution.json')
    assert attr['verified_forecast_seals']==360
    assert attr['reference_complete_sha256']==verified['matured_slope_probe_v1']['complete_sha256']
    original_stream=Path(read(root/'matured_slope_probe_v1','protocol.json')['stream'])
    cold={r['shot']:r for r in read(original_stream,'per_shot.json')['cold']}
    diagnostics=[]
    for r in attr['largest_local_errors'][:12]:
        q=cold[r['shot']]
        diagnostics.append(dict(shot=r['shot'],aux_excess=r['total_excess'],channel=r['channel'],
            old_cold_status=('TP' if q['timely'] else 'MISS_OR_TARDY') if q['is_disrupt'] else ('FP' if q['alarm'] else 'TN')))
    channels=attr['channel_contributions'];total=sum(r['excess'] for r in channels)
    excess_top3=sum(r['excess'] for r in channels[:3])
    paired={}
    for name,field in (('matured_slope_probe_v1','memory'),('matured_slope_release_v1','delayed')):
        rows=read(root/name,'complete.json')['rows']
        difference=np.array([r['errors'][field]-r['errors']['persistence'] for r in rows])
        paired[name]=dict(mean=float(difference.mean()),median=float(np.median(difference)),
            q25=float(np.quantile(difference,.25)),q75=float(np.quantile(difference,.75)),
            positive=int((difference>0).sum()),negative=int((difference<0).sum()),
            interpretation='descriptive_shot_differences_not_independent_confirmation_or_risk')
    paths=list(Path(__file__).parent.glob('*.py'))
    paths += [io23.ROOT/n for n in ('MATURED_SLOPE_PLAN.md','SLOPE_RELEASE_PLAN.md','ANCHORED_FORECAST_PLAN.md')]
    paths += [io23.ROOT/'experiments/risk_aligned_online/tests/test_matured_slope.py',
              io23.ROOT/'experiments/risk_aligned_online/tests/test_anchored_forecast.py']
    archive=root/'slope_series_code_snapshot.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for f in paths:z.write(f,str(f.relative_to(io23.ROOT)))
    result=dict(complete=True,source_unchanged=True,source_sha256=PARENT_SHA256,
        four_source_training_orders_identical=True,source_val=160,results=verified,
        top3_channels=[r['channel'] for r in channels[:3]],top3_excess_fraction=excess_top3/total,
        paired_differences=paired,largest_aux_errors_old_risk_status=diagnostics,
        snapshot_sha256=sha(archive),code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in paths})
    _durable_json(out,result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':run()
