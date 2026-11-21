import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.negative_sampling_20261023.learning import paired_batch
from experiments.risk_aligned_online.simple_train import pools
from experiments.risk_aligned_online.tests.test_horizon_memory import records
from experiments.risk_aligned_online.tests.test_task_geometry import small_model


def test_replaces_only_disruptive_negative_preserving_source_draw_clock():
    data=records();group=pools(data);cfg=small_model().cfg
    original_y=[s['y'].copy() for s in data]
    rng_a=np.random.default_rng(131);rng_b=np.random.default_rng(131)
    replace_a=np.random.default_rng(6131);replace_b=np.random.default_rng(6131)
    by_id={s['shot']:s for s in data};total=0
    for _ in range(8):
        xa,ma,ya,ra,pa=paired_batch(data,group,rng_a,replace_a,cfg,'legacy',device='cpu')
        xb,mb,yb,rb,pb=paired_batch(data,group,rng_b,replace_b,cfg,'safe_negative',device='cpu')
        assert pa['proposed']==pb['proposed']==ra
        assert not pa['replaced']
        torch.testing.assert_close(ya,yb,rtol=0,atol=0)
        for i,(a,b) in enumerate(zip(ra,rb)):
            changed=a['label']==0 and by_id[a['shot']]['is_disrupt']
            if changed:
                total+=1
                assert not by_id[b['shot']]['is_disrupt'] and b['label']==0
                assert by_id[b['shot']]['valid'][b['end']]
                assert any(r['index']==i and r['before']==a and r['after']==b for r in pb['replaced'])
            else:
                assert a==b
                torch.testing.assert_close(xa[i],xb[i],rtol=0,atol=0)
                torch.testing.assert_close(ma[i],mb[i],rtol=0,atol=0)
    assert total>0 and rng_a.bit_generator.state==rng_b.bit_generator.state
    for s,y in zip(data,original_y):np.testing.assert_array_equal(s['y'],y)


def test_non_east_or_unsupported_policy_cannot_silently_change_training():
    data=records();cfg=small_model().cfg;group=pools(data)
    with pytest.raises(ValueError):paired_batch(data,group,np.random.default_rng(1),np.random.default_rng(2),cfg,'invented',device='cpu')
    data[0]['device']='J-TEXT'
    with pytest.raises(ValueError):paired_batch(data,group,np.random.default_rng(1),np.random.default_rng(2),cfg,'safe_negative',device='cpu')
