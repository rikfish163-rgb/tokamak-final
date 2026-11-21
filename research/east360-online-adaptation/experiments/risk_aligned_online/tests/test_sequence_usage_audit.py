import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.sequence_usage_audit import (
    history_view, summarize_records, choose_endpoints)


def fixture():
    x = torch.arange(128*23, dtype=torch.float32).reshape(1,128,23)
    m = torch.ones_like(x, dtype=torch.bool)
    return x,m


def test_identity_is_exact_and_does_not_alias_inputs():
    x,m=fixture(); y,a=history_view(x,m,'identity')
    assert torch.equal(x,y) and torch.equal(m,a)
    y.zero_(); a.zero_()
    assert bool(x.any()) and bool(m.all())


def test_reverse_old_patch_order_preserves_rows_masks_and_latest_patch():
    x,m=fixture(); m[:,20,3]=False
    y,a=history_view(x,m,'reverse_old_patches')
    assert torch.equal(y[:,120:],x[:,120:]) and torch.equal(a[:,120:],m[:,120:])
    assert torch.equal(y[:,:8],x[:,112:120])
    assert torch.equal(a[:,96:104],m[:,16:24])
    z,b=history_view(y,a,'reverse_old_patches')
    assert torch.equal(x,z) and torch.equal(m,b)


def test_padding_stays_in_place_and_never_becomes_an_observation():
    x,m=fixture(); m[:,:101]=False; x[:,:101]=0
    y,a=history_view(x,m,'reverse_old_patches')
    assert torch.equal(a,m) and torch.equal(y[:,:101],x[:,:101])
    # Only complete observed patches 104:112 and 112:120 exchange.
    assert torch.equal(y[:,104:112],x[:,112:120])
    assert torch.equal(y[:,101:104],x[:,101:104])


def test_hold_uses_last_real_value_per_channel_not_zero_missing_tail():
    x,m=fixture(); m[:,125:,2]=False; m[:,:,3]=False
    y,a=history_view(x,m,'hold_old_at_last')
    assert torch.equal(a,m) and torch.equal(y[:,120:],x[:,120:])
    assert torch.equal(y[:,:120,2],x[:,124:125,2].expand(1,120))
    assert torch.equal(y[:,:,3],x[:,:,3])


def test_reverse_preserves_contemporaneous_multichannel_rows():
    x,m=fixture(); x[:,:,4]=x[:,:,2]+3
    y,_=history_view(x,m,'reverse_old_patches')
    assert torch.equal(y[:,:,4],y[:,:,2]+3)


@pytest.mark.parametrize('view',['no_such_view',None])
def test_unknown_views_fail(view):
    x,m=fixture()
    with pytest.raises(ValueError):history_view(x,m,view)


def test_invalid_shape_and_nonboolean_masks_fail():
    x,m=fixture()
    with pytest.raises(ValueError):history_view(x[:,:127],m[:,:127],'identity')
    with pytest.raises(ValueError):history_view(x,m.float(),'identity')


def test_endpoint_selection_uses_no_outcomes_and_includes_first_alert():
    scores=np.zeros(1300,np.float32); scores[17]=.9
    points=choose_endpoints(scores,.7)
    assert points==[0,5,17,25,127,500,1000,1299]
    assert choose_endpoints(np.zeros(5,np.float32),.7)==[0,4]


def test_summary_uses_shots_not_endpoint_counts_as_independent_units():
    rows=[dict(shot=1,kind='first_alarm',category='FP',base=.9,changed=.8,
               view='hold_old_at_last',changed_patches=2),
          dict(shot=1,kind='scheduled',category='FP',base=.9,changed=.1,
               view='hold_old_at_last',changed_patches=2),
          dict(shot=2,kind='first_alarm',category='TP',base=.9,changed=.95,
               view='hold_old_at_last',changed_patches=1)]
    s=summarize_records(rows,.7)
    assert s['first_alarm']['hold_old_at_last']['FP']['shots']==1
    assert s['first_alarm']['hold_old_at_last']['FP']['lost_crossings']==0
    assert s['first_alarm']['hold_old_at_last']['FP']['median_probability_change']==pytest.approx(-.1)
    assert s['first_alarm']['hold_old_at_last']['TP']['shots']==1


def test_no_op_views_are_reported_not_counted_as_valid_interventions():
    rows=[dict(shot=1,kind='first_alarm',category='FP',base=.9,changed=.9,
               view='reverse_old_patches',changed_patches=0)]
    s=summarize_records(rows,.7)
    assert s['first_alarm']['reverse_old_patches']['FP']['valid_shots']==0

