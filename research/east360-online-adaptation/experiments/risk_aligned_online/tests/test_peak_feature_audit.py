"""One-feature, observed-slot-only diagnostics are not physical interventions."""
import numpy as np
import pytest

from experiments.risk_aligned_online.peak_feature_audit import intervene_window, summarize
from experiments.risk_aligned_online.io23 import RAW23


def inputs():
    rng=np.random.default_rng(131)
    x=rng.normal(size=(128,23)).astype(np.float32)
    a=np.ones_like(x,bool);a[:20,12]=False;x[:20,12]=999.
    return x,a,np.arange(23,dtype=np.float32)/10,np.linspace(.5,2,23,dtype=np.float32)


def test_center_replacement_changes_only_observed_selected_values():
    x,a,c,s=inputs();old=x.copy();mask=a.copy()
    y,receipt=intervene_window(x,a,c,s,12,'source_center')
    np.testing.assert_array_equal(y[a[:,12],12],np.full(a[:,12].sum(),c[12]))
    np.testing.assert_array_equal(y[~a[:,12],12],x[~a[:,12],12])
    np.testing.assert_array_equal(y[:,np.arange(23)!=12],x[:,np.arange(23)!=12])
    np.testing.assert_array_equal(x,old);np.testing.assert_array_equal(a,mask)
    assert receipt['status']=='applied' and receipt['observed_points']==108


def test_centered_shape_removes_only_source_coordinate_location():
    x,a,c,s=inputs()
    y,receipt=intervene_window(x,a,c,s,12,'centered_shape')
    v=a[:,12]
    z=np.arcsinh((x[v,12].astype(float)-c[12])/s[12])
    changed=np.arcsinh((y[v,12].astype(float)-c[12])/s[12])
    np.testing.assert_allclose(changed,z-np.median(z),atol=3e-7,rtol=3e-7)
    np.testing.assert_allclose(np.diff(changed),np.diff(z),atol=5e-7,rtol=5e-7)
    np.testing.assert_array_equal(y[~v,12],x[~v,12])
    assert receipt['normalized_location_removed']==pytest.approx(float(np.median(z)))


@pytest.mark.parametrize('method',['source_center','centered_shape'])
def test_no_observations_is_reported_identity_not_a_synthetic_measurement(method):
    x,a,c,s=inputs();a[:,12]=False
    y,row=intervene_window(x,a,c,s,12,method)
    np.testing.assert_array_equal(y,x)
    assert row['status']=='no_observed_values' and row['observed_points']==0


@pytest.mark.parametrize('method',['source_center','centered_shape'])
def test_discrete_mode_is_not_treated_as_a_continuous_measurement(method):
    x,a,c,s=inputs()
    y,row=intervene_window(x,a,c,s,RAW23.index('mode_number_n'),method)
    np.testing.assert_array_equal(y,x)
    assert row['status']=='not_continuous_feature'


@pytest.mark.parametrize('bad',['shape','mask','nan','scale','channel','method'])
def test_invalid_input_is_rejected(bad):
    x,a,c,s=inputs();channel=12;method='source_center'
    if bad=='shape':x=x[:-1]
    elif bad=='mask':a=a.astype(np.float32)
    elif bad=='nan':x[100,12]=np.nan
    elif bad=='scale':s[1]=0.
    elif bad=='channel':channel=True
    else:method='unknown'
    with pytest.raises(ValueError):intervene_window(x,a,c,s,channel,method)


def test_summary_reports_peak_crossings_not_whole_shot_rescues():
    row=dict(coordinates='raw',category='safe_fp',baseline_probability=.8,
             interventions=[dict(channel='ne_nG',method='source_center',probability=.6,status='applied')])
    out=summarize([row],.7)
    group=out['raw']['safe_fp']['ne_nG']['source_center']
    assert group['original_peak_crossing_down']==1 and group['n_windows']==1
    assert group['mean_probability_change']==pytest.approx(-.2)
    assert 'tp' not in group and 'fp' not in group and 'tpr' not in group and 'fpr' not in group
