"""Restrict a numerical normalization intervention, not the 23-input model."""
import numpy as np
import pytest

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.efd_normalization import source_coordinate_map,EfdReplay,ChannelMoments
from experiments.risk_aligned_online.operational_memory_experiment import validate_normalization
from experiments.risk_aligned_online.tests.test_efd_normalization import arrays,moments,Data,finish


def test_density_only_bridge_changes_no_other_observed_input_or_mask():
    x=arrays(300.).astype(np.float32);mask=np.ones_like(x,bool);mask[:10,12]=False
    original=x.copy();saved=mask.copy();source=moments(arrays());target=moments(arrays(100.))
    out,receipt=source_coordinate_map(x,mask,source,target,channels=('ne_nG',))
    c=io23.RAW23.index('ne_nG');other=np.arange(23)!=c
    np.testing.assert_array_equal(out[:,other],x[:,other])
    np.testing.assert_array_equal(out[~mask],x[~mask])
    np.testing.assert_array_equal(x,original);np.testing.assert_array_equal(mask,saved)
    assert receipt['aligned'].sum()==1 and receipt['aligned'][c]
    expected=source['mean'][c]+source['std'][c]*(x[mask[:,c],c]-target['mean'][c])/target['std'][c]
    np.testing.assert_allclose(out[mask[:,c],c],expected,rtol=1e-6)


def test_fractional_average_mode_is_not_assumed_to_be_integer_category():
    x=arrays(300.).astype(np.float32)/10
    source=moments(arrays()/10);target=moments(arrays(100.)/10)
    out,receipt=source_coordinate_map(x,np.ones_like(x,bool),source,target,channels=('mode_number_n',))
    c=io23.RAW23.index('mode_number_n')
    assert receipt['aligned'][c] and not np.array_equal(out[:,c],x[:,c])
    assert receipt['status'][c]=='device_zscore_in_source_coordinates'


@pytest.mark.parametrize('channels',[(),('ne_nG','ne_nG'),('unknown',),'ne_nG',(12,)])
def test_invalid_channel_selection_is_rejected_not_treated_as_all(channels):
    x=arrays();stats=moments(x)
    with pytest.raises(ValueError,match='channels'):
        source_coordinate_map(x,np.ones_like(x,bool),stats,stats,channels=channels)


def test_provider_keeps_same_seal_boundary_and_never_double_maps(tmp_path):
    path=tmp_path/'p';path.mkdir()
    p=EfdReplay(np.zeros(23),np.ones(23),source_statistics=moments(arrays()),
        output_dir=path,order=[1,2,3,4],calibration_count=2,data_api=Data(),channels=('ne_nG',))
    old=finish(p,1);finish(p,2)
    shot=p.read_shot('EAST',3);c=io23.RAW23.index('ne_nG');other=np.arange(23)!=c
    assert not old['alignment_active'] and shot['alignment_active']
    np.testing.assert_array_equal(shot['values'][:,other],shot['_raw_values'][:,other])
    np.testing.assert_array_equal(shot['available'],shot['_raw_available'])
    mapped=p.reencode(old);np.testing.assert_array_equal(p.reencode(mapped)['values'],mapped['values'])
    assert p.statistics['calibration_ids']==[1,2]


@pytest.mark.parametrize('normalization,path',[('density_robust',None),('efd_density_zscore','source.json')])
def test_explicit_source_moment_requirement(normalization,path):
    validate_normalization(normalization,path)


@pytest.mark.parametrize('normalization,path',[('density_robust','source.json'),
    ('efd_density_zscore',None),('unknown',None)])
def test_no_silent_source_moment_or_default_normalization(normalization,path):
    with pytest.raises(ValueError):validate_normalization(normalization,path)
