import h5py
import numpy as np
import pytest

from .. import io23, simple_data


def synthetic(tmp_path, monkeypatch, rate):
    path=tmp_path/'1.hdf5'
    with h5py.File(path,'w') as f:
        for key in io23.RAW23:
            ds=f.create_dataset('data/'+key,data=np.arange(4,dtype=np.float32))
            ds.attrs['SampleRate']=rate
            ds.attrs['StartTime']=.5
    monkeypatch.setattr(io23,'path_for',lambda device,shot:path)


@pytest.mark.parametrize('rate',[1000.,np.nextafter(1000.,np.inf),np.nextafter(1000.,0.)])
def test_float_neighbor_rate_keeps_actual_times(tmp_path, monkeypatch, rate):
    synthetic(tmp_path,monkeypatch,rate)
    d=io23.read_signals('EAST',1)
    assert d['rate'] == rate
    np.testing.assert_array_equal(d['times'],.5+np.arange(4)/rate)


@pytest.mark.parametrize('rate',[999.5,999.999,1000.001,500.,np.nan])
def test_real_rate_shift_is_not_silently_rounded_or_resampled(tmp_path, monkeypatch, rate):
    synthetic(tmp_path,monkeypatch,rate)
    with pytest.raises(ValueError,match='1 kHz'):
        io23.read_signals('EAST',1)


@pytest.mark.parametrize('rate',[999.5,999.5004995004996,999.9980484479389,1000.,1000.5])
def test_explicit_nominal_east_contract_retains_declared_grid(tmp_path,monkeypatch,rate):
    synthetic(tmp_path,monkeypatch,rate)
    d=io23.read_signals('EAST',1,rate_contract='nominal')
    assert d['rate']==rate
    np.testing.assert_array_equal(d['times'],.5+np.arange(4)/rate)


@pytest.mark.parametrize('rate',[999.499,1000.501,500.,np.inf,np.nan])
def test_nominal_contract_has_a_finite_hard_boundary(tmp_path,monkeypatch,rate):
    synthetic(tmp_path,monkeypatch,rate)
    with pytest.raises(ValueError,match='1 kHz'):
        io23.read_signals('EAST',1,rate_contract='nominal')


@pytest.mark.parametrize('device,contract',[('J-TEXT','nominal'),('EAST','anything'),('EAST',True)])
def test_nominal_is_target_only_and_contract_is_whitelisted(tmp_path,monkeypatch,device,contract):
    synthetic(tmp_path,monkeypatch,1000.)
    with pytest.raises(ValueError,match='rate contract'):
        io23.read_signals(device,1,rate_contract=contract)


def test_nominal_target_preprocessing_does_not_fit_a_scaler_or_read_outcomes(tmp_path,monkeypatch):
    rate=999.5004995004996
    synthetic(tmp_path,monkeypatch,rate)
    # Fixture has NO meta group: input reading must not access the outcome.
    d=simple_data.read_shot('EAST',1,rate_contract='nominal')
    np.testing.assert_array_equal(d['times'],.5+np.arange(4)/rate)
    np.testing.assert_array_equal(d['values'][:,io23.RAW23.index('ip')],np.arange(4,dtype=np.float32))
    assert not d['available'][:,io23.RAW23.index('MNM')].any()
    assert 'y' not in d and 'down' not in d
    with pytest.raises(ValueError,match='1 kHz'):
        simple_data.read_shot('EAST',1)
