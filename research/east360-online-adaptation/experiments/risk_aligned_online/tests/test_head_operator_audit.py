import pytest

from experiments.risk_aligned_online.head_operator_20261022.audit import validate_update


def test_update_receipt_enforces_arrival_and_operator_bound():
    receipt=dict(last_observation=255,first_affected_prediction=256,
                 forecast_origins=[230,205,180,155],observed_targets=2200,
                 gradient_steps=1,gradient_norm=2.,loss_before=.2,loss_after=.19,
                 distance_from_origin=.9,step_norm=.05)
    validate_update('token_original',receipt,400)
    with pytest.raises(AssertionError):
        validate_update('token_original',dict(receipt,forecast_origins=[240]),400)
    with pytest.raises(AssertionError):
        validate_update('token_original',dict(receipt,distance_from_origin=1.01),400)
    with pytest.raises(AssertionError):
        validate_update('token_original',dict(receipt,first_affected_prediction=400),400)
    encoder=dict(receipt,parameter_delta_norm=.001)
    validate_update('encoder_original',encoder,400)
    with pytest.raises(AssertionError):
        validate_update('encoder_original',dict(encoder,parameter_delta_norm=.01),400)
