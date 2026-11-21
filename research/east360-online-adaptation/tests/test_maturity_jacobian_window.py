"""CPU clock and input-boundary contracts for the origin+25 window."""

import pytest
import torch

from experiments.risk_aligned_online.maturity_jacobian_20261030.window import maturity_window


def windows(dtype=torch.float64):
    time = torch.arange(178, dtype=torch.float64)[None, :, None]
    channel = torch.arange(23, dtype=torch.float64)[None, None, :]
    batch = torch.arange(2, dtype=torch.float64)[:, None, None]
    values = (time + channel / 32 + 1000 * batch).to(dtype)
    masks = (time.long() + channel.long() + batch.long()) % 4 != 0
    inputs = (values[:, :128].clone(), masks[:, :128].clone(),
              values[:, 50:].clone(), masks[:, 50:].clone())
    return inputs, (values[:, 25:153].clone(), masks[:, 25:153].clone())


def test_numbered_sequence_is_exactly_25_through_152_for_all_channels_and_masks():
    inputs, expected = windows()
    mature_x, mature_mask = maturity_window(*inputs)
    assert mature_x.shape == mature_mask.shape == (2, 128, 23)
    assert torch.equal(mature_x, expected[0])
    assert torch.equal(mature_mask, expected[1])
    assert torch.equal(mature_x[0, :, 0], torch.arange(25, 153, dtype=torch.float64))
    assert torch.equal(mature_x[:, :103], inputs[0][:, 25:])
    assert torch.equal(mature_x[:, 103:], inputs[2][:, 78:103])
    assert torch.count_nonzero(mature_x[~mature_mask]) > 0  # Masks do not rewrite values.


@pytest.mark.parametrize('future_value', [float('nan'), float('inf'), -float('inf'), -12345.])
def test_future_153_through_177_values_and_masks_are_ignored(future_value):
    inputs, expected = windows()
    inputs[2][:, 103:] = future_value
    inputs[3][:, 103:] = ~inputs[3][:, 103:]
    originals = [value.clone() for value in inputs]
    actual = maturity_window(*inputs)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    for value, original in zip(inputs, originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0, equal_nan=True)


def test_unused_early_history_is_not_checked_for_finiteness():
    inputs, expected = windows()
    inputs[0][:, :25] = float('nan')
    inputs[1][:, :25] = ~inputs[1][:, :25]
    actual = maturity_window(*inputs)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


@pytest.mark.parametrize('overlap_index', [0, 77])
def test_overlap_values_must_match_exactly_even_for_unavailable_channels(overlap_index):
    inputs, _ = windows()
    inputs[1][1, overlap_index + 50, 22] = False
    inputs[3][1, overlap_index, 22] = False
    inputs[2][1, overlap_index, 22] = torch.nextafter(
        inputs[2][1, overlap_index, 22], torch.tensor(float('inf'), dtype=torch.float64))
    with pytest.raises(ValueError, match='overlap'):
        maturity_window(*inputs)


@pytest.mark.parametrize('overlap_index', [0, 77])
def test_overlap_masks_must_match(overlap_index):
    inputs, _ = windows()
    inputs[3][0, overlap_index, 3] = ~inputs[3][0, overlap_index, 3]
    with pytest.raises(ValueError, match='overlap'):
        maturity_window(*inputs)


@pytest.mark.parametrize('source,index', [(0, 25), (0, 127), (2, 0), (2, 77), (2, 78), (2, 102)])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_used_and_past_overlap_values_must_be_finite(source, index, value):
    inputs, _ = windows()
    inputs[source][0, index, 4] = value
    with pytest.raises(ValueError, match='finite'):
        maturity_window(*inputs)


@pytest.mark.parametrize('source,index', [(0, 25), (2, 102)])
def test_used_nonfinite_values_are_rejected_even_when_mask_is_false(source, index):
    inputs, _ = windows()
    inputs[source][0, index, 4] = float('nan')
    inputs[source + 1][0, index, 4] = False
    with pytest.raises(ValueError, match='finite'):
        maturity_window(*inputs)


def test_outputs_are_detached_independent_and_inputs_are_not_mutated():
    inputs, expected = windows()
    inputs[0].requires_grad_()
    inputs[2].requires_grad_()
    originals = [value.detach().clone() for value in inputs]
    outputs = maturity_window(*inputs)
    for output in outputs:
        assert not output.requires_grad and output.grad_fn is None
        assert all(output.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
                   for value in inputs)
    for value, original in zip(inputs, originals):
        assert torch.equal(value, original)
    outputs[0].fill_(-9)
    outputs[1].logical_not_()
    for value, original in zip(inputs, originals):
        assert torch.equal(value, original)

    outputs = maturity_window(*inputs)
    with torch.no_grad():
        for value in inputs:
            value.zero_()
    assert torch.equal(outputs[0], expected[0])
    assert torch.equal(outputs[1], expected[1])
    assert inputs[0].grad is None and inputs[2].grad is None


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_floating_dtype_is_preserved_and_noncontiguous_inputs_work(dtype):
    inputs, expected = windows(dtype)
    inputs = [value.transpose(1, 2).contiguous().transpose(1, 2) for value in inputs]
    actual = maturity_window(*inputs)
    assert actual[0].dtype == dtype and actual[1].dtype == torch.bool
    assert actual[0].device.type == actual[1].device.type == 'cpu'
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


@pytest.mark.parametrize('defect', [
    'x_type', 'x_ndim', 'x_time', 'x_channels', 'x_integer', 'x_complex',
    'mask_type', 'mask_shape', 'mask_dtype', 'mask_device',
    'later_type', 'later_batch', 'later_time', 'later_channels', 'later_dtype', 'later_device',
    'later_mask_type', 'later_mask_shape', 'later_mask_dtype', 'later_mask_device',
])
def test_input_shape_dtype_and_device_errors_are_rejected(defect):
    inputs, _ = windows()
    x, mask, later_x, later_mask = inputs
    if defect == 'x_type':
        x = None
    elif defect == 'x_ndim':
        x = x[0]
    elif defect == 'x_time':
        x = x[:, :-1]
    elif defect == 'x_channels':
        x = x[:, :, :-1]
    elif defect == 'x_integer':
        x = x.long()
    elif defect == 'x_complex':
        x = x.to(torch.complex128)
    elif defect == 'mask_type':
        mask = mask.tolist()
    elif defect == 'mask_shape':
        mask = mask[:, :-1]
    elif defect == 'mask_dtype':
        mask = mask.long()
    elif defect == 'mask_device':
        mask = torch.empty(mask.shape, dtype=torch.bool, device='meta')
    elif defect == 'later_type':
        later_x = later_x.tolist()
    elif defect == 'later_batch':
        later_x = later_x[:1]
    elif defect == 'later_time':
        later_x = later_x[:, :-1]
    elif defect == 'later_channels':
        later_x = later_x[:, :, :-1]
    elif defect == 'later_dtype':
        later_x = later_x.float()
    elif defect == 'later_device':
        later_x = torch.empty(later_x.shape, dtype=later_x.dtype, device='meta')
    elif defect == 'later_mask_type':
        later_mask = None
    elif defect == 'later_mask_shape':
        later_mask = later_mask[:1]
    elif defect == 'later_mask_dtype':
        later_mask = later_mask.float()
    else:
        later_mask = torch.empty(later_mask.shape, dtype=torch.bool, device='meta')
    with pytest.raises(ValueError):
        maturity_window(x, mask, later_x, later_mask)


def test_empty_batch_preserves_window_shape():
    x = torch.empty(0, 128, 23)
    mask = torch.empty_like(x, dtype=torch.bool)
    actual = maturity_window(x, mask, x.clone(), mask.clone())
    assert actual[0].shape == actual[1].shape == (0, 128, 23)
