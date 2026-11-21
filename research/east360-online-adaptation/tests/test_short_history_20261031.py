import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.delayed_recurrent_memory_20261025.learning import batch
from experiments.risk_aligned_online.lifecycle_stream_20261025.learning import (
    cache_example,
    logits,
    window_loss,
)
from experiments.risk_aligned_online.lifecycle_stream_20261025.model import LifecycleAdapter
from experiments.risk_aligned_online.lifecycle_stream_20261025.protocol import sampled_endpoints
from experiments.risk_aligned_online.short_history_20261031.sampling import sampled_with_prefix
from experiments.risk_aligned_online.simple23 import Config, Simple23


@pytest.fixture
def arrays():
    n = 300
    return dict(
        times=np.arange(n, dtype=float) / 1000,
        y=(np.arange(n) >= 200).astype(float),
        valid=np.ones(n, dtype=bool),
        scores=np.sin(np.arange(n) / 9),
    )


def test_safe_prefix_includes_both_boundaries_and_preserves_all_old_endpoints(arrays):
    arrays['y'][:] = 0
    old = sampled_endpoints(**arrays, down=None)
    ends = sampled_with_prefix(**arrays, down=None)

    assert set(old).issubset(ends)
    np.testing.assert_array_equal(ends, np.unique(ends))
    prefix = ends[ends < 127]
    np.testing.assert_array_equal(prefix, np.linspace(0, 126, 16, dtype=int))
    assert len(ends) - len(old) == 16
    assert 0 in ends and 126 in ends


@pytest.mark.parametrize('valid_prefix', [
    [0, 5, 126],
    list(range(0, 83, 2)) + [97, 109, 126],
])
def test_prefix_is_uniform_over_valid_indices_and_never_exceeds_sixteen(arrays, valid_prefix):
    arrays['valid'][:127] = False
    arrays['valid'][valid_prefix] = True
    ends = sampled_with_prefix(**arrays, down=None)
    candidates = np.asarray(valid_prefix)
    expected = candidates[np.linspace(0, len(candidates) - 1,
                                       min(16, len(candidates)), dtype=int)]

    np.testing.assert_array_equal(ends[ends < 127], expected)
    assert len(ends[ends < 127]) <= 16
    assert arrays['valid'][ends].all()


def test_disruptive_prefix_uses_rounded_strict_30ms_boundary(arrays):
    # Nonmonotone synthetic timestamps exercise the boundary inside the prefix
    # while retaining old, timely endpoints after index 126.
    arrays['valid'][:127] = False
    arrays['valid'][:5] = True
    arrays['times'][:5] = [.169, .170, .171, .1699999999996, .169999999]
    old = sampled_endpoints(**arrays, down=.2)
    ends = sampled_with_prefix(**arrays, down=.2)

    assert set(old).issubset(ends)
    np.testing.assert_array_equal(ends[ends < 127], [0, 4])
    assert np.all(np.round((.2 - arrays['times'][ends]) * 1000, 9) > 30)


def test_prefix_selection_does_not_depend_on_labels_or_scores(arrays):
    first = sampled_with_prefix(**arrays, down=None)
    altered = {key: value.copy() for key, value in arrays.items()}
    altered['y'] = 1 - altered['y']
    altered['scores'] = -altered['scores'] * 7
    second = sampled_with_prefix(**altered, down=None)

    np.testing.assert_array_equal(first[first < 127], second[second < 127])


@pytest.mark.parametrize('down', [None, .2])
def test_no_valid_prefix_returns_exact_old_result(arrays, down):
    arrays['valid'][:127] = False
    old = sampled_endpoints(**arrays, down=down)
    ends = sampled_with_prefix(**arrays, down=down)

    np.testing.assert_array_equal(ends, old)


def test_no_timely_prefix_returns_exact_old_result(arrays):
    # Keep later synthetic timestamps timely to exercise this independent case.
    arrays['times'][:127] = .2
    old = sampled_endpoints(**arrays, down=.2)
    ends = sampled_with_prefix(**arrays, down=.2)

    np.testing.assert_array_equal(ends, old)


@pytest.mark.parametrize('key', ['times', 'y', 'valid', 'scores'])
@pytest.mark.parametrize('shape_change', ['longer', 'shorter', 'two_dimensional'])
def test_mismatched_or_non_1d_arrays_are_rejected(arrays, key, shape_change):
    if shape_change == 'longer':
        arrays[key] = np.concatenate((arrays[key], arrays[key][-1:]))
    elif shape_change == 'shorter':
        arrays[key] = arrays[key][:-1]
    else:
        arrays[key] = arrays[key][:, None]

    with pytest.raises(ValueError):
        sampled_with_prefix(**arrays, down=None)


@pytest.mark.parametrize('key', ['times', 'y', 'scores'])
@pytest.mark.parametrize('invalid', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('index', [0, 299])
def test_nonfinite_values_are_rejected_even_at_invalid_or_future_indices(arrays, key, invalid, index):
    arrays['valid'][index] = False
    arrays[key][index] = invalid

    with pytest.raises(ValueError):
        sampled_with_prefix(**arrays, down=None)


@pytest.mark.parametrize('invalid', [
    lambda a: a.astype(int),
    lambda a: a.astype(float),
    lambda a: a.astype(object),
    lambda a: np.full(len(a), 'true'),
])
def test_valid_must_be_a_boolean_array(arrays, invalid):
    arrays['valid'] = invalid(arrays['valid'])
    with pytest.raises(ValueError):
        sampled_with_prefix(**arrays, down=None)


@pytest.mark.parametrize('key', ['times', 'y', 'scores'])
def test_nonnumeric_values_are_rejected(arrays, key):
    arrays[key] = arrays[key].astype(object)
    arrays[key][50] = 'not a number'
    with pytest.raises(ValueError):
        sampled_with_prefix(**arrays, down=None)


@pytest.mark.parametrize('down', [np.nan, np.inf, -np.inf, [0.2]])
def test_invalid_disruption_time_is_rejected(arrays, down):
    with pytest.raises(ValueError):
        sampled_with_prefix(**arrays, down=down)


@pytest.mark.parametrize('case', ['empty', 'all_invalid', 'only_short_history', 'no_timely_old'])
def test_original_no_data_error_is_not_masked(arrays, case):
    down = None
    if case == 'empty':
        arrays = {key: value[:0] for key, value in arrays.items()}
    elif case == 'all_invalid':
        arrays['valid'][:] = False
    elif case == 'only_short_history':
        arrays = {key: value[:127] for key, value in arrays.items()}
    else:
        down = .1

    with pytest.raises(ValueError) as old_error:
        sampled_endpoints(**arrays, down=down)
    with pytest.raises(ValueError) as new_error:
        sampled_with_prefix(**arrays, down=down)
    assert str(new_error.value) == str(old_error.value)


def test_early_selected_endpoints_work_with_existing_batch_and_loss(arrays):
    ends = sampled_with_prefix(**arrays, down=.3)
    signal = dict(
        values=np.repeat(np.arange(300, dtype=np.float32)[:, None], 23, axis=1),
        available=np.ones((300, 23), dtype=bool),
    )
    windows, masks = batch(signal, [0, 126, 127], torch.device('cpu'))
    assert windows.shape == masks.shape == (3, 128, 23)
    assert not masks[0, :-1].any() and masks[0, -1].all()
    assert not masks[1, 0].any() and masks[1, 1:].all()
    assert masks[2].all()
    assert torch.equal(windows[1, -1], torch.full((23,), 126.))
    changed = dict(signal, values=signal['values'].copy())
    changed['values'][127:] = -1000
    changed_windows, changed_masks = batch(changed, [0, 126], torch.device('cpu'))
    assert torch.equal(changed_windows, windows[:2])
    assert torch.equal(changed_masks, masks[:2])

    labels = torch.tensor(arrays['y'][ends], dtype=torch.float32)
    logits = torch.zeros(len(ends), requires_grad=True)
    loss = window_loss(logits, labels, True, 'disruptive_positive_only')
    gradient, = torch.autograd.grad(loss, logits)
    assert (labels == 0).any() and (labels == 1).any()
    assert torch.all(gradient[labels == 0] == 0)
    assert torch.all(gradient[labels == 1] < 0)


def test_selected_short_history_reaches_existing_model_and_masked_loss(arrays):
    torch.set_num_threads(1)
    torch.manual_seed(131)
    ends = sampled_with_prefix(**arrays, down=.3)
    positive = ends[arrays['y'][ends] > 0][-1]
    selected = np.array([0, 126, positive])
    assert set(selected).issubset(ends)
    base = Simple23(Config(dim=16, fast_site='tokens', fast_rule='normalized'),
                    np.zeros(23), np.ones(23)).eval()
    model = LifecycleAdapter(base).eval()
    signal = dict(values=np.ones((300, 23), dtype=np.float32),
                  available=np.ones((300, 23), dtype=bool))
    example = cache_example(model, signal, selected, arrays['y'][selected],
                            sid=1, flag=True)
    z = logits(model, example, fast=True)
    z.retain_grad()
    loss = window_loss(z, example['y'], example['flag'], 'disruptive_positive_only')
    loss.backward()

    assert example['ends'] == selected.tolist()
    assert torch.isfinite(z).all() and torch.isfinite(loss)
    assert torch.equal(z.grad[:2], torch.zeros(2))
    assert z.grad[2] < 0


def test_inputs_are_not_mutated(arrays):
    before = {key: value.copy() for key, value in arrays.items()}
    sampled_with_prefix(**arrays, down=.3)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before[key])
