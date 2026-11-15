"""Causal ablation: replace arrived values by a deliberately flat target."""
from unittest.mock import patch

from ..increment_forecast_20261023 import learning


def persistence_target(examples):
    x, available, truth, truth_mask, ends = examples
    return x, available, x[:, -1:].expand_as(truth).clone(), truth_mask, ends


def predict_sham(model, signal):
    original = learning.forecast_examples
    def changed(*args, **kwargs):
        return persistence_target(original(*args, **kwargs))
    # Only this single-threaded experimental call substitutes the arrived
    # target. Context restoration also runs on exceptions. Archived code stays
    # byte-identical; real feedback and prediction dispatch are not edited.
    with patch.object(learning, 'forecast_examples', side_effect=changed):
        return learning.predict(model, signal)
