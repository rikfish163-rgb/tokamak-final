import json

import numpy as np
import pytest
import torch

from moving_peak_component_audit_20261105 import component_actions


def test_layer_and_gain_bias_ablations_partition_original_action():
    x = torch.arange(256.0)[None]
    arms = component_actions(x)
    torch.testing.assert_close(arms["layer0"] + arms["layer1"], x)
    torch.testing.assert_close(arms["gain"] + arms["bias"], x)
    assert arms["layer0"][0, 128:].eq(0).all()
    assert arms["gain"][0, 64:128].eq(0).all() and arms["gain"][0, 192:].eq(0).all()
    assert arms["zero"].eq(0).all()


def test_endpoint_parity_is_a_json_safe_python_number():
    from moving_peak_component_audit_20261105 import endpoint_error

    result = endpoint_error(0.6, 0.75, np.float32(0.5), np.float32(0.75))
    assert json.loads(json.dumps(result)) == pytest.approx(0.1)
