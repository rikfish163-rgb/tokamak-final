import numpy as np
import pytest

from run_layer0_direct_shadow_20261112 import load_released_proposal
from moving_peak_replay_20261105 import sha


def test_frozen_proposal_cannot_be_loaded_one_shot_before_its_release(tmp_path):
    proposal = tmp_path/'02_proposal.npz'
    delta = np.full(128, .03, dtype=np.float32)
    np.savez(proposal, delta=delta)
    receipt = dict(first_next_rank=41, proposal_sha256=sha(proposal), fit={'steps':20})
    with pytest.raises(ValueError, match='release'):
        load_released_proposal(proposal, receipt, completed_rank=39)
    np.testing.assert_array_equal(load_released_proposal(proposal, receipt, completed_rank=40), delta)
