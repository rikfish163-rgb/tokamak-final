import numpy as np
import torch

from experiments.risk_aligned_online.latent_features import extract_shot
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_frozen_features_reproduce_base_scores_and_are_prefix_causal():
    torch.set_num_threads(2)
    torch.manual_seed(9)
    model = Simple23(Config(window=8, patch=4, dim=8, heads=2, layers=1),
                     torch.zeros(23), torch.ones(23)).eval()
    x = np.random.default_rng(3).normal(size=(18, 23)).astype(np.float32)
    shot = dict(values=x, available=np.ones_like(x, bool))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    features, logits = extract_shot(model, shot, 'cpu', batch_size=5)
    assert features.shape == (18, 8) and logits.shape == (18,)
    torch.testing.assert_close(model.risk_head(torch.as_tensor(features)).squeeze(-1),
                               torch.as_tensor(logits))
    changed = dict(values=x.copy(), available=shot['available'])
    changed['values'][10:] += 100
    future_features, future_logits = extract_shot(model, changed, 'cpu', batch_size=5)
    np.testing.assert_array_equal(features[:10], future_features[:10])
    np.testing.assert_array_equal(logits[:10], future_logits[:10])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert len(model.risk_head._forward_pre_hooks) == 0


def test_feature_extraction_reads_no_shot_labels():
    class Guard(dict):
        def __getitem__(self, key):
            assert key not in ('y', 'valid', 'is_disrupt', 'down')
            return super().__getitem__(key)
    model = Simple23(Config(window=8, patch=4, dim=8, heads=2, layers=1),
                     torch.zeros(23), torch.ones(23)).eval()
    x = np.zeros((6, 23), np.float32)
    features, logits = extract_shot(model, Guard(values=x, available=np.ones_like(x, bool)), 'cpu')
    assert np.isfinite(features).all() and np.isfinite(logits).all()
