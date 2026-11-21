import copy

import pytest
import torch

from experiments.risk_aligned_online.anticipatory_decoder import load_anchored_decoder
from experiments.risk_aligned_online.simple23 import Simple23, Config
from experiments.risk_aligned_online.persistent_ffn import PersistentFfnModel


def sample(tmp_path):
    torch.manual_seed(131)
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'), torch.zeros(23), torch.ones(23)).eval()
    head = {k: torch.zeros_like(v) for k, v in base.forecast_head.state_dict().items()}
    protocol = dict(parent_sha256='a'*64, source_only=True, decoder_kind='last_observation_plus_direct_increment',
        source_classifier_frozen=True, source_scaler_frozen=True, original_source_labels_unchanged=True,
        no_east_reads=True, source_train_ids=[1, 2], fit_ids=[1], selection_ids=[2])
    path = tmp_path/'decoder.pt'
    torch.save(dict(forecast_head=head, protocol=protocol), path)
    return base, path, protocol


def test_anchored_head_is_loaded_as_change_not_misread_as_absolute(tmp_path):
    base, path, _ = sample(tmp_path)
    original = copy.deepcopy(base)
    model = load_anchored_decoder(base, path, parent_sha256='a'*64, train_ids=[1, 2])
    x = torch.ones(1, 128, 23)*2; a = torch.ones_like(x, dtype=torch.bool)
    out = model(x, a)
    torch.testing.assert_close(out['forecast'], torch.asinh(torch.full((1, 25, 23), 2.)))
    reference = PersistentFfnModel(original)(x, a)
    torch.testing.assert_close(out['logit'], reference['logit'], rtol=0, atol=0)
    assert not model.memory.any()


@pytest.mark.parametrize('bad', ['parent', 'east', 'kind', 'overlap', 'roles'])
def test_incompatible_or_target_trained_decoder_is_rejected(tmp_path, bad):
    base, path, protocol = sample(tmp_path)
    artifact = torch.load(path, weights_only=False)
    if bad == 'parent': protocol['parent_sha256'] = 'b'*64
    if bad == 'east': protocol['no_east_reads'] = False
    if bad == 'kind': protocol['decoder_kind'] = 'absolute'
    if bad == 'overlap': protocol['selection_ids'] = [1]
    if bad == 'roles': protocol['source_train_ids'] = [1, 3]
    artifact['protocol'] = protocol
    torch.save(artifact, path)
    with pytest.raises(ValueError, match='source'):
        load_anchored_decoder(base, path, parent_sha256='a'*64, train_ids=[1, 2])

