"""The outcome reader must inspect the exact durable prediction directory."""
import json
import sys
from types import SimpleNamespace

import pytest

from experiments.risk_aligned_online import continual_decoder_experiment as driver


def test_driver_provider_checks_same_seal_directory_as_runner(tmp_path, monkeypatch):
    reference = tmp_path/'reference'
    reference.mkdir()
    (reference/'order.json').write_text(json.dumps([{'shot':i} for i in range(31)]))
    monkeypatch.setattr(driver, 'REFERENCE', reference)
    monkeypatch.setattr(driver, 'load_parent', lambda _: (None, dict(
        threshold=.7, source_center=None, source_scale=None)))
    fake_model = SimpleNamespace()
    fake_model.to = lambda _: fake_model
    fake_model.eval = lambda: fake_model
    monkeypatch.setattr(driver, 'DecoderFfnModel', lambda _: fake_model)
    monkeypatch.setattr(driver, 'RollingDecoder', lambda *a, **k: None)
    monkeypatch.setattr(driver, 'EndedDensityReplay', lambda *a, **k: SimpleNamespace(**k))
    class CheckedBeforeFirstRead(Exception):
        pass
    def check_runner(model, *, output_dir, data_api, **kwargs):
        assert data_api.output_dir == output_dir
        raise CheckedBeforeFirstRead
    monkeypatch.setattr(driver, 'run_arm', check_runner)
    monkeypatch.setattr(sys, 'argv', ['driver', '--output', str(tmp_path/'arm'),
                                   '--arm', 'early_continual', '--limit', '31'])
    with pytest.raises(CheckedBeforeFirstRead):
        driver.main()
