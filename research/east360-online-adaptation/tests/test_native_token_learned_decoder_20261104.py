import hashlib
import json

import pytest

from experiments.risk_aligned_online.native_token_probe_20261104 import run


def test_legacy_decoder_source_remains_explicit_and_unchanged():
    path, digest, bound = run.decoder_source(None, "forecast_only")
    assert path == run.PRIOR and digest == run.PRIOR_SHA and bound == {}


def test_fitted_decoder_requires_verified_source_and_artifact(tmp_path):
    payload = b"fixture weights bytes, not a real model"
    (tmp_path / "heads.pt").write_bytes(payload)
    completion = dict(status="TRAINED_NOT_VALIDATED", artifact_sha256={
        "heads.pt": hashlib.sha256(payload).hexdigest()})
    (tmp_path / "completed.json").write_text(json.dumps(completion))
    (tmp_path / "protocol.json").write_text(json.dumps(dict(
        source_sha256=run.PARENT_SHA256, source_parameter_updates=0,
        east_signal_reads=0, test_signal_reads=0)))
    path, digest, bound = run.decoder_source(tmp_path, "anticipation125")
    assert path == tmp_path / "heads.pt" and digest == completion["artifact_sha256"]["heads.pt"]
    assert len(bound) == 2
    (tmp_path / "heads.pt").write_bytes(payload + b"changed")
    with pytest.raises(ValueError, match="artifact"):
        run.decoder_source(tmp_path, "anticipation125")


@pytest.mark.parametrize("fit,key", [(None, "anticipation125"), ("missing", "forecast_only")])
def test_decoder_identity_is_not_silently_substituted(fit, key):
    with pytest.raises(ValueError, match="decoder"):
        run.decoder_source(fit, key)
