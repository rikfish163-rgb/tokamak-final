import json

import pytest

from experiments.risk_aligned_online.export_source_gate import export
from experiments.risk_aligned_online.source_gate import qualify_source
from experiments.risk_aligned_online.tests.test_source_gate import fixture


def test_export_uses_static_test_not_intra_and_preserves_checkpoint(tmp_path):
    checkpoint=fixture(tmp_path)
    original=checkpoint.read_bytes()
    report=json.loads((tmp_path/'test220.json').read_text())
    result=dict(split='test',checkpoint_sha256=report['provenance']['checkpoint_sha256'],
                arms=dict(static=dict(metrics=report['jtext']),intra=dict(metrics={})))
    (tmp_path/'test220.json').unlink()
    (tmp_path/'hashes.json').unlink()
    paired=tmp_path/'paired.json'
    paired.write_text(json.dumps(result))
    export(checkpoint,paired,expected_ids=list(range(220)))
    assert qualify_source(checkpoint)['qualified']
    assert checkpoint.read_bytes()==original
    with pytest.raises(FileExistsError):
        export(checkpoint,paired,expected_ids=list(range(220)))


@pytest.mark.parametrize('change',[{'split':'val'},{'checkpoint_sha256':'wrong'}])
def test_export_rejects_wrong_split_or_model(tmp_path,change):
    checkpoint=tmp_path/'source.pt'
    checkpoint.write_bytes(b'model')
    report=tmp_path/'paired.json'
    payload=dict(split='test',checkpoint_sha256='wrong')
    payload.update(change)
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        export(checkpoint,report,expected_ids=list(range(220)))
