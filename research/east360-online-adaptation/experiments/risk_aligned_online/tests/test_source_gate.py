import hashlib
import json

import pytest

from experiments.risk_aligned_online.source_gate import qualify_source


def fixture(tmp_path, tp=99, fp=21):
    checkpoint = tmp_path/'source.pt'
    checkpoint.write_bytes(b'unit test checkpoint, not a torch model')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    predictions = [dict(shot=i, is_disrupt=True, alarm=i<tp,
                        timely=i<tp, lead_ms=11. if i<tp else None) for i in range(110)]
    predictions += [dict(shot=110+i, is_disrupt=False, alarm=i<fp,
                         timely=False, lead_ms=None) for i in range(110)]
    report = dict(evidence='DEVELOPMENT_ONLY',
        provenance=dict(checkpoint_sha256=digest, selected_test_count=220),
        jtext=dict(tpr=tp/110, fpr=fp/110, tp=tp, fp=fp, n_disrupt=110,
                   n_safe=110, predictions=predictions))
    (tmp_path/'test220.json').write_text(json.dumps(report))
    (tmp_path/'hashes.json').write_text(json.dumps(dict(source_pt_sha256=digest,
        test220_sha256=hashlib.sha256((tmp_path/'test220.json').read_bytes()).hexdigest())))
    return checkpoint


def test_full_matched_source_passes(tmp_path):
    assert qualify_source(fixture(tmp_path))['qualified']


@pytest.mark.parametrize('tp,fp', [(98, 0), (99, 22)])
def test_low_tpr_or_exact_fpr_cap_fails(tmp_path, tp, fp):
    assert not qualify_source(fixture(tmp_path, tp, fp))['qualified']


def test_modified_artifact_is_not_accepted(tmp_path):
    checkpoint = fixture(tmp_path)
    checkpoint.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='hash'):
        qualify_source(checkpoint)


def test_no_report_cannot_prove_qualification(tmp_path):
    checkpoint = tmp_path/'source.pt'
    checkpoint.write_bytes(b'checkpoint')
    with pytest.raises(FileNotFoundError):
        qualify_source(checkpoint)
