import hashlib
import json

import pytest

from experiments.risk_aligned_online.input_factors_20261031.integrity import (
    verify_archived_controls,
    verify_local_seals,
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    data = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(data)
    return digest(data)


@pytest.fixture
def archive(tmp_path):
    protocol = b'fixed protocol'
    barrier = b'fixed barrier'
    (tmp_path / 'protocol.json').write_bytes(protocol)
    (tmp_path / 'all_predictions_sealed.json').write_bytes(barrier)
    complete = dict(complete=True, artifact_sha256={
        'protocol.json': digest(protocol),
        'all_predictions_sealed.json': digest(barrier),
    })
    complete_sha = write_json(tmp_path / 'complete.json', complete)
    return tmp_path, complete, complete_sha


def test_archive_accepts_matching_complete_and_advertised_controls(archive):
    root, _, complete_sha = archive
    verify_archived_controls(root, complete_sha)


def test_complete_hash_checked_before_control_files(archive):
    root, _, _ = archive
    (root / 'protocol.json').unlink()
    with pytest.raises(ValueError, match='complete.json'):
        verify_archived_controls(root, '0' * 64)


@pytest.mark.parametrize('name', ['protocol.json', 'all_predictions_sealed.json'])
def test_archive_rejects_tampered_controls(archive, name):
    root, _, complete_sha = archive
    (root / name).write_bytes(b'tampered')
    with pytest.raises(ValueError, match=name):
        verify_archived_controls(root, complete_sha)


@pytest.mark.parametrize('name', ['complete.json', 'protocol.json', 'all_predictions_sealed.json'])
def test_archive_rejects_missing_files(archive, name):
    root, _, complete_sha = archive
    (root / name).unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        verify_archived_controls(root, complete_sha)


def test_archive_rejects_missing_advertised_hash(archive):
    root, complete, _ = archive
    del complete['artifact_sha256']['protocol.json']
    complete_sha = write_json(root / 'complete.json', complete)
    with pytest.raises((KeyError, ValueError)):
        verify_archived_controls(root, complete_sha)


def test_archive_rejects_incomplete_marker(archive):
    root, complete, _ = archive
    complete['complete'] = False
    complete_sha = write_json(root / 'complete.json', complete)
    with pytest.raises(ValueError):
        verify_archived_controls(root, complete_sha)


@pytest.fixture
def local_receipts(tmp_path):
    shots = tmp_path / 'shots'
    shots.mkdir()
    receipts = []
    for index, shot in enumerate((71304, 71265, 71415), 1):
        stem = f'{index:04d}_{shot}'
        data = f'prediction {shot}'.encode()
        (shots / f'{stem}.npz').write_bytes(data)
        seal = dict(index=index, shot=shot, sha256=digest(data))
        seal_sha = write_json(shots / f'{stem}_seal.json', seal)
        receipts.append({**seal, 'seal_sha256': seal_sha})
    return tmp_path, receipts


def test_verify_every_matching_prediction_and_seal(local_receipts):
    root, receipts = local_receipts
    verify_local_seals(root, receipts)


def test_rejects_tampered_last_prediction(local_receipts):
    root, receipts = local_receipts
    (root / 'shots/0003_71415.npz').write_bytes(b'tampered last shot')
    with pytest.raises(ValueError, match='0003_71415'):
        verify_local_seals(root, receipts)


def test_rejects_tampered_seal(local_receipts):
    root, receipts = local_receipts
    (root / 'shots/0003_71415_seal.json').write_bytes(b'tampered seal')
    with pytest.raises(ValueError, match='0003_71415'):
        verify_local_seals(root, receipts)


@pytest.mark.parametrize('name', ['0003_71415.npz', '0003_71415_seal.json'])
def test_rejects_missing_prediction_or_seal(local_receipts, name):
    root, receipts = local_receipts
    (root / 'shots' / name).unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        verify_local_seals(root, receipts)


def test_rejects_legacy_receipt_without_seal_hash(local_receipts):
    root, receipts = local_receipts
    del receipts[-1]['seal_sha256']
    with pytest.raises((KeyError, ValueError)):
        verify_local_seals(root, receipts)


def test_rejects_seal_contents_inconsistent_with_receipt(local_receipts):
    root, receipts = local_receipts
    last = receipts[-1]
    seal_path = root / 'shots/0003_71415_seal.json'
    seal = dict(index=last['index'], shot=1, sha256=last['sha256'])
    last['seal_sha256'] = write_json(seal_path, seal)
    with pytest.raises(ValueError):
        verify_local_seals(root, receipts)


@pytest.mark.parametrize('mutation', ['empty', 'duplicate', 'bad_index', 'bad_digest'])
def test_rejects_invalid_receipts(local_receipts, mutation):
    root, receipts = local_receipts
    if mutation == 'empty':
        receipts = []
    elif mutation == 'duplicate':
        receipts.append(receipts[-1])
    elif mutation == 'bad_index':
        receipts[-1]['index'] = '../3'
    else:
        receipts[-1]['seal_sha256'] = 'not-a-sha256'
    with pytest.raises((KeyError, TypeError, ValueError)):
        verify_local_seals(root, receipts)
