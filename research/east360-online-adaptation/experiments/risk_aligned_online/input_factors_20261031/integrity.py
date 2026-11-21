"""Read-only SHA256 checks for archived controls and future local seals."""
import hashlib
import json
from pathlib import Path


def _checked_bytes(path, expected):
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected)):
        raise ValueError(f'invalid SHA256 for {path}')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'SHA256 mismatch: {path}')
    return data


def verify_archived_controls(archive, expected_complete_sha256):
    """Verify the pinned complete file before its advertised control files."""
    root = Path(archive)
    complete = json.loads(_checked_bytes(root / 'complete.json', expected_complete_sha256))
    if not isinstance(complete, dict) or complete.get('complete') is not True:
        raise ValueError('archive is not marked complete')
    hashes = complete.get('artifact_sha256')
    if not isinstance(hashes, dict):
        raise ValueError('missing advertised artifact SHA256 values')
    for name in ('protocol.json', 'all_predictions_sealed.json'):
        _checked_bytes(root / name, hashes.get(name))


def verify_local_seals(root, receipts):
    """Verify each supplied NPZ and independently recorded seal SHA256.

    The caller must separately establish the expected receipt count/order.
    Missing ``seal_sha256`` is an error, never reconstructed from a file.
    """
    if not isinstance(receipts, (list, tuple)) or not receipts:
        raise ValueError('nonempty receipt list required')
    root = Path(root) / 'shots'
    seen_indices, seen_shots = set(), set()
    for receipt in receipts:
        if not isinstance(receipt, dict):
            raise ValueError('receipt must be a dictionary')
        index, shot = receipt.get('index'), receipt.get('shot')
        if (type(index) is not int or index < 1 or type(shot) is not int or shot < 1
                or index in seen_indices or shot in seen_shots):
            raise ValueError('unique positive integer index and shot required')
        seen_indices.add(index)
        seen_shots.add(shot)
        stem = f'{index:04d}_{shot}'
        _checked_bytes(root / f'{stem}.npz', receipt.get('sha256'))
        seal = json.loads(_checked_bytes(root / f'{stem}_seal.json', receipt.get('seal_sha256')))
        if (not isinstance(seal, dict) or seal.get('index') != index
                or seal.get('shot') != shot or seal.get('sha256') != receipt.get('sha256')):
            raise ValueError(f'seal contents differ from receipt: {stem}')
