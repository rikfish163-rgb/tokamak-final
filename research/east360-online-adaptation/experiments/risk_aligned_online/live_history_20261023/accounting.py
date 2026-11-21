"""Separate actual learning steps from no-op forward-evaluation differences.

Original run receipts stay immutable. Their loss_after was computed in
no_grad mode, unlike loss_before; a decrease alone is not a parameter update.
This supplemental audit counts learning only when the recorded parameter
change is nonzero. It does not change predictions or the acceptance policy.
"""
import argparse
import json
import math
from pathlib import Path

from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import ARMS


def summarize_updates(rows):
    result = dict(proposals=0, nonzero_updates=0, zero_updates=0,
                  raw_forward_decreases=0, nonzero_update_decreases=0,
                  zero_update_nonzero_loss_difference=0,
                  zero_update_max_abs_loss_difference=0.)
    for row in rows:
        norm = row['parameter_delta_norm']
        before, after = row['loss_before'], row['loss_after']
        if not all(math.isfinite(x) for x in (norm, before, after)) or norm < 0:
            raise ValueError('finite losses and nonnegative parameter change required')
        updated = norm > 0
        if row['gradient_steps'] != int(updated):
            raise ValueError('update counter must agree with actual parameter change')
        result['proposals'] += 1
        result['raw_forward_decreases'] += int(after < before)
        if updated:
            result['nonzero_updates'] += 1
            result['nonzero_update_decreases'] += int(after < before)
        else:
            result['zero_updates'] += 1
            result['zero_update_nonzero_loss_difference'] += int(after != before)
            result['zero_update_max_abs_loss_difference'] = max(
                result['zero_update_max_abs_loss_difference'], abs(after-before))
    return result


def main(root):
    output = root/'update_accounting.json'
    if output.exists():
        raise FileExistsError(output)
    read = lambda path: json.loads(path.read_text())
    verified = read(root/'verification.json')
    assert verified['status'] == 'VERIFIED'
    assert verified['completed_sha256'] == sha(root/'completed.json')
    seals = read(root/'prediction_receipts.json')
    rows = {arm: [] for arm in ARMS}
    for seal in seals:
        path = root/'shots'/f"{seal['shot']}_updates.json"
        assert sha(path) == seal['updates_sha256']
        updates = read(path)
        for arm in ARMS:
            # This experiment has mature observed targets at every proposal;
            # do not silently exclude absent-loss rows from its denominator.
            assert all(row['observed_targets'] > 0 for row in updates[arm])
            rows[arm].extend(updates[arm])
    summaries = {arm: summarize_updates(values) for arm, values in rows.items()}
    for arm, values in summaries.items():
        prior = verified['update_summary'][arm]
        assert values['proposals'] == prior['proposals']
        assert values['nonzero_updates'] == prior['updates']
        assert values['raw_forward_decreases'] == prior['forecast_decreased']
    report = dict(status='UPDATE_ACCOUNTING_VERIFIED', summaries=summaries,
                  code_sha256=sha(Path(__file__)), verification_sha256=sha(root/'verification.json'),
                  completed_sha256=sha(root/'completed.json'),
                  caveat='No-op loss differences are not learning; original immutable receipts and TPR/FPR are unchanged.')
    _durable_json(output, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
