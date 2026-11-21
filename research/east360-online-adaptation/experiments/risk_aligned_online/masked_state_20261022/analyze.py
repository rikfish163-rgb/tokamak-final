"""Independent source/diagnostic recount; no training or threshold selection."""
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23
from ..event_adapt_20261022.train import load_source
from ..fixed_test_protocol import load_test_contract
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_train import score, sha
from ..simple_online import _durable_json
from .probe import load_source_candidate, summary


def main():
    torch.set_num_threads(2)
    root = Path(__file__).parents[1]/'runs'
    contract = load_test_contract('J-TEXT')
    test = load_source(contract['shot_ids'], io23.advanced_times())
    role_val = set(io23.source_roles()['val'])
    val = {s['shot']:s for s in load_source(sorted(role_val), io23.advanced_times())}
    report = dict(status='VERIFIED_RECOUNT', parent_unchanged=sha(DEFAULT_PARENT) == PARENT_SHA256, arms={})
    if not report['parent_unchanged']: raise AssertionError('parent file changed')
    for mode in ('none','real'):
        run = root/f'masked_state_{mode}_20261022'
        receipt = json.loads((run/'completed.json').read_text())
        config = json.loads((run/'config.json').read_text())
        for path, digest in config['code_sha256'].items():
            if sha(io23.ROOT/path) != digest: raise AssertionError(f'training implementation changed: {path}')
        traces = np.load(run/'test220_traces.npz')
        ids = {int(k.split('_')[0]) for k in traces.files}
        if ids != set(contract['shot_ids']): raise AssertionError('fixed TEST220 IDs changed')
        threshold = receipt['metrics']['threshold']
        tp = fp = nd = ns = 0
        for s in test:
            times, scores = traces[f'{s["shot"]}_times'], traces[f'{s["shot"]}_scores']
            if not np.array_equal(times, s['times'][s['valid']]): raise AssertionError('HDF5 time mismatch')
            if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any(): raise AssertionError('invalid probabilities')
            idx = np.flatnonzero(scores >= threshold)
            if s['is_disrupt']:
                nd += 1
                if len(idx) and round((s['down']-times[idx[0]])*1000., 9) > 10.: tp += 1
            else:
                ns += 1; fp += bool(len(idx))
        if nd != 110 or ns != 110: raise AssertionError('fixed class count mismatch')
        if tp != receipt['metrics']['tp'] or fp != receipt['metrics']['fp']: raise AssertionError('independent count mismatch')
        qualified = tp/nd >= .9 and fp/ns < .2
        if qualified != receipt['source_qualified']: raise AssertionError('source qualification mismatch')
        model, _, _ = load_source_candidate(run/'source.pt', 'cuda')
        reproduced = score(model, test, 'cuda')
        difference = max(float(np.max(np.abs(r['scores']-traces[f'{r["shot"]}_scores']))) for r in reproduced)
        if difference > 2e-6: raise AssertionError('source curve reproduction failed')
        probe = root/f'masked_state_{mode}_probe_20261022'
        protocol = json.loads((probe/'protocol.json').read_text())
        probe_receipt = json.loads((probe/'completed.json').read_text())
        for name, digest in protocol['code_sha256'].items():
            if sha(Path(__file__).parent/name) != digest: raise AssertionError('probe implementation changed')
        total = 0
        for rate in protocol['learning_rates']:
            raw = json.loads((probe/f'rate_{rate}.json').read_text())
            rows = raw['records']; total += len(rows)
            if any(r['shot'] not in role_val for r in rows): raise AssertionError('non-VAL diagnostic')
            for r in rows:
                s, end = val[r['shot']], r['end']
                if not 0 <= end < len(s['times']) or not s['valid'][end]:
                    raise AssertionError('invalid diagnostic endpoint')
                if r['y'] != int(s['y'][end] > .5):
                    raise AssertionError('diagnostic label mismatch')
                if s['is_disrupt'] and round((s['down']-s['times'][end])*1000., 9) <= 10.:
                    raise AssertionError('non-timely source diagnostic endpoint')
                for side in ('before','after'):
                    expected = float(np.logaddexp(0., r[f'{side}_logit'])-r['y']*r[f'{side}_logit'])
                    if abs(expected-r[f'{side}_bce']) > 1e-12: raise AssertionError('stored BCE mismatch')
            if summary(rows) != raw['summary']: raise AssertionError('probe aggregation mismatch')
            m = raw['summary']
            gate = m['clean']['after_bce'] <= m['clean']['before_bce'] and m['shift']['after_bce'] < m['shift']['before_bce']
            if gate != raw['probe_gate']: raise AssertionError('probe gate mismatch')
        if probe_receipt['allow_east_candidate']: raise AssertionError('failed source was promoted')
        report['arms'][mode] = dict(tp=tp, fp=fp, n_disrupt=nd, n_safe=ns,
            tpr=tp/nd, fpr=fp/ns, qualification=qualified, replay_max_abs_difference=difference,
            checkpoint_sha256=sha(run/'source.pt'), traces_sha256=sha(run/'test220_traces.npz'),
            probe_records_recounted=total, all_probe_gates_failed=not any(r['probe_gate'] for r in probe_receipt['results']),
            east_used=False)
    destination = io23.ROOT/'outputs/source_event_rebuild_audit_20261022/masked_state_summary.json'
    _durable_json(destination, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
