"""Independent arithmetic/provenance audit, without using the guard-rule helper."""
import argparse
import hashlib
from pathlib import Path
import zipfile

from .. import io23
from ..east_context_operator_20261031.run import read
from ..simple_online import _durable_json
from ..simple_train import sha
from ..slow_fast_attribution_20261031.run import REFERENCE


def main(root):
    output = root/'independent_audit.json'
    if output.exists():
        raise FileExistsError(output)
    p, done = read(root/'protocol.json'), read(root/'completed.json')
    assert done['status'] == 'COMPLETED'
    assert sha(root/'rows.json') == done['rows_sha256']
    assert sha(REFERENCE/'completed.json') == p['original_completed_sha256']
    assert sha(Path(p['state_parent'])/'completed.json') == p['state_completed_sha256']
    with zipfile.ZipFile(root/'code_snapshot.zip') as archive:
        for path, digest in p['code_sha256'].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == digest
    rows = read(root/'rows.json')
    ids = [r['shot'] for r in read(REFERENCE/'order.json')]
    receipts = read(REFERENCE/'receipts.json')
    expected = [i+1 for i in range(20,360) if receipts[i]['updates']['both']['accepted']]
    assert expected == [r['rank'] for r in rows] == p['selected_ranks']
    total = {'original_pass':0, 'dual_absolute_pass':0, 'paired_pass':0}
    input_rows = read(root/'inputs.json')
    for row in rows:
        index = row['rank']-1
        assert row['shot'] == ids[index]
        assert row['history_ids'] == ids[index-8:index]
        assert all(input_rows[str(sid)]['rank'] < row['rank'] for sid in row['history_ids'])
        old = receipts[index]['updates']['both']
        assert row['losses']['before']['on'] == old['guard_before']
        assert row['losses']['after']['on'] == old['guard_after']
        a, b = row['losses']['before'], row['losses']['after']
        keys = set(a['on'])
        assert keys == set(a['off']) == set(b['on']) == set(b['off'])
        original = all(b['on'][g] <= a['on'][g]+1e-7 for g in keys)
        dual = original and all(b['off'][g] <= a['off'][g]+1e-7 for g in keys)
        paired = dual and all(b['on'][g]-b['off'][g] <= a['on'][g]-a['off'][g]+1e-7 for g in keys)
        for name, result in zip(total, (original,dual,paired), strict=True):
            assert result == row['rules'][name]
            total[name] += int(result)
        assert row == read(root/'rows'/f"{row['rank']:04d}_{row['shot']}.json")
    assert total == {name:done['summary'][name] for name in total}
    for sid, record in input_rows.items():
        assert sha(io23.path_for('EAST',int(sid))) == record['raw_sha256']
    _durable_json(output, dict(status='PASS', accepted_updates=len(rows), rules=total,
        exact_original_on_losses=2*len(rows), past_only_history_ids_verified=True,
        distinct_history_raw_files_checked=len(input_rows), code_archive_verified=True,
        off_losses_not_independently_neurally_recomputed=True, arithmetic_helper_not_used=True,
        completed_sha256=sha(root/'completed.json'), verifier_sha256=sha(Path(__file__))))
    print(read(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
