"""Seal one fixed Ip-only order, then append descriptive class composition."""
import json
from pathlib import Path

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.current_order_20261031.order import order_stream
from experiments.risk_aligned_online.simple_online import _durable_json
from experiments.risk_aligned_online.simple_train import sha


def main():
    root = Path(__file__).parent
    parent = root/'experiments/risk_aligned_online/runs/lifecycle_context8360_service_20261028'
    old = json.loads((parent/'order.json').read_text())
    ids = [r['shot'] for r in old]
    assert len(ids)==len(set(ids))==360
    out = root/'probe_outputs/current_order_manifest_v1'
    if out.exists():
        raise FileExistsError(out)
    rows = order_stream(sorted(ids))
    assert {r['shot'] for r in rows}==set(ids)
    out.mkdir()
    _durable_json(out/'order.json',rows)
    _durable_json(out/'protocol.json',dict(
        scope='DEVELOPMENT_OFFLINE_ORDER_CONTROL_NOT_MEASURED_DAMAGE',
        selection='same 360 identities; Ip-only ascending; shot-ID tie',
        order_sha256=sha(out/'order.json'),old_order_sha256=sha(parent/'order.json'),
        plan_sha256=sha(root/'CURRENT_ORDER_PLAN_20261031.md'),
        raw_prefix_reader='data/ip only, first 100ms of recorded signal',
        sorting_used_outcomes=False,sorting_used_predictions=False,
        analyst_composition_previously_exposed_by_history_audit=True))
    # This descriptive sidecar is never passed to the online learner.
    flags = {r['shot']:io23.read_outcome('EAST',r['shot'])[0] for r in rows}
    summary = {str(n):dict(n=n,disrupt=sum(flags[r['shot']] for r in rows[:n]),
                          safe=sum(not flags[r['shot']] for r in rows[:n])) for n in (20,40,60,80,360)}
    _durable_json(out/'composition_after_order.json',dict(order_sha256=sha(out/'order.json'),
        summaries=summary,first20_ids=[r['shot'] for r in rows[:20]],
        first20_safe_ids=[r['shot'] for r in rows[:20] if not flags[r['shot']]],
        flags=flags,label_authority='HDF5, not externally adjudicated'))
    print(dict(order_sha256=sha(out/'order.json'),composition=summary))


if __name__=='__main__':
    main()
