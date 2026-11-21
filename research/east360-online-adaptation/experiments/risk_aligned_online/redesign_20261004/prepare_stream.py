"""Save a label-blind EAST order; join outcomes ONLY for separate composition audit."""
import argparse
import json
from pathlib import Path

from .. import io23
from ..simple_online import _durable_json
from .stream_manifest import build_manifest, manifest_bytes


def save_stream(output, *, terminal_slack_ms=2.,rate_contract='strict'):
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    root=io23.DATA_ROOT/'EAST'/'unified_hdf5'
    order, audit=build_manifest(roots=[root],order='shot_number',
        compatible_only=True,offline_outcomes=True,terminal_slack_ms=terminal_slack_ms,
        rate_contract=rate_contract)
    # The order and selection are already finalized. The outcome join below
    # cannot choose, discard or reorder a shot; it is not passed to the runner.
    labels=[io23.read_outcome('EAST',row['shot'])[0] for row in order]
    def counts(start,stop):
        selected=labels[start:stop]
        return dict(first_index=start+1,last_index=start+len(selected),
                    n_disrupt=sum(selected),n_safe=len(selected)-sum(selected))
    summary=dict(total=counts(0,len(order)),first30=counts(0,30),
                 first240=counts(0,240),after30=counts(30,len(order)),
                 interpretation='offline composition only; not input to selection/policy',
                 ordering='ascending shot number proxy, not verified damage ranking')
    output.mkdir(parents=True,exist_ok=False)
    # Canonical bytes keep helper's manifest SHA equal to the on-disk order SHA.
    (output/'order.json').write_bytes(manifest_bytes(order))
    (output/'roster.json').write_bytes(manifest_bytes(audit['roster']))
    _durable_json(output/'inventory.json',audit)
    _durable_json(output/'composition.json',summary)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rate-contract',choices=('strict','nominal'),default='strict')
    p.add_argument('--terminal-slack-ms',type=float,default=2.,
                   help='measured end-of-record gap only; original samples and DownTime unchanged')
    args=p.parse_args(argv)
    result=save_stream(args.output,terminal_slack_ms=args.terminal_slack_ms,
                       rate_contract=args.rate_contract)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
