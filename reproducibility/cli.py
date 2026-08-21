from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reproducibility.data_contract import DataContractError, audit_data
from reproducibility.paths import data_root, project_root
from reproducibility.reproduce import SEEDS, run_reproduction
from reproducibility.verify_results import compare_multiseed_results


def _write_or_print(payload: dict, output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fusion-reproduce")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="validate data layout and HDF5 contract")
    audit.add_argument("--data-root", type=Path, default=None)
    audit.add_argument("--full", action="store_true")
    audit.add_argument(
        "--content-hash", action="store_true",
        help="SHA-256 every required HDF5 file and emit a per-file manifest",
    )
    audit.add_argument(
        "--portable", action="store_true",
        help="replace the machine-local data root in JSON with a portable token",
    )
    audit.add_argument("--output", type=Path, default=None)

    smoke = sub.add_parser("smoke", help="check data, CUDA and frozen references")
    smoke.add_argument("--data-root", type=Path, default=None)

    reproduce = sub.add_parser("reproduce", help="run the locked three-stage pipeline")
    reproduce.add_argument("--data-root", type=Path, default=None)
    reproduce.add_argument("--output", type=Path, required=True)
    reproduce.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    reproduce.add_argument("--dry-run", action="store_true")

    verify = sub.add_parser("verify", help="compare a run with reference metrics")
    verify.add_argument("--actual", type=Path, required=True)
    verify.add_argument(
        "--expected", type=Path,
        default=(
            project_root()
            / "results/reference/paper_v3c_multiseed_summary.json"
        ),
    )
    verify.add_argument("--tolerance", type=float, default=1e-8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "audit":
            payload = audit_data(
                args.data_root or data_root(), full=args.full,
                content_hash=args.content_hash,
                portable=args.portable,
            )
            _write_or_print(payload, args.output)
            return 0
        if args.command == "smoke":
            import torch

            selected = (args.data_root or data_root()).resolve()
            payload = {
                "data": audit_data(selected, full=False),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
                "reference_present": (
                    project_root()
                    / "results/reference/paper_v3c_multiseed_summary.json"
                ).is_file(),
            }
            payload["passed"] = (
                payload["cuda_available"] and payload["reference_present"]
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload["passed"] else 2
        if args.command == "reproduce":
            payload = run_reproduction(
                output_root=args.output,
                seeds=tuple(args.seeds),
                selected_data_root=args.data_root,
                dry_run=args.dry_run,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "verify":
            actual = json.loads(args.actual.read_text())
            expected = json.loads(args.expected.read_text())
            payload = compare_multiseed_results(
                actual, expected, tolerance=args.tolerance,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload["passed"] else 3
    except (DataContractError, FileExistsError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
