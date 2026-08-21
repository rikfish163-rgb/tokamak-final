#!/usr/bin/env python3
"""Audit context-length decodability for the deployed source lineage."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path(
    os.environ.get("FUSION_PROJECT_ROOT", Path(__file__).resolve().parents[4])
).expanduser().resolve()
HERE = ROOT / "experiments/transfer_methods_2026"
PACKAGE = HERE / "paper_grade_package"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PACKAGE / "scripts"))

import grid_search as GS  # noqa: E402
import rebuilt_harness as RH  # noqa: E402
import train_jtext_source_transformer as BASE  # noqa: E402
import train_source_padfix as PADFIX  # noqa: E402


def main() -> None:
    roles = BASE.load_canonical_jtext_roles(ROOT)
    train = BASE.load_jtext_shots(roles["train"])
    validation = BASE.load_jtext_shots(roles["val"])
    center, scale = GS.fit_scale(train)
    GS.apply_scale(train, center, scale)
    GS.apply_scale(validation, center, scale)
    evidence = {}
    for seed in (131, 133, 137):
        checkpoint_path = HERE / f"source_transformer_runs/seed{seed}/source_checkpoint.pt"
        encoder, head, checkpoint = GS.load_source_checkpoint(
            checkpoint_path,
            channels=len(RH.REBUILD),
            expected_seed=seed,
            device="cuda",
        )
        evidence[str(seed)] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": BASE.sha256_file(checkpoint_path),
            "audit": PADFIX._validation_context_audit(
                encoder,
                head,
                train,
                validation,
                threshold=float(checkpoint["validation_threshold"]),
                device="cuda",
            ),
        }
    output = PACKAGE / "evidence"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "SOURCE_CONTEXT_SHORTCUT_AUDITED",
        "split": "TRAIN1353 probe fit; VAL160 audit; TEST220 unread by audit",
        "probe": "multinomial linear probe over same-endpoint 32/64/96/128-sample views",
        "gate": {
            "paired_context_probe_macro_auc_max": 0.65,
            "paired_context_mean_abs_risk_difference_max": 0.05,
            "paired_context_alarm_flip_rate_max": 0.05,
        },
        "seeds": evidence,
    }
    path = output / "source_context_shortcut_audit.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
