from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from reproducibility.prospective import (
    ProspectiveContractError,
    _excluded_shots,
    create_prospective_seal,
    verify_prospective_seal,
)
from reproducibility.cli import build_parser


ROOT = Path(__file__).resolve().parents[2]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "prospective@example.invalid")
    _git(repo, "config", "user.name", "Prospective Test")
    parent = repo / "parent.pt"
    config = repo / "method.json"
    split_root = repo / "config/splits"
    split_root.mkdir(parents=True)
    paper131 = split_root / "east_paper131_split_all_in.json"
    balanced360 = split_root / "east_hdf5_authoritative_balanced360_v1.json"
    parent.write_bytes(b"sealed-parent-v1\n")
    config.write_text('{"method":"v3c"}\n')
    paper131.write_text(json.dumps({
        "train": {"disruption_shots": [10], "non_disruption_shots": [20]},
        "test": {"disruption_shots": [30], "non_disruption_shots": [40]},
    }))
    balanced360.write_text(json.dumps({
        "test": {"disruption_shots": [50], "non_disruption_shots": [60]},
    }))
    _git(repo, "add", "parent.pt", "method.json", "config/splits")
    _git(repo, "commit", "-qm", "frozen parent")
    return repo, parent, config


def _spec(repo: Path, parent: Path, config: Path) -> dict:
    return {
        "schema_version": "prospective-east-confirmation-v1",
        "claim_status": "PROSPECTIVE_PREREGISTRATION_ONLY",
        "frozen_git_sha": _git(repo, "rev-parse", "HEAD"),
        "seeds": [131, 133, 137],
        "arms": [
            "static", "active", "no_label", "causal_random_query",
            "label_frontload", "east_only",
        ],
        "label_source": ["/meta/IsDisrupt", "/meta/DownTime"],
        "primary_metric": "TPR_at_lead_ge_30ms_minus_FPR",
        "admission_gates": {
            "tpr_at_lead_ge_30ms_min": 0.80,
            "fpr_max": 0.20,
            "median_lead_ms_min": 30.0,
            "worst_block_tpr_at_lead_ge_30ms_min": 0.80,
            "worst_block_fpr_max": 0.20,
            "absolute_shot_id_spearman_max": 0.20,
        },
        "over_3000ms_counts_as_tpr": True,
        "retuning_prohibited": True,
        "complete_stream_reporting": True,
        "prediction_order": "predict-seal-query-selected-reveal-later-shot-update",
        "target_validation_completed": True,
        "target_validation_artifact": str(config.relative_to(repo)),
        "random_schedule_artifact": str(config.relative_to(repo)),
        "execution_plan_artifact": str(config.relative_to(repo)),
        "frozen_artifacts": [
            str(parent.relative_to(repo)), str(config.relative_to(repo)),
        ],
        "excluded_shots": [],
        "excluded_shot_sources": [
            {
                "path": "config/splits/east_paper131_split_all_in.json",
                "json_pointers": [
                    "/train/disruption_shots",
                    "/train/non_disruption_shots",
                    "/test/disruption_shots",
                    "/test/non_disruption_shots",
                ],
            },
            {
                "path": "config/splits/east_hdf5_authoritative_balanced360_v1.json",
                "json_pointers": [
                    "/test/disruption_shots",
                    "/test/non_disruption_shots",
                ],
            },
        ],
    }


def test_create_and_verify_prospective_seal_is_hash_bound(tmp_path: Path) -> None:
    repo, parent, config = _repository(tmp_path)
    spec = tmp_path / "prospective.json"
    roster = tmp_path / "future_shots.txt"
    validation_roster = tmp_path / "validation_shots.txt"
    seal = tmp_path / "prospective-seal.json"
    spec.write_text(json.dumps(_spec(repo, parent, config)))
    roster.write_text("100\n101\n102\n")
    validation_roster.write_text("80\n90\n")

    payload = create_prospective_seal(
        spec_path=spec, validation_roster_path=validation_roster,
        roster_path=roster, output_path=seal, repo_root=repo,
    )

    assert payload["status"] == "SEALED_BEFORE_OUTCOME_ACCESS"
    assert payload["roster"] == {
        "count": 3,
        "first_shot": 100,
        "last_shot": 102,
        "sha256": payload["roster"]["sha256"],
    }
    assert payload["outcome_fields_present"] is False
    assert verify_prospective_seal(
        seal_path=seal, spec_path=spec, validation_roster_path=validation_roster,
        roster_path=roster, repo_root=repo,
    )["passed"] is True

    parent.write_bytes(b"changed-after-seal\n")
    verdict = verify_prospective_seal(
        seal_path=seal, spec_path=spec, validation_roster_path=validation_roster,
        roster_path=roster, repo_root=repo,
    )
    assert verdict["passed"] is False
    assert any("parent.pt" in item for item in verdict["errors"])


@pytest.mark.parametrize("roster_text", ["100\n100\n", "101\n100\n"])
def test_roster_must_be_unique_and_strictly_chronological(
    tmp_path: Path, roster_text: str,
) -> None:
    repo, parent, config = _repository(tmp_path)
    spec = tmp_path / "prospective.json"
    roster = tmp_path / "future_shots.txt"
    validation_roster = tmp_path / "validation_shots.txt"
    spec.write_text(json.dumps(_spec(repo, parent, config)))
    roster.write_text(roster_text)
    validation_roster.write_text("80\n90\n")

    with pytest.raises(ProspectiveContractError, match="strictly increasing"):
        create_prospective_seal(
            spec_path=spec, validation_roster_path=validation_roster,
            roster_path=roster,
            output_path=tmp_path / "seal.json", repo_root=repo,
        )


def test_roster_rejects_metadata_and_known_shot_overlap(tmp_path: Path) -> None:
    repo, parent, config = _repository(tmp_path)
    spec = tmp_path / "prospective.json"
    spec.write_text(json.dumps(_spec(repo, parent, config)))
    validation_roster = tmp_path / "validation_shots.txt"
    validation_roster.write_text("80\n90\n")

    metadata_roster = tmp_path / "metadata.csv"
    metadata_roster.write_text("shot,is_disrupt\n100,1\n")
    with pytest.raises(ProspectiveContractError, match="shot IDs only"):
        create_prospective_seal(
            spec_path=spec, validation_roster_path=validation_roster,
            roster_path=metadata_roster,
            output_path=tmp_path / "metadata-seal.json", repo_root=repo,
        )

    overlap_roster = tmp_path / "overlap.txt"
    overlap_roster.write_text("20\n100\n")
    with pytest.raises(ProspectiveContractError, match="overlaps excluded shots"):
        create_prospective_seal(
            spec_path=spec, validation_roster_path=validation_roster,
            roster_path=overlap_roster,
            output_path=tmp_path / "overlap-seal.json", repo_root=repo,
        )


def test_excluded_shots_can_be_bound_to_registered_json_rosters(
    tmp_path: Path,
) -> None:
    repo, parent, config = _repository(tmp_path)
    registered = repo / "registered.json"
    registered.write_text(json.dumps({"train": {"shots": [40, 50]}}))
    _git(repo, "add", "registered.json")
    _git(repo, "commit", "-qm", "register prior shots")
    payload = _spec(repo, parent, config)
    payload["excluded_shot_sources"].append({
        "path": "registered.json", "json_pointers": ["/train/shots"],
    })
    spec = tmp_path / "prospective.json"
    roster = tmp_path / "future_shots.txt"
    validation_roster = tmp_path / "validation_shots.txt"
    spec.write_text(json.dumps(payload))
    roster.write_text("50\n100\n")
    validation_roster.write_text("80\n90\n")

    with pytest.raises(ProspectiveContractError, match="overlaps excluded shots"):
        create_prospective_seal(
            spec_path=spec, validation_roster_path=validation_roster,
            roster_path=roster,
            output_path=tmp_path / "seal.json", repo_root=repo,
        )


def test_spec_cannot_drop_a_canonical_prior_east_roster(tmp_path: Path) -> None:
    repo, parent, config = _repository(tmp_path)
    payload = _spec(repo, parent, config)
    payload["excluded_shot_sources"] = payload["excluded_shot_sources"][:1]
    spec = tmp_path / "prospective.json"
    spec.write_text(json.dumps(payload))
    validation_roster = tmp_path / "validation_shots.txt"
    validation_roster.write_text("80\n90\n")
    roster = tmp_path / "future_shots.txt"
    roster.write_text("100\n101\n")

    with pytest.raises(ProspectiveContractError, match="canonical prior EAST"):
        create_prospective_seal(
            spec_path=spec,
            validation_roster_path=validation_roster,
            roster_path=roster,
            output_path=tmp_path / "seal.json",
            repo_root=repo,
        )


def test_release_template_excludes_every_registered_east_development_shot() -> None:
    template = json.loads(
        (ROOT / "configs/prospective_east_confirmation.template.json").read_text()
    )
    sources = {
        source["path"]: set(source["json_pointers"])
        for source in template["excluded_shot_sources"]
    }

    assert sources["config/splits/east_paper131_split_all_in.json"] == {
        "/train/disruption_shots",
        "/train/non_disruption_shots",
        "/test/disruption_shots",
        "/test/non_disruption_shots",
    }
    assert sources["config/splits/east_hdf5_authoritative_balanced360_v1.json"] == {
        "/test/disruption_shots",
        "/test/non_disruption_shots",
    }

    excluded, _ = _excluded_shots(template, ROOT)
    balanced = json.loads(
        (ROOT / "config/splits/east_hdf5_authoritative_balanced360_v1.json").read_text()
    )["test"]
    prior_balanced360 = set(balanced["disruption_shots"]) | set(
        balanced["non_disruption_shots"]
    )
    assert len(prior_balanced360) == 360
    assert prior_balanced360 <= excluded


def test_target_validation_roster_must_be_disjoint_from_confirmation(
    tmp_path: Path,
) -> None:
    repo, parent, config = _repository(tmp_path)
    spec = tmp_path / "prospective.json"
    validation_roster = tmp_path / "validation_shots.txt"
    confirmation_roster = tmp_path / "confirmation_shots.txt"
    spec.write_text(json.dumps(_spec(repo, parent, config)))
    validation_roster.write_text("80\n100\n")
    confirmation_roster.write_text("100\n101\n")

    with pytest.raises(ProspectiveContractError, match="validation.*confirmation"):
        create_prospective_seal(
            spec_path=spec,
            validation_roster_path=validation_roster,
            roster_path=confirmation_roster,
            output_path=tmp_path / "seal.json",
            repo_root=repo,
        )


def test_seal_requires_clean_matching_git_and_fails_if_output_exists(
    tmp_path: Path,
) -> None:
    repo, parent, config = _repository(tmp_path)
    spec = tmp_path / "prospective.json"
    roster = tmp_path / "future_shots.txt"
    validation_roster = tmp_path / "validation_shots.txt"
    output = tmp_path / "seal.json"
    spec.write_text(json.dumps(_spec(repo, parent, config)))
    roster.write_text("100\n101\n")
    validation_roster.write_text("80\n90\n")
    (repo / "untracked.txt").write_text("dirty\n")

    with pytest.raises(ProspectiveContractError, match="clean git worktree"):
        create_prospective_seal(
            spec_path=spec, validation_roster_path=validation_roster,
            roster_path=roster, output_path=output,
            repo_root=repo,
        )

    (repo / "untracked.txt").unlink()
    output.write_text("do not overwrite\n")
    with pytest.raises(FileExistsError):
        create_prospective_seal(
            spec_path=spec, validation_roster_path=validation_roster,
            roster_path=roster, output_path=output,
            repo_root=repo,
        )


def test_cli_exposes_prospective_seal_and_verify() -> None:
    parser = build_parser()
    seal = parser.parse_args([
        "prospective-seal", "--spec", "spec.json",
        "--validation-roster", "validation.txt", "--roster", "shots.txt",
        "--output", "seal.json",
    ])
    assert seal.command == "prospective-seal"
    verify = parser.parse_args([
        "prospective-verify", "--seal", "seal.json", "--spec", "spec.json",
        "--validation-roster", "validation.txt",
        "--roster", "shots.txt",
    ])
    assert verify.command == "prospective-verify"
