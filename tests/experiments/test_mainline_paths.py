from pathlib import Path

import pytest

from experiments.transfer_methods_2026.mainline_paths import (
    PROJECT_ROOT,
    assert_project_workspace_path,
    resolve_artifact_root,
)


def test_legacy_local_run_root_is_rejected():
    with pytest.raises(RuntimeError, match="legacy local artifact root"):
        resolve_artifact_root("/home/hetaisheng/fuison_ext4_runs")


def test_any_home_artifact_root_is_rejected():
    with pytest.raises(RuntimeError, match="under /home"):
        resolve_artifact_root("/home/hetaisheng/another-run-root")


def test_dev_shm_is_not_a_final_artifact_root():
    with pytest.raises(RuntimeError, match="scratch space"):
        resolve_artifact_root("/dev/shm/fuison-test-artifacts")


def test_other_mount_is_not_accepted_as_project_artifact_root(tmp_path):
    # The test creates a local temporary path only to exercise the ancestry
    # guard; the call must fail before any write is attempted.
    with pytest.raises(RuntimeError, match="mounted project workspace"):
        resolve_artifact_root(tmp_path / "foreign-mount-run")


def test_project_data_root_must_exist_on_project_mount():
    with pytest.raises(FileNotFoundError):
        assert_project_workspace_path(PROJECT_ROOT / "path-that-cannot-exist")
