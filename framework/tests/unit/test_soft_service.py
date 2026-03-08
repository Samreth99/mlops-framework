"""
Unit tests for software_service — no external dependencies (no MLflow, Redis, AWS).
"""
import pytest
from mlops.dimensions.soft import software_service as svc


@pytest.fixture(autouse=True)
def reset_store(tmp_path, monkeypatch):
    # Point the registry file to a temp path so tests are isolated
    registry_file = tmp_path / "soft_registry.json"
    monkeypatch.setattr(svc, "_REGISTRY_FILE", registry_file)
    yield


# ── Repo ──────────────────────────────────────────────────────────
def test_create_repo():
    repo = svc.create_repo(name="my-repo", url="https://github.com/x/y", owner="x")
    assert repo["repoId"].startswith("repo-")
    assert repo["name"] == "my-repo"
    assert repo["owner"] == "x"


def test_list_repos_empty():
    assert svc.list_repos() == []


def test_list_repos_returns_created():
    svc.create_repo(name="r1", url="u1", owner="o1")
    svc.create_repo(name="r2", url="u2", owner="o2")
    assert len(svc.list_repos()) == 2


# ── Build ─────────────────────────────────────────────────────────
def test_trigger_build():
    result = svc.trigger_build(
        repo_id="r1", build_config_ref="Dockerfile",
        ticket_id="tkt-001", commit_sha="abc123",
    )
    assert result["buildId"].startswith("build-")
    assert result["status"] == "QUEUED"


def test_update_build_status():
    b = svc.trigger_build(
        repo_id="r1", build_config_ref="Dockerfile",
        ticket_id="tkt-001", commit_sha="abc123",
    )
    svc.update_build(b["buildId"], status="BUILT", image_ref="ecr://repo:abc123")
    updated = svc.get_build(b["buildId"])
    assert updated["status"] == "BUILT"
    assert updated["environment"]["imageRef"] == "ecr://repo:abc123"


def test_get_build_not_found():
    assert svc.get_build("nonexistent") is None


# ── Package ───────────────────────────────────────────────────────
def test_create_package():
    b = svc.trigger_build(
        repo_id="r1", build_config_ref="Dockerfile", ticket_id="tkt-001",
    )
    pkg = svc.create_package(
        package_type="image", digest="sha256:abc", version="1.0.0",
        build_id=b["buildId"], storage_ref="ecr://mlops:1.0.0",
    )
    assert pkg["packageId"].startswith("pkg-")
    assert pkg["type"] == "image"
    assert pkg["storageRef"] == "ecr://mlops:1.0.0"


def test_get_package_not_found():
    assert svc.get_package("nonexistent") is None


# ── Test run ──────────────────────────────────────────────────────
def test_trigger_test_run():
    run = svc.trigger_test(test_suite_ref="tests/unit.yaml", ticket_id="tkt-001")
    assert run["testRunId"].startswith("testrun-")
    assert run["status"] == "QUEUED"


def test_update_test_run_passed():
    run = svc.trigger_test(test_suite_ref="tests/unit.yaml", ticket_id="tkt-001")
    svc.update_test_run(run["testRunId"], status="PASSED", passed=True)
    updated = svc.get_test_run(run["testRunId"])
    assert updated["status"] == "PASSED"
    assert updated["passed"] is True


# ── Release ───────────────────────────────────────────────────────
def test_create_release():
    b = svc.trigger_build(
        repo_id="r1", build_config_ref="Dockerfile", ticket_id="tkt-001",
    )
    pkg = svc.create_package(
        package_type="image", digest="sha256:abc", version="1.0.0",
        build_id=b["buildId"], storage_ref="ecr://x:1.0",
    )
    rel = svc.create_release(
        package_id=pkg["packageId"], release_notes="First release", ticket_id="tkt-001",
    )
    assert rel["releaseId"].startswith("release-")
    assert rel["status"] == "PENDING"


def test_update_release_status():
    b = svc.trigger_build(
        repo_id="r1", build_config_ref="Dockerfile", ticket_id="tkt-001",
    )
    pkg = svc.create_package(
        package_type="image", digest="sha256:abc", version="1.0.0",
        build_id=b["buildId"], storage_ref="ecr://x:1.0",
    )
    rel = svc.create_release(
        package_id=pkg["packageId"], release_notes="", ticket_id="tkt-001",
    )
    svc.update_release(rel["releaseId"], status="PUBLISHED")
    updated = svc.get_release(rel["releaseId"])
    assert updated["status"] == "PUBLISHED"
