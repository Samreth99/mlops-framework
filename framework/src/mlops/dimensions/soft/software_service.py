"""
Software dimension service layer.

Provides a JSON-file-backed store for:
  - Code repositories, commits, and tags
  - Build executions
  - Packaged artifacts
  - Test runs
  - Software releases

All IDs are generated with uuid4. Timestamps are ISO-8601 UTC strings.
The store is persisted to soft_registry.json on every write so it survives restarts.
"""
from __future__ import annotations

import json
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
# Registry file (same folder as this module)
# ─────────────────────────────────────────────
_REGISTRY_FILE = Path(__file__).parent / "soft_registry.json"
_registry_lock = threading.Lock()

_EMPTY_STORE: Dict[str, Any] = {
    "repos": {},
    "commits": {},
    "tags": {},
    "builds": {},
    "packages": {},
    "test_runs": {},
    "releases": {},
}


def _load_registry() -> Dict[str, Any]:
    """Load registry from disk. Caller must hold _registry_lock."""
    if _REGISTRY_FILE.exists():
        try:
            with open(_REGISTRY_FILE, encoding="utf-8") as f:
                data = json.load(f)
            # Ensure all keys exist (forward-compat)
            for key in _EMPTY_STORE:
                data.setdefault(key, {})
            return data
        except (json.JSONDecodeError, OSError):
            pass
    return {k: {} for k in _EMPTY_STORE}


def _save_registry(reg: Dict[str, Any]) -> None:
    """Atomic write: write to .tmp then rename. Caller must hold _registry_lock."""
    tmp = _REGISTRY_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)
    tmp.replace(_REGISTRY_FILE)


@contextmanager
def _registry():
    """Context manager: load → yield → save (only on clean exit)."""
    with _registry_lock:
        reg = _load_registry()
        try:
            yield reg
        except Exception:
            raise
        else:
            _save_registry(reg)


# Keep a module-level reference for read-only access (no write needed)
def _read_store() -> Dict[str, Any]:
    with _registry_lock:
        return _load_registry()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


# ─────────────────────────────────────────────
# Repositories
# ─────────────────────────────────────────────
def create_repo(
    name: str,
    url: str,
    owner: str,
    default_branch: str = "main",
    access_policy_ref: Optional[str] = None,
) -> Dict[str, Any]:
    repo_id = _new_id("repo-")
    record = {
        "repoId": repo_id,
        "name": name,
        "url": url,
        "owner": owner,
        "defaultBranch": default_branch,
        "accessPolicyRef": access_policy_ref,
        "created_at": _now(),
    }
    with _registry() as reg:
        reg["repos"][repo_id] = record
    return record


def list_repos() -> List[Dict[str, Any]]:
    return list(_read_store()["repos"].values())


# ─────────────────────────────────────────────
# Commits
# ─────────────────────────────────────────────
def create_commit(
    repo_id: str,
    commit_sha: str,
    message: Optional[str] = None,
    author: Optional[str] = None,
    ticket_id: Optional[str] = None,
) -> Dict[str, Any]:
    commit_id = _new_id("commit-")
    record = {
        "commitId": commit_id,
        "repoId": repo_id,
        "commitSha": commit_sha,
        "message": message,
        "author": author,
        "ticketId": ticket_id,
        "buildRefs": [],
        "created_at": _now(),
    }
    with _registry() as reg:
        reg["commits"][commit_id] = record
    return record


def get_commit(commit_sha: str, repo_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    for record in _read_store()["commits"].values():
        if record["commitSha"] == commit_sha:
            if repo_id is None or record["repoId"] == repo_id:
                return record
    return None


def list_commits(repo_id: Optional[str] = None) -> List[Dict[str, Any]]:
    commits = list(_read_store()["commits"].values())
    if repo_id:
        commits = [c for c in commits if c["repoId"] == repo_id]
    return commits


# ─────────────────────────────────────────────
# Tags
# ─────────────────────────────────────────────
def create_tag(
    repo_id: str,
    tag: str,
    commit_sha: str,
    release_notes: Optional[str] = None,
) -> Dict[str, Any]:
    tag_id = _new_id("tag-")
    record = {
        "tagId": tag_id,
        "repoId": repo_id,
        "tag": tag,
        "commitSha": commit_sha,
        "releaseNotes": release_notes,
        "created_at": _now(),
    }
    with _registry() as reg:
        reg["tags"][tag_id] = record
    return record


def list_tags(repo_id: Optional[str] = None) -> List[Dict[str, Any]]:
    tags = list(_read_store()["tags"].values())
    if repo_id:
        tags = [t for t in tags if t["repoId"] == repo_id]
    return tags


# ─────────────────────────────────────────────
# Builds
# ─────────────────────────────────────────────
def _resolve_tag_for_commit(reg: Dict[str, Any], repo_id: str, commit_sha: str) -> Optional[str]:
    for t in reg["tags"].values():
        if t["repoId"] == repo_id and t["commitSha"] == commit_sha:
            return t["tag"]
    return None


def trigger_build(
    repo_id: str,
    build_config_ref: str,
    ticket_id: str,
    commit_sha: Optional[str] = None,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    build_id = _new_id("build-")
    with _registry() as reg:
        if commit_sha and tag is None:
            tag = _resolve_tag_for_commit(reg, repo_id, commit_sha)
        if tag and commit_sha is None:
            for t in reg["tags"].values():
                if t["repoId"] == repo_id and t["tag"] == tag:
                    commit_sha = t["commitSha"]
                    break

        record = {
            "buildId": build_id,
            "repoId": repo_id,
            "commitSha": commit_sha,
            "tag": tag,
            "buildConfigRef": build_config_ref,
            "ticketId": ticket_id,
            "status": "QUEUED",
            "inputs": {
                "repoId": repo_id,
                "commitSha": commit_sha,
                "tag": tag,
                "buildConfigRef": build_config_ref,
            },
            "environment": {},
            "packageRefs": [],
            "logsRef": None,
            "errorSummary": None,
            "artifactRefs": [],
            "queued_at": _now(),
            "started_at": None,
            "finished_at": None,
        }
        reg["builds"][build_id] = record

        # Link build ref back to commit record if found
        if commit_sha:
            for c in reg["commits"].values():
                if c["commitSha"] == commit_sha and c["repoId"] == repo_id:
                    c["buildRefs"].append(build_id)
                    break

    return record


def get_build(build_id: str) -> Optional[Dict[str, Any]]:
    return _read_store()["builds"].get(build_id)


def list_builds(repo_id: Optional[str] = None) -> List[Dict[str, Any]]:
    builds = list(_read_store()["builds"].values())
    if repo_id:
        builds = [b for b in builds if b["repoId"] == repo_id]
    return builds


def update_build(
    build_id: str,
    status: str,
    image_ref: Optional[str] = None,
    digest: Optional[str] = None,
    logs_ref: Optional[str] = None,
    error_summary: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    with _registry() as reg:
        record = reg["builds"].get(build_id)
        if record is None:
            return None
        record["status"] = status
        record["finished_at"] = _now()
        if image_ref:
            record["environment"]["imageRef"] = image_ref
            if image_ref not in record["artifactRefs"]:
                record["artifactRefs"].append(image_ref)
        if digest:
            record["environment"]["imageDigest"] = digest
        if logs_ref:
            record["logsRef"] = logs_ref
        if error_summary:
            record["errorSummary"] = error_summary
    return record


def get_build_status(build_id: str) -> Optional[Dict[str, Any]]:
    build = _read_store()["builds"].get(build_id)
    if build is None:
        return None
    return {
        "buildId": build_id,
        "state": build["status"],
        "logsRef": build.get("logsRef"),
        "errorSummary": build.get("errorSummary"),
        "artifactRefs": build.get("artifactRefs", []),
        "updated_at": build.get("finished_at") or build.get("queued_at"),
    }


# ─────────────────────────────────────────────
# GitHub run_id patching (used by router polling)
# ─────────────────────────────────────────────
def patch_build_github_run(build_id: str, run_id: Optional[int], logs_ref: Optional[str]) -> None:
    with _registry() as reg:
        record = reg["builds"].get(build_id)
        if record:
            record["githubRunId"] = run_id
            if logs_ref:
                record["logsRef"] = logs_ref


def patch_test_run_github_run(test_run_id: str, run_id: Optional[int], logs_ref: Optional[str]) -> None:
    with _registry() as reg:
        record = reg["test_runs"].get(test_run_id)
        if record:
            record["githubRunId"] = run_id
            if logs_ref:
                record["logsRef"] = logs_ref


# ─────────────────────────────────────────────
# Packages
# ─────────────────────────────────────────────
def create_package(
    package_type: str,
    digest: str,
    version: str,
    build_id: str,
    storage_ref: str,
) -> Dict[str, Any]:
    package_id = _new_id("pkg-")
    with _registry() as reg:
        record = {
            "packageId": package_id,
            "type": package_type,
            "digest": digest,
            "version": version,
            "buildId": build_id,
            "storageRef": storage_ref,
            "provenanceLinks": [f"build://{build_id}"],
            "sbomRef": None,
            "securityRef": None,
            "created_at": _now(),
        }
        reg["packages"][package_id] = record

        build = reg["builds"].get(build_id)
        if build:
            build["packageRefs"].append(package_id)
            build["artifactRefs"].append(package_id)

    return record


def get_package(package_id: str) -> Optional[Dict[str, Any]]:
    return _read_store()["packages"].get(package_id)


def list_packages() -> List[Dict[str, Any]]:
    return list(_read_store()["packages"].values())


# ─────────────────────────────────────────────
# Test Runs
# ─────────────────────────────────────────────
def trigger_test(
    test_suite_ref: str,
    ticket_id: str,
    package_id: Optional[str] = None,
    build_id: Optional[str] = None,
    env_ref: Optional[str] = None,
) -> Dict[str, Any]:
    test_run_id = _new_id("testrun-")
    with _registry() as reg:
        if package_id and build_id is None:
            pkg = reg["packages"].get(package_id)
            if pkg:
                build_id = pkg["buildId"]
        if build_id and package_id is None:
            build = reg["builds"].get(build_id)
            if build and build["packageRefs"]:
                package_id = build["packageRefs"][0]

        record = {
            "testRunId": test_run_id,
            "packageId": package_id,
            "buildId": build_id,
            "testSuiteRef": test_suite_ref,
            "envRef": env_ref,
            "ticketId": ticket_id,
            "status": "QUEUED",
            "passed": None,
            "coverageRef": None,
            "qualityMetricRefs": [],
            "reportRef": None,
            "logsRef": None,
            "queued_at": _now(),
            "finished_at": None,
        }
        reg["test_runs"][test_run_id] = record
    return record


def update_test_run(
    test_run_id: str,
    status: str,
    passed: Optional[bool] = None,
    report_ref: Optional[str] = None,
    coverage_ref: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    with _registry() as reg:
        record = reg["test_runs"].get(test_run_id)
        if record is None:
            return None
        record["status"] = status
        record["finished_at"] = _now()
        if passed is not None:
            record["passed"] = passed
        if report_ref:
            record["reportRef"] = report_ref
        if coverage_ref:
            record["coverageRef"] = coverage_ref
    return record


def get_test_run(test_run_id: str) -> Optional[Dict[str, Any]]:
    return _read_store()["test_runs"].get(test_run_id)


def list_test_runs(
    package_id: Optional[str] = None,
    build_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    runs = list(_read_store()["test_runs"].values())
    if package_id:
        runs = [r for r in runs if r["packageId"] == package_id]
    if build_id:
        runs = [r for r in runs if r["buildId"] == build_id]
    return runs


# ─────────────────────────────────────────────
# Releases
# ─────────────────────────────────────────────
def create_release(
    package_id: str,
    release_notes: str,
    ticket_id: str,
    target_envs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    release_id = _new_id("release-")
    with _registry() as reg:
        record = {
            "releaseId": release_id,
            "packageId": package_id,
            "releaseNotes": release_notes,
            "targetEnvs": target_envs or [],
            "ticketId": ticket_id,
            "status": "PENDING",
            "notes": None,
            "deploymentRefs": [],
            "created_at": _now(),
            "updated_at": None,
        }
        reg["releases"][release_id] = record
    return record


def get_release(release_id: str) -> Optional[Dict[str, Any]]:
    return _read_store()["releases"].get(release_id)


def update_release(
    release_id: str,
    status: Optional[str] = None,
    notes: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    with _registry() as reg:
        record = reg["releases"].get(release_id)
        if record is None:
            return None
        if status is not None:
            record["status"] = status
        if notes is not None:
            record["notes"] = notes
        record["updated_at"] = _now()
    return record


def list_releases() -> List[Dict[str, Any]]:
    return list(_read_store()["releases"].values())
