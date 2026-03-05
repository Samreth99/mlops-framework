"""
Software dimension service layer.

Provides an in-memory store for:
  - Code repositories, commits, and tags
  - Build executions
  - Packaged artifacts
  - Test runs
  - Software releases

All IDs are generated with uuid4. Timestamps are ISO-8601 UTC strings.
The store resets on process restart — production deployments should swap
_store with a database-backed implementation.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
# In-memory store
# ─────────────────────────────────────────────
_store: Dict[str, Dict[str, Any]] = {
    "repos": {},
    "commits": {},
    "tags": {},
    "builds": {},
    "packages": {},
    "test_runs": {},
    "releases": {},
}


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
    _store["repos"][repo_id] = record
    return record


def list_repos() -> List[Dict[str, Any]]:
    return list(_store["repos"].values())


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
    _store["commits"][commit_id] = record
    return record


def get_commit(commit_sha: str, repo_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    for record in _store["commits"].values():
        if record["commitSha"] == commit_sha:
            if repo_id is None or record["repoId"] == repo_id:
                return record
    return None


def list_commits(repo_id: Optional[str] = None) -> List[Dict[str, Any]]:
    commits = list(_store["commits"].values())
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
    _store["tags"][tag_id] = record
    return record


def list_tags(repo_id: Optional[str] = None) -> List[Dict[str, Any]]:
    tags = list(_store["tags"].values())
    if repo_id:
        tags = [t for t in tags if t["repoId"] == repo_id]
    return tags


# ─────────────────────────────────────────────
# Builds
# ─────────────────────────────────────────────
def _resolve_tag_for_commit(repo_id: str, commit_sha: str) -> Optional[str]:
    """Return the tag name pointing to this commit in this repo, if one exists."""
    for t in _store["tags"].values():
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
    # If triggered by commitSha, auto-resolve any existing tag pointing to it
    if commit_sha and tag is None:
        tag = _resolve_tag_for_commit(repo_id, commit_sha)

    # If triggered by tag name, resolve the commitSha from the tag record
    if tag and commit_sha is None:
        for t in _store["tags"].values():
            if t["repoId"] == repo_id and t["tag"] == tag:
                commit_sha = t["commitSha"]
                break

    build_id = _new_id("build-")
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
    _store["builds"][build_id] = record

    # Link build ref back to commit record if found
    if commit_sha:
        commit = get_commit(commit_sha, repo_id)
        if commit:
            commit["buildRefs"].append(build_id)

    return record


def get_build(build_id: str) -> Optional[Dict[str, Any]]:
    return _store["builds"].get(build_id)


def list_builds(repo_id: Optional[str] = None) -> List[Dict[str, Any]]:
    builds = list(_store["builds"].values())
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
    """Called by GitHub Actions CI callback (PATCH) to set the real build result."""
    record = _store["builds"].get(build_id)
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
    build = _store["builds"].get(build_id)
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
    _store["packages"][package_id] = record

    # Link package back to build's packageRefs
    build = _store["builds"].get(build_id)
    if build:
        build["packageRefs"].append(package_id)
        build["artifactRefs"].append(package_id)

    return record


def get_package(package_id: str) -> Optional[Dict[str, Any]]:
    return _store["packages"].get(package_id)


def list_packages() -> List[Dict[str, Any]]:
    return list(_store["packages"].values())


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
    # Auto-resolve buildId from the package record if not explicitly provided
    if package_id and build_id is None:
        pkg = _store["packages"].get(package_id)
        if pkg:
            build_id = pkg["buildId"]

    # Auto-resolve packageId from the build's packageRefs if not explicitly provided
    if build_id and package_id is None:
        build = _store["builds"].get(build_id)
        if build and build["packageRefs"]:
            package_id = build["packageRefs"][0]

    test_run_id = _new_id("testrun-")
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
        "queued_at": _now(),
        "finished_at": None,
    }
    _store["test_runs"][test_run_id] = record
    return record


def update_test_run(
    test_run_id: str,
    status: str,
    passed: Optional[bool] = None,
    report_ref: Optional[str] = None,
    coverage_ref: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Called by GitHub Actions CI callback (PATCH) to set the real test result.
    status: PASSED | FAILED
    """
    record = _store["test_runs"].get(test_run_id)
    if record is None:
        return None
    record["status"]      = status
    record["finished_at"] = _now()
    if passed is not None:
        record["passed"] = passed
    if report_ref:
        record["reportRef"] = report_ref
    if coverage_ref:
        record["coverageRef"] = coverage_ref
    return record


def get_test_run(test_run_id: str) -> Optional[Dict[str, Any]]:
    return _store["test_runs"].get(test_run_id)


def list_test_runs(
    package_id: Optional[str] = None,
    build_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    runs = list(_store["test_runs"].values())
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
    _store["releases"][release_id] = record
    return record


def get_release(release_id: str) -> Optional[Dict[str, Any]]:
    return _store["releases"].get(release_id)


def update_release(
    release_id: str,
    status: Optional[str] = None,
    notes: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    record = _store["releases"].get(release_id)
    if record is None:
        return None
    if status is not None:
        record["status"] = status
    if notes is not None:
        record["notes"] = notes
    record["updated_at"] = _now()
    return record


def list_releases() -> List[Dict[str, Any]]:
    return list(_store["releases"].values())
