from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Code – Repositories
# ─────────────────────────────────────────────
class RepoCreateRequest(BaseModel):
    name: str = Field(..., description="Repository name")
    url: str = Field(..., description="Repository URL")
    owner: str = Field(..., description="Repository owner")
    defaultBranch: str = Field("main", description="Default branch name")
    accessPolicyRef: Optional[str] = Field(None, description="Reference to access policy")


class RepoResponse(BaseModel):
    repoId: str
    name: str
    url: str
    owner: str
    defaultBranch: str
    accessPolicyRef: Optional[str] = None
    created_at: Optional[str] = None


# ─────────────────────────────────────────────
# Code – Commits
# ─────────────────────────────────────────────
class CommitCreateRequest(BaseModel):
    repoId: str = Field(..., description="Repository identifier")
    commitSha: str = Field(..., description="Full commit SHA")
    message: Optional[str] = Field(None, description="Commit message")
    author: Optional[str] = Field(None, description="Commit author")
    ticketId: Optional[str] = Field(None, description="Linked orchestration ticket")


class CommitResponse(BaseModel):
    commitId: str
    repoId: str
    commitSha: str
    message: Optional[str] = None
    author: Optional[str] = None
    ticketId: Optional[str] = None
    buildRefs: List[str] = []
    created_at: Optional[str] = None


# ─────────────────────────────────────────────
# Code – Tags
# ─────────────────────────────────────────────
class TagCreateRequest(BaseModel):
    repoId: str = Field(..., description="Repository identifier")
    tag: str = Field(..., description="Tag name (e.g. v1.0.0)")
    commitSha: str = Field(..., description="Commit SHA this tag points to")
    releaseNotes: Optional[str] = Field(None, description="Optional release notes")


class TagResponse(BaseModel):
    tagId: str
    repoId: str
    tag: str
    commitSha: str
    releaseNotes: Optional[str] = None
    created_at: Optional[str] = None


# ─────────────────────────────────────────────
# Builds
# ─────────────────────────────────────────────
class BuildTriggerRequest(BaseModel):
    repoId: str = Field(..., description="Source repository identifier")
    commitSha: Optional[str] = Field(None, description="Commit SHA to build from")
    tag: Optional[str] = Field(None, description="Tag to build from (alternative to commitSha)")
    buildConfigRef: str = Field(..., description="Reference to the build configuration")
    ticketId: str = Field(..., description="Linked orchestration ticket")


class BuildResponse(BaseModel):
    buildId: str
    repoId: str
    commitSha: Optional[str] = None
    tag: Optional[str] = None
    buildConfigRef: str
    ticketId: str
    status: str = "QUEUED"
    queued_at: Optional[str] = None


class BuildDetailResponse(BaseModel):
    buildId: str
    repoId: str
    commitSha: Optional[str] = None
    tag: Optional[str] = None
    buildConfigRef: str
    ticketId: str
    status: str
    inputs: Dict[str, Any] = {}
    environment: Dict[str, Any] = {}
    packageRefs: List[str] = []
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class BuildStatusResponse(BaseModel):
    buildId: str
    state: str
    logsRef: Optional[str] = None
    errorSummary: Optional[str] = None
    artifactRefs: List[str] = []
    updated_at: Optional[str] = None


# ─────────────────────────────────────────────
# Packages
# ─────────────────────────────────────────────
class PackageCreateRequest(BaseModel):
    type: str = Field(..., description="Package type: image | jar | wheel")
    digest: str = Field(..., description="Content digest/hash of the package")
    version: str = Field(..., description="Package version string")
    buildId: str = Field(..., description="Build that produced this package")
    storageRef: str = Field(..., description="Storage location reference")


class PackageResponse(BaseModel):
    packageId: str
    type: str
    digest: str
    version: str
    buildId: str
    storageRef: str
    provenanceLinks: List[str] = []
    sbomRef: Optional[str] = None
    securityRef: Optional[str] = None
    created_at: Optional[str] = None


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────
class TestTriggerRequest(BaseModel):
    packageId: Optional[str] = Field(None, description="Package to test")
    buildId: Optional[str] = Field(None, description="Build to test (alternative to packageId)")
    testSuiteRef: str = Field(..., description="Reference to the test suite definition")
    envRef: Optional[str] = Field(None, description="Environment reference for test execution")
    ticketId: str = Field(..., description="Linked orchestration ticket")


class TestRunResponse(BaseModel):
    testRunId: str
    packageId: Optional[str] = None
    buildId: Optional[str] = None
    testSuiteRef: str
    ticketId: str
    status: str = "QUEUED"
    queued_at: Optional[str] = None


class TestRunDetailResponse(BaseModel):
    testRunId: str
    packageId: Optional[str] = None
    buildId: Optional[str] = None
    testSuiteRef: str
    ticketId: str
    status: str
    passed: Optional[bool] = None
    coverageRef: Optional[str] = None
    qualityMetricRefs: List[str] = []
    reportRef: Optional[str] = None
    finished_at: Optional[str] = None


# ─────────────────────────────────────────────
# Releases
# ─────────────────────────────────────────────
class ReleaseCreateRequest(BaseModel):
    packageId: str = Field(..., description="Package to release")
    releaseNotes: str = Field(..., description="Release notes")
    targetEnvs: Optional[List[str]] = Field(None, description="Target deployment environments")
    ticketId: str = Field(..., description="Linked orchestration ticket")


class ReleaseResponse(BaseModel):
    releaseId: str
    packageId: str
    releaseNotes: str
    targetEnvs: List[str] = []
    ticketId: str
    status: str = "PENDING"
    created_at: Optional[str] = None


class ReleaseUpdateRequest(BaseModel):
    status: Optional[str] = Field(None, description="New release status")
    notes: Optional[str] = Field(None, description="Additional notes")


class ReleaseDetailResponse(BaseModel):
    releaseId: str
    packageId: str
    releaseNotes: str
    targetEnvs: List[str] = []
    ticketId: str
    status: str
    notes: Optional[str] = None
    deploymentRefs: List[str] = []
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
