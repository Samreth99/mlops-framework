"""
Software – Code Management endpoints.

  POST  /code/repositories   – Register a code repository
  GET   /code/repositories   – List registered repositories
  POST  /code/commits        – Record a commit tied to a build
  GET   /code/commits        – List/lookup commits
  POST  /code/tags           – Create a tag/release pointer
  GET   /code/tags           – List tags
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from ....core.logs import make_log
from ..schemas import (
    RepoCreateRequest,
    RepoResponse,
    CommitCreateRequest,
    CommitResponse,
    TagCreateRequest,
    TagResponse,
)
from .. import software_service as svc

router = APIRouter()


# ─────────────────────────────────────────────
# REPOSITORIES
# ─────────────────────────────────────────────
@router.post("/code/repositories", response_model=RepoResponse,
             summary="Register a code repository used by the system")
def create_repo(req: RepoCreateRequest):
    """
    Register a source code repository.

    **API spec inputs:** name, url, owner, defaultBranch, accessPolicyRef?
    **Returns:** repoId
    """
    make_log(
        area="Soft",
        component="Code Management",
        endpoint="/soft/code/repositories",
        meta={"name": req.name, "owner": req.owner},
    )
    try:
        result = svc.create_repo(
            name=req.name,
            url=req.url,
            owner=req.owner,
            default_branch=req.defaultBranch,
            access_policy_ref=req.accessPolicyRef,
        )
        return RepoResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "repo_creation_failed", "message": str(e)}
        })


@router.get("/code/repositories", response_model=List[RepoResponse],
            summary="List registered code repositories")
def list_repos():
    """
    List all registered code repositories.
    """
    make_log(
        area="Soft",
        component="Code Management",
        endpoint="/soft/code/repositories",
    )
    try:
        return [RepoResponse(**r) for r in svc.list_repos()]
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "list_repos_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# COMMITS
# ─────────────────────────────────────────────
@router.post("/code/commits", response_model=CommitResponse,
             summary="Record a commit identifier tied to builds")
def create_commit(req: CommitCreateRequest):
    """
    Record a commit SHA for traceability and build linkage.

    **API spec inputs:** repoId, commitSha, message?, author?, ticketId?
    **Returns:** commit record + links to builds
    """
    make_log(
        area="Soft",
        component="Code Management",
        endpoint="/soft/code/commits",
        meta={"repoId": req.repoId, "commitSha": req.commitSha},
    )
    try:
        result = svc.create_commit(
            repo_id=req.repoId,
            commit_sha=req.commitSha,
            message=req.message,
            author=req.author,
            ticket_id=req.ticketId,
        )
        return CommitResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "commit_creation_failed", "message": str(e)}
        })


@router.get("/code/commits", response_model=List[CommitResponse],
            summary="List/lookup commits, optionally filtered by repo")
def list_commits(repoId: Optional[str] = Query(None, description="Filter by repository ID")):
    """
    List commits, optionally filtered by repository.
    """
    make_log(
        area="Soft",
        component="Code Management",
        endpoint="/soft/code/commits",
        meta={"repoId": repoId},
    )
    try:
        return [CommitResponse(**c) for c in svc.list_commits(repo_id=repoId)]
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "list_commits_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# TAGS
# ─────────────────────────────────────────────
@router.post("/code/tags", response_model=TagResponse,
             summary="Create a tag/release pointer for reproducibility")
def create_tag(req: TagCreateRequest):
    """
    Manage tags/releases pointers used for reproducibility.

    **API spec inputs:** repoId, tag, commitSha, releaseNotes?
    **Returns:** tag record
    """
    make_log(
        area="Soft",
        component="Code Management",
        endpoint="/soft/code/tags",
        meta={"repoId": req.repoId, "tag": req.tag, "commitSha": req.commitSha},
    )
    try:
        result = svc.create_tag(
            repo_id=req.repoId,
            tag=req.tag,
            commit_sha=req.commitSha,
            release_notes=req.releaseNotes,
        )
        return TagResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "tag_creation_failed", "message": str(e)}
        })


@router.get("/code/tags", response_model=List[TagResponse],
            summary="List tags, optionally filtered by repo")
def list_tags(repoId: Optional[str] = Query(None, description="Filter by repository ID")):
    """
    List all tags/release pointers, optionally filtered by repository.
    """
    make_log(
        area="Soft",
        component="Code Management",
        endpoint="/soft/code/tags",
        meta={"repoId": repoId},
    )
    try:
        return [TagResponse(**t) for t in svc.list_tags(repo_id=repoId)]
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "list_tags_failed", "message": str(e)}
        })
