"""
Software – Build Management endpoints.

  POST  /builds              – Trigger a build execution (CI job / container build)
  GET   /builds              – List build executions
  GET   /builds/{buildId}    – Build metadata (inputs, outputs, environment)
  GET   /builds/{buildId}/status – Poll build progress/outcome and logs pointers
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from ....core.logs import make_log
from ..schemas import (
    BuildTriggerRequest,
    BuildResponse,
    BuildDetailResponse,
    BuildStatusResponse,
)
from .. import software_service as svc

router = APIRouter()


# ─────────────────────────────────────────────
# BUILDS
# ─────────────────────────────────────────────
@router.post("/builds", response_model=BuildResponse,
             summary="Trigger a build execution (CI job / container build)")
def trigger_build(req: BuildTriggerRequest):
    """
    Queue a new build execution for a given repository commit or tag.

    **API spec inputs:** repoId, commitSha|tag, buildConfigRef, ticketId
    **Returns:** buildId, queued status
    """
    make_log(
        area="Soft",
        component="Build Management",
        endpoint="/soft/builds",
        meta={"repoId": req.repoId, "commitSha": req.commitSha, "tag": req.tag},
    )
    if not req.commitSha and not req.tag:
        raise HTTPException(status_code=400, detail={
            "error": {"code": "missing_source_ref", "message": "Provide either commitSha or tag"}
        })
    try:
        result = svc.trigger_build(
            repo_id=req.repoId,
            build_config_ref=req.buildConfigRef,
            ticket_id=req.ticketId,
            commit_sha=req.commitSha,
            tag=req.tag,
        )
        return BuildResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "build_trigger_failed", "message": str(e)}
        })


@router.get("/builds", response_model=List[BuildResponse],
            summary="List build executions, optionally filtered by repo")
def list_builds(repoId: Optional[str] = Query(None, description="Filter by repository ID")):
    """
    List all build executions, optionally filtered by repository.
    """
    make_log(
        area="Soft",
        component="Build Management",
        endpoint="/soft/builds",
        meta={"repoId": repoId},
    )
    try:
        return [BuildResponse(**b) for b in svc.list_builds(repo_id=repoId)]
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "list_builds_failed", "message": str(e)}
        })


@router.get("/builds/{buildId}", response_model=BuildDetailResponse,
            summary="Build metadata (inputs, outputs, environment)")
def get_build(buildId: str):
    """
    Retrieve detailed build metadata including inputs, environment, and produced package refs.

    **API spec inputs:** none
    **Returns:** build inputs, environment, produced package refs
    """
    make_log(
        area="Soft",
        component="Build Management",
        endpoint=f"/soft/builds/{buildId}",
        meta={"buildId": buildId},
    )
    build = svc.get_build(buildId)
    if build is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "build_not_found", "message": f"Build '{buildId}' not found"}
        })
    return BuildDetailResponse(**build)


@router.get("/builds/{buildId}/status", response_model=BuildStatusResponse,
            summary="Poll build progress/outcome and logs pointers")
def get_build_status(buildId: str):
    """
    Poll the current state, log reference, and artifact refs for a build.

    **API spec inputs:** none
    **Returns:** state, logs ref, error summary, artifact refs
    """
    make_log(
        area="Soft",
        component="Build Management",
        endpoint=f"/soft/builds/{buildId}/status",
        meta={"buildId": buildId},
    )
    status = svc.get_build_status(buildId)
    if status is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "build_not_found", "message": f"Build '{buildId}' not found"}
        })
    return BuildStatusResponse(**status)
