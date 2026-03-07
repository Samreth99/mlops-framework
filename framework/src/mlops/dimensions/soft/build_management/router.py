"""
Software – Build Management endpoints.

  POST   /builds              – Trigger a build execution (dispatches Docker build via GitHub Actions)
  GET    /builds              – List build executions
  GET    /builds/{buildId}    – Build metadata (inputs, outputs, environment)
  GET    /builds/{buildId}/status – Poll build progress/outcome and logs pointers
  PATCH  /builds/{buildId}   – CI callback: update Docker build result from GitHub Actions
"""
from __future__ import annotations

import logging
from typing import List, Optional

import httpx

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from ....core.logs import make_log
from ....config.settings import settings
from ..schemas import (
    BuildTriggerRequest,
    BuildResponse,
    BuildDetailResponse,
    BuildStatusResponse,
    BuildCallbackRequest,
)
from .. import software_service as svc

router = APIRouter()

# ─────────────────────────────────────────────
# GitHub Actions workflow file for Docker builds
# ─────────────────────────────────────────────
BUILD_WORKFLOW_FILE = "docker-build-ci.yml"


def _dispatch_github_build(
    build_id: str,
    commit_sha: Optional[str],
    tag: Optional[str],
) -> None:
    """
    Dispatch the Docker build workflow on GitHub Actions as a background task.
    Marks the build as FAILED if GitHub is not configured or dispatch fails.
    """
    token      = settings.github_token
    owner      = settings.github_repo_owner
    repo       = settings.github_repo_name
    public_url = settings.public_api_url

    if not all([token, owner, repo, public_url]):
        svc.update_build(
            build_id,
            status="FAILED",
            error_summary="GitHub Actions not configured (missing token/owner/repo/public_url)",
        )
        return

    callback_url = f"{public_url}/soft/builds/{build_id}"
    # Use tag name or commit SHA as the Docker image tag
    image_tag = tag or commit_sha or build_id

    try:
        resp = httpx.post(
            f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{BUILD_WORKFLOW_FILE}/dispatches",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept":        "application/vnd.github.v3+json",
            },
            json={
                "ref": "ks-software",
                "inputs": {
                    "buildId":     build_id,
                    "callbackUrl": callback_url,
                    "imageTag":    image_tag,
                },
            },
            timeout=10.0,
        )
        if resp.status_code != 204:
            logging.error(
                "[Build Dispatch] GitHub returned %s: %s",
                resp.status_code, resp.text,
            )
            svc.update_build(
                build_id,
                status="FAILED",
                error_summary=f"GitHub Actions dispatch failed (HTTP {resp.status_code})",
            )
    except Exception as exc:
        logging.error("[Build Dispatch] Exception: %s", exc)
        svc.update_build(build_id, status="FAILED", error_summary=str(exc))


# ─────────────────────────────────────────────
# BUILDS
# ─────────────────────────────────────────────
@router.post("/builds", response_model=BuildResponse,
             summary="Trigger a Docker build via GitHub Actions")
def trigger_build(req: BuildTriggerRequest, background_tasks: BackgroundTasks):
    """
    Queue a new Docker build for a given repository commit or tag.
    Dispatches the `docker-build-ci.yml` GitHub Actions workflow which
    builds `framework/Dockerfile`, pushes to GHCR, then calls back
    `PATCH /soft/builds/{buildId}` with the image ref and digest.

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
        background_tasks.add_task(
            _dispatch_github_build,
            result["buildId"],
            result.get("commitSha"),
            result.get("tag"),
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
    Retrieve detailed build metadata including inputs, environment (imageRef, imageDigest),
    and produced package refs.

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


@router.patch("/builds/{buildId}", response_model=BuildStatusResponse,
              summary="CI callback: update Docker build result from GitHub Actions")
def ci_build_callback(buildId: str, req: BuildCallbackRequest):
    """
    Called by the `docker-build-ci.yml` GitHub Actions workflow at the end of a Docker build.

    GitHub Actions sends:
      status      : BUILT | FAILED
      imageRef    : ghcr.io/owner/repo:sha  (on success)
      digest      : sha256:...              (on success)
      logsRef     : link to build run logs
      errorSummary: short failure message   (on failure)
    """
    make_log(
        area="Soft",
        component="Build Management",
        endpoint=f"/soft/builds/{buildId}",
        meta={"buildId": buildId, "status": req.status, "imageRef": req.imageRef},
    )
    svc.update_build(
        build_id=buildId,
        status=req.status,
        image_ref=req.imageRef,
        digest=req.digest,
        logs_ref=req.logsRef,
        error_summary=req.errorSummary,
    )
    result = svc.get_build_status(buildId)
    if result is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "build_not_found", "message": f"Build '{buildId}' not found"}
        })
    return BuildStatusResponse(**result)
