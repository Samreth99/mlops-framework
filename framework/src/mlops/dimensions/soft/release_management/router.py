"""
Software – Release Management endpoints.

  POST  /releases              – Create a software release (deployment candidate)
  GET   /releases              – List software releases
  GET   /releases/{releaseId}  – Release details (what package/version is released)
  POST  /releases/{releaseId}  – Update release status or notes
"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from ....core.logs import make_log
from ..schemas import (
    ReleaseCreateRequest,
    ReleaseResponse,
    ReleaseUpdateRequest,
    ReleaseDetailResponse,
)
from .. import software_service as svc

router = APIRouter()


# ─────────────────────────────────────────────
# RELEASES
# ─────────────────────────────────────────────
@router.post("/releases", response_model=ReleaseResponse,
             summary="Create a software release (deployment candidate)")
def create_release(req: ReleaseCreateRequest):
    """
    Create a software release entry that marks a package as a deployment candidate.

    **API spec inputs:** packageId, releaseNotes, targetEnvs?[], ticketId
    **Returns:** releaseId
    """
    make_log(
        area="Soft",
        component="Release Management",
        endpoint="/soft/releases",
        meta={"packageId": req.packageId, "ticketId": req.ticketId},
    )
    try:
        result = svc.create_release(
            package_id=req.packageId,
            release_notes=req.releaseNotes,
            ticket_id=req.ticketId,
            target_envs=req.targetEnvs,
        )
        return ReleaseResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "release_creation_failed", "message": str(e)}
        })


@router.get("/releases", response_model=List[ReleaseResponse],
            summary="List all software releases")
def list_releases():
    """
    List all software releases (deployment candidates).
    """
    make_log(
        area="Soft",
        component="Release Management",
        endpoint="/soft/releases",
    )
    try:
        return [ReleaseResponse(**r) for r in svc.list_releases()]
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "list_releases_failed", "message": str(e)}
        })


@router.get("/releases/{releaseId}", response_model=ReleaseDetailResponse,
            summary="Release details (what package/version is released)")
def get_release(releaseId: str):
    """
    Retrieve full release details including package, target envs, and deployment links.

    **API spec inputs:** none
    **Returns:** release snapshot + links to deployments
    """
    make_log(
        area="Soft",
        component="Release Management",
        endpoint=f"/soft/releases/{releaseId}",
        meta={"releaseId": releaseId},
    )
    release = svc.get_release(releaseId)
    if release is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "release_not_found", "message": f"Release '{releaseId}' not found"}
        })
    return ReleaseDetailResponse(**release)


@router.post("/releases/{releaseId}", response_model=ReleaseDetailResponse,
             summary="Update release status or notes")
def update_release(releaseId: str, req: ReleaseUpdateRequest):
    """
    Update a release's status (e.g. PENDING → APPROVED → DEPLOYED) or add notes.

    **API spec inputs:** status?, notes?
    **Returns:** release snapshot + links to deployments
    """
    make_log(
        area="Soft",
        component="Release Management",
        endpoint=f"/soft/releases/{releaseId}",
        meta={"releaseId": releaseId, "status": req.status},
    )
    result = svc.update_release(
        release_id=releaseId,
        status=req.status,
        notes=req.notes,
    )
    if result is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "release_not_found", "message": f"Release '{releaseId}' not found"}
        })
    return ReleaseDetailResponse(**result)
