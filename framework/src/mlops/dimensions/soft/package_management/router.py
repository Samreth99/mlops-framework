"""
Software – Package Management endpoints.

  POST  /packages              – Create/register a packaged artifact
  GET   /packages              – List packaged artifacts
  GET   /packages/{packageId}  – Package metadata and digest/version identifiers
"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from ....core.logs import make_log
from ..schemas import (
    PackageCreateRequest,
    PackageResponse,
)
from .. import software_service as svc

router = APIRouter()


# ─────────────────────────────────────────────
# PACKAGES
# ─────────────────────────────────────────────
@router.post("/packages", response_model=PackageResponse,
             summary="Create/register a packaged artifact (image, wheel, jar)")
def create_package(req: PackageCreateRequest):
    """
    Register a packaged artifact produced by a build.

    **API spec inputs:** type (image/jar/wheel), digest, version, buildId, storageRef
    **Returns:** packageId
    """
    make_log(
        area="Soft",
        component="Package Management",
        endpoint="/soft/packages",
        meta={"type": req.type, "version": req.version, "buildId": req.buildId},
    )
    try:
        result = svc.create_package(
            package_type=req.type,
            digest=req.digest,
            version=req.version,
            build_id=req.buildId,
            storage_ref=req.storageRef,
        )
        return PackageResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "package_creation_failed", "message": str(e)}
        })


@router.get("/packages", response_model=List[PackageResponse],
            summary="List all packaged artifacts")
def list_packages():
    """
    List all registered packaged artifacts.
    """
    make_log(
        area="Soft",
        component="Package Management",
        endpoint="/soft/packages",
    )
    try:
        return [PackageResponse(**p) for p in svc.list_packages()]
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "list_packages_failed", "message": str(e)}
        })


@router.get("/packages/{packageId}", response_model=PackageResponse,
            summary="Package metadata and digest/version identifiers")
def get_package(packageId: str):
    """
    Retrieve package metadata including digest, provenance links, and SBOM/security refs.

    **API spec inputs:** none
    **Returns:** digest, provenance links, SBOM/security refs
    """
    make_log(
        area="Soft",
        component="Package Management",
        endpoint=f"/soft/packages/{packageId}",
        meta={"packageId": packageId},
    )
    pkg = svc.get_package(packageId)
    if pkg is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "package_not_found", "message": f"Package '{packageId}' not found"}
        })
    return PackageResponse(**pkg)
