"""
Repository Management API endpoints.
  POST   /repositories/bootstrap  – Initialize repos/registries for code/data/model artifacts
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ....core.logs import make_log
from ..schemas import RepoBootstrapRequest, RepoBootstrapResponse
from .. import plan_service as svc

router = APIRouter()


@router.post("/repositories/bootstrap", response_model=RepoBootstrapResponse,
             summary="Initialize repos/registries for code, data, and model artifacts")
def bootstrap_repositories(req: RepoBootstrapRequest):
    """
    Bootstrap one or more artifact repositories for a project in a single call.

    **Inputs:** projectId, repoTypes[] (code|data|model), provider, naming?, accessPolicy?
    **Returns:** repo URIs/ids + access metadata
    """
    log = make_log(
        area="Plan", component="Repository Management",
        endpoint="/plan/repositories/bootstrap",
        meta={"projectId": req.projectId, "repoTypes": req.repoTypes, "provider": req.provider},
    )
    try:
        result = svc.bootstrap_repositories(
            project_id=req.projectId,
            repo_types=req.repoTypes,
            provider=req.provider,
            naming=req.naming,
            access_policy=req.accessPolicy,
        )
        return RepoBootstrapResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "repository_bootstrap_failed", "message": str(e)}
        })
