"""
Model Storage & Versioning router for MLOps.

Implements the following Model-dimension endpoints from the API specification:

  POST   /models                                       – Register a logical model family
  GET    /models                                       – List registered models
  GET    /models/{modelId}                             – Get model metadata
  POST   /models/{modelId}                             – Update model metadata
  POST   /models/{modelId}/versions                    – Create a model version
  GET    /models/{modelId}/versions                    – List model versions
  GET    /models/{modelId}/versions/{versionId}        – Fetch a specific version
  POST   /models/{modelId}/register                    – Register a trained artifact as version
  POST   /models/{modelId}/versions/{versionId}/evidence – Attach evidence
  GET    /models/{modelId}/versions/{versionId}/evidence – Query evidence

All backed by the MLflow Model Registry.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from ....core.logs import make_log
from ..schemas import (
    ModelCreateRequest,
    ModelResponse,
    ModelUpdateRequest,
    ModelVersionCreateRequest,
    ModelVersionResponse,
    ModelRegisterRequest,
    ModelRegisterResponse,
    EvidenceAttachRequest,
    EvidenceResponse,
)
from .. import mlflow_service as svc

router = APIRouter()


# ─────────────────────────────────────────────
# MODELS (Registry)
# ─────────────────────────────────────────────
@router.post("/models", response_model=ModelResponse,
             summary="Register a logical model family")
def create_model(req: ModelCreateRequest):
    """
    Register (or return existing) a model family in the MLflow Model Registry.

    **API spec inputs:** name, owner, intendedUse, riskNotes
    **Returns:** modelId, name, owner, intendedUse, riskNotes, timestamps, tags
    """
    log = make_log(area="Model", component="Model Storage & Versioning", endpoint="/model/models", meta={"name": req.name})
    try:
        result = svc.register_model_family(
            name=req.name,
            owner=req.owner,
            intended_use=req.intendedUse,
            risk_notes=req.riskNotes,
            tags=req.tags,
        )
        return ModelResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "model_creation_failed", "message": str(e)}
        })


@router.get("/models", response_model=List[ModelResponse],
            summary="List all registered models")
def list_models(filter: Optional[str] = Query(None, description="MLflow filter string")):
    """
    List all registered model families from the MLflow Model Registry.
    """
    log = make_log(area="Model", component="Model Storage & Versioning", endpoint="/model/models")
    try:
        results = svc.list_models(filter_string=filter)
        return [ModelResponse(**r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "list_models_failed", "message": str(e)}
        })


@router.get("/models/{modelId}", response_model=ModelResponse,
            summary="Get model metadata")
def get_model(modelId: str):
    """
    Retrieve metadata for a registered model (purpose, owners, intended use).

    **modelId** corresponds to the registered model name in MLflow.
    """
    log = make_log(area="Model", component="Model Storage & Versioning", endpoint=f"/model/models/{modelId}", meta={"modelId": modelId})
    try:
        result = svc.get_model(modelId)
        return ModelResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "model_not_found", "message": str(e)}
        })


@router.post("/models/{modelId}", response_model=ModelResponse,
             summary="Update model metadata")
def update_model(modelId: str, req: ModelUpdateRequest):
    """
    Update model metadata such as description, tags, intended use.
    """
    log = make_log(area="Model", component="Model Storage & Versioning", endpoint=f"/model/models/{modelId}", meta={"modelId": modelId})
    try:
        # Build tags from individual fields
        tags = dict(req.tags) if req.tags else {}
        if req.intendedUse:
            tags["intendedUse"] = req.intendedUse
        if req.riskNotes:
            tags["riskNotes"] = req.riskNotes
        if req.owners:
            tags["owner"] = ",".join(req.owners)

        result = svc.update_model(
            model_id=modelId,
            description=req.description or req.intendedUse,
            tags=tags if tags else None,
        )
        return ModelResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "model_update_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# MODEL VERSIONS
# ─────────────────────────────────────────────
@router.post("/models/{modelId}/versions", response_model=ModelVersionResponse,
             summary="Create a model version (with lineage)")
def create_model_version(modelId: str, req: ModelVersionCreateRequest):
    """
    Create a new model version from an artifact reference.

    **API spec inputs:** artifactRef, trainingRunId, metricsRef,
    dataset/feature refs, ticketId, stage
    """
    log = make_log(area="Model", component="Model Storage & Versioning", endpoint=f"/model/models/{modelId}/versions", meta={"modelId": modelId, "artifactRef": req.artifactRef})
    try:
        tags = {}
        if req.datasetRef:
            tags["datasetRef"] = req.datasetRef
        if req.featureRef:
            tags["featureRef"] = req.featureRef
        if req.metricsRef:
            tags["metricsRef"] = req.metricsRef
        if req.ticketId:
            tags["ticketId"] = req.ticketId

        result = svc.create_model_version(
            model_id=modelId,
            artifact_ref=req.artifactRef,
            run_id=req.trainingRunId,
            description=req.description,
            stage=req.stage,
            tags=tags if tags else None,
        )
        return ModelVersionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "version_creation_failed", "message": str(e)}
        })


@router.get("/models/{modelId}/versions", response_model=List[ModelVersionResponse],
            summary="List all versions of a model")
def list_model_versions(modelId: str):
    """
    List all versions of a registered model, with lineage metadata.
    """
    log = make_log(area="Model", component="Model Storage & Versioning", endpoint=f"/model/models/{modelId}/versions", meta={"modelId": modelId})
    try:
        results = svc.list_model_versions(modelId)
        return [ModelVersionResponse(**r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "list_versions_failed", "message": str(e)}
        })


@router.get("/models/{modelId}/versions/{versionId}", response_model=ModelVersionResponse,
            summary="Fetch a specific model artifact version")
def get_model_version(modelId: str, versionId: str):
    """
    Retrieve artifact pointer, lineage, metrics/evidence refs for a specific version.
    """
    log = make_log(area="Model", component="Model Storage & Versioning", endpoint=f"/model/models/{modelId}/versions/{versionId}", meta={"modelId": modelId, "versionId": versionId})
    try:
        result = svc.get_model_version(modelId, versionId)
        return ModelVersionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "version_not_found", "message": str(e)}
        })


# ─────────────────────────────────────────────
# REGISTER (convenience endpoint)
# ─────────────────────────────────────────────
@router.post("/models/{modelId}/register", response_model=ModelRegisterResponse,
             summary="Register a trained artifact + metadata as a version")
def register_model_artifact(modelId: str, req: ModelRegisterRequest):
    """
    Convenience endpoint that ensures the model family exists and creates
    a new version from a trained artifact.

    **API spec inputs:** artifactRef, runId, metadata, evidenceRefs[], ticketId
    **Returns:** versionId
    """
    log = make_log(area="Model", component="Model Storage & Versioning", endpoint=f"/model/models/{modelId}/register", meta={"modelId": modelId, "artifactRef": req.artifactRef})
    try:
        result = svc.register_artifact_as_version(
            model_id=modelId,
            artifact_ref=req.artifactRef,
            run_id=req.runId,
            metadata=req.metadata,
            evidence_refs=req.evidenceRefs,
            description=req.description,
        )
        return ModelRegisterResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "registration_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# EVIDENCE
# ─────────────────────────────────────────────
@router.post("/models/{modelId}/versions/{versionId}/evidence",
             response_model=EvidenceResponse,
             summary="Attach evaluation reports and approvals")
def attach_evidence(modelId: str, versionId: str, req: EvidenceAttachRequest):
    """
    Attach evidence (reports, approvals, logs) to a model version.

    **API spec inputs:** evidenceType, uri|blobRef, summary, approver, ticketId
    **Returns:** evidenceId, linked to model version
    """
    log = make_log(area="Model", component="Model Storage & Versioning", endpoint=f"/model/models/{modelId}/versions/{versionId}/evidence", meta={"modelId": modelId, "versionId": versionId, "evidenceType": req.evidenceType})
    try:
        result = svc.attach_evidence(
            model_id=modelId,
            version_id=versionId,
            evidence_type=req.evidenceType,
            summary=req.summary,
            uri=req.uri or req.blobRef,
            approver=req.approver,
        )
        return EvidenceResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "attach_evidence_failed", "message": str(e)}
        })


@router.get("/models/{modelId}/versions/{versionId}/evidence",
            response_model=List[EvidenceResponse],
            summary="Query evaluation reports and approvals")
def query_evidence(modelId: str, versionId: str):
    """
    Retrieve all evidence attached to a specific model version.
    """
    log = make_log(area="Model", component="Model Storage & Versioning", endpoint=f"/model/models/{modelId}/versions/{versionId}/evidence", meta={"modelId": modelId, "versionId": versionId})
    try:
        results = svc.list_evidence(modelId, versionId)
        return [EvidenceResponse(**r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "query_evidence_failed", "message": str(e)}
        })
