"""
Feature Storage & Versioning router — MLOps

Implements:
  POST   /data/features                                      – Register a logical feature set
  GET    /data/features                                      – List feature sets
  GET    /data/features/{featureSetId}                       – Feature set metadata
  POST   /data/features/{featureSetId}                       – Update feature set metadata
  POST   /data/features/{featureSetId}/versions              – Create/publish a feature version (S3 + DVC)
  GET    /data/features/{featureSetId}/versions              – List feature versions
  GET    /data/features/{featureSetId}/versions/{versionId}  – Fetch a specific version pointer
  GET    /data/features/online/get                           – Retrieve online features by entity keys
  GET    /data/features/offline/get                          – Retrieve offline feature table ref
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from ....core.logs import make_log
from ..schemas import (
    FeatureSetCreateRequest,
    FeatureSetResponse,
    FeatureSetUpdateRequest,
    FeatureVersionCreateRequest,
    FeatureVersionResponse,
    OfflineFeatureGetResponse,
    OnlineFeatureGetResponse,
)
from .. import s3_service as svc

router = APIRouter()


# ─────────────────────────────────────────────
# Note: /online/get and /offline/get must be
# declared BEFORE /{featureSetId} so FastAPI
# does not try to match "online" as a featureSetId
# ─────────────────────────────────────────────
@router.get(
    "/features/online/get",
    response_model=OnlineFeatureGetResponse,
    summary="Retrieve online features for serving by entity keys",
)
def get_online_features(
    featureSetId: str = Query(..., description="Feature set ID"),
    versionId: Optional[str] = Query(None, description="Specific version (latest if omitted)"),
    entityKeyValues: str = Query(..., description='JSON-encoded entity key map, e.g. {"patient_id":"42"}'),
    asOfTime: Optional[str] = Query(None, description="ISO timestamp for point-in-time lookup"),
):
    """
    Return feature vectors for given entity key values from the registered feature set.

    **Returns:** feature vectors + retrieval metadata
    """
    import json

    make_log(
        area="Data",
        component="Feature Storage & Versioning",
        endpoint="/data/features/online/get",
        meta={"featureSetId": featureSetId, "versionId": versionId},
    )

    fs = svc.get_feature_set(featureSetId)
    if not fs:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "feature_set_not_found", "message": f"Feature set '{featureSetId}' not found"}},
        )

    try:
        entity_kv: Dict[str, Any] = json.loads(entityKeyValues)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "invalid_entity_keys", "message": "entityKeyValues must be valid JSON"}},
        )

    # Resolve version
    versions = svc.list_feature_versions(featureSetId)
    target_version: Optional[Dict[str, Any]] = None
    if versionId:
        target_version = svc.get_feature_version(featureSetId, versionId)
    elif versions:
        target_version = sorted(versions, key=lambda v: v["createdAt"], reverse=True)[0]

    features: Dict[str, Any] = {}
    if target_version:
        storage_ref = target_version.get("storageRef", "")
        try:
            from pathlib import Path
            import pandas as pd

            local_ref = target_version.get("localRef") or storage_ref
            if Path(local_ref).exists():
                df = pd.read_csv(local_ref)
                # Filter by entity key values
                mask = None
                for key, val in entity_kv.items():
                    if key in df.columns:
                        m = df[key].astype(str) == str(val)
                        mask = m if mask is None else (mask & m)
                if mask is not None:
                    row = df[mask].head(1)
                    if not row.empty:
                        features = row.iloc[0].to_dict()
        except Exception:
            pass

    return OnlineFeatureGetResponse(
        featureSetId=featureSetId,
        versionId=target_version["versionId"] if target_version else versionId,
        entityKeyValues=entity_kv,
        features=features,
        asOfTime=asOfTime,
        retrievalMetadata={
            "storageRef": target_version.get("storageRef") if target_version else None,
            "featureSetName": fs.get("name"),
        },
    )


@router.get(
    "/features/offline/get",
    response_model=OfflineFeatureGetResponse,
    summary="Retrieve offline feature table ref for training",
)
def get_offline_features(
    featureSetId: str = Query(..., description="Feature set ID"),
    versionId: Optional[str] = Query(None, description="Specific version (latest if omitted)"),
    entityKeyColumn: Optional[str] = Query(None, description="Column name of the entity key"),
    timeWindow: Optional[str] = Query(None, description="Time window: e.g. 2024-01-01/2024-12-31"),
    joinConfig: Optional[str] = Query(None, description="JSON-encoded join configuration"),
):
    """
    Return the offline feature table reference (S3 URI) for training use.

    **Returns:** offline feature table ref + schema
    """
    make_log(
        area="Data",
        component="Feature Storage & Versioning",
        endpoint="/data/features/offline/get",
        meta={"featureSetId": featureSetId, "versionId": versionId},
    )

    fs = svc.get_feature_set(featureSetId)
    if not fs:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "feature_set_not_found", "message": f"Feature set '{featureSetId}' not found"}},
        )

    versions = svc.list_feature_versions(featureSetId)
    target_version: Optional[Dict[str, Any]] = None
    if versionId:
        target_version = svc.get_feature_version(featureSetId, versionId)
    elif versions:
        target_version = sorted(versions, key=lambda v: v["createdAt"], reverse=True)[0]

    if not target_version:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "no_versions", "message": f"No versions found for feature set '{featureSetId}'"}},
        )

    schema: Dict[str, Any] = {}
    row_count: Optional[int] = None
    try:
        from pathlib import Path
        import pandas as pd

        local_ref = target_version.get("localRef") or target_version.get("storageRef", "")
        if Path(local_ref).exists():
            df = pd.read_csv(local_ref)
            schema = {c: str(t) for c, t in df.dtypes.items()}
            row_count = len(df)
    except Exception:
        pass

    return OfflineFeatureGetResponse(
        featureSetId=featureSetId,
        versionId=target_version["versionId"],
        storageRef=target_version["storageRef"],
        columnSchema=schema,
        rowCount=row_count,
    )


# ─────────────────────────────────────────────
# FEATURE SETS
# ─────────────────────────────────────────────
@router.post(
    "/features",
    response_model=FeatureSetResponse,
    summary="Register a logical feature set",
)
def create_feature_set(req: FeatureSetCreateRequest):
    """
    **Inputs:** name, owner, entitySchema, description
    **Returns:** featureSetId
    """
    make_log(
        area="Data",
        component="Feature Storage & Versioning",
        endpoint="/data/features",
        meta={"name": req.name},
    )
    record = svc.create_feature_set(
        name=req.name,
        owner=req.owner,
        entity_schema=req.entitySchema,
        description=req.description,
    )
    return FeatureSetResponse(**record)


@router.get(
    "/features",
    response_model=List[FeatureSetResponse],
    summary="List all registered feature sets",
)
def list_feature_sets():
    make_log(
        area="Data",
        component="Feature Storage & Versioning",
        endpoint="/data/features",
    )
    return [FeatureSetResponse(**fs) for fs in svc.list_feature_sets()]


@router.get(
    "/features/{featureSetId}",
    response_model=FeatureSetResponse,
    summary="Get feature set metadata",
)
def get_feature_set(featureSetId: str):
    make_log(
        area="Data",
        component="Feature Storage & Versioning",
        endpoint=f"/data/features/{featureSetId}",
        meta={"featureSetId": featureSetId},
    )
    record = svc.get_feature_set(featureSetId)
    if not record:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "feature_set_not_found", "message": f"Feature set '{featureSetId}' not found"}},
        )
    return FeatureSetResponse(**record)


@router.post(
    "/features/{featureSetId}",
    response_model=FeatureSetResponse,
    summary="Update feature set metadata",
)
def update_feature_set(featureSetId: str, req: FeatureSetUpdateRequest):
    make_log(
        area="Data",
        component="Feature Storage & Versioning",
        endpoint=f"/data/features/{featureSetId}",
        meta={"featureSetId": featureSetId},
    )
    record = svc.update_feature_set(
        featureSetId,
        description=req.description,
        owner=req.owner,
        entitySchema=req.entitySchema,
    )
    if not record:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "feature_set_not_found", "message": f"Feature set '{featureSetId}' not found"}},
        )
    return FeatureSetResponse(**record)


# ─────────────────────────────────────────────
# FEATURE VERSIONS
# ─────────────────────────────────────────────
@router.post(
    "/features/{featureSetId}/versions",
    response_model=FeatureVersionResponse,
    summary="Create / publish a feature version (S3 upload + DVC)",
)
def create_feature_version(featureSetId: str, req: FeatureVersionCreateRequest):
    """
    Upload feature table to S3 and track with DVC.

    **Inputs:** storageRef (local path or S3 URI), schemaRef?, computedFrom (datasetVersionId)?, ticketId?
    **Returns:** versionId, digest/checksum, storageRef (S3 URI)
    """
    make_log(
        area="Data",
        component="Feature Storage & Versioning",
        endpoint=f"/data/features/{featureSetId}/versions",
        meta={"featureSetId": featureSetId, "storageRef": req.storageRef},
    )
    try:
        version = svc.create_feature_version(
            fs_id=featureSetId,
            storage_ref=req.storageRef,
            schema_ref=req.schemaRef,
            computed_from=req.computedFrom,
            ticket_id=req.ticketId,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "feature_set_not_found", "message": str(exc)}},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "version_creation_failed", "message": str(exc)}},
        )
    return FeatureVersionResponse(
        versionId=version["versionId"],
        featureSetId=version["featureSetId"],
        storageRef=version["storageRef"],
        schemaRef=version.get("schemaRef"),
        computedFrom=version.get("computedFrom"),
        digest=version.get("digest"),
        dvcTracked=version.get("dvcTracked", False),
        createdAt=version["createdAt"],
    )


@router.get(
    "/features/{featureSetId}/versions",
    response_model=List[FeatureVersionResponse],
    summary="List all versions of a feature set",
)
def list_feature_versions(featureSetId: str):
    make_log(
        area="Data",
        component="Feature Storage & Versioning",
        endpoint=f"/data/features/{featureSetId}/versions",
        meta={"featureSetId": featureSetId},
    )
    return [
        FeatureVersionResponse(
            versionId=v["versionId"],
            featureSetId=v["featureSetId"],
            storageRef=v["storageRef"],
            schemaRef=v.get("schemaRef"),
            computedFrom=v.get("computedFrom"),
            digest=v.get("digest"),
            dvcTracked=v.get("dvcTracked", False),
            createdAt=v["createdAt"],
        )
        for v in svc.list_feature_versions(featureSetId)
    ]


@router.get(
    "/features/{featureSetId}/versions/{versionId}",
    response_model=FeatureVersionResponse,
    summary="Fetch a specific feature version pointer",
)
def get_feature_version(featureSetId: str, versionId: str):
    make_log(
        area="Data",
        component="Feature Storage & Versioning",
        endpoint=f"/data/features/{featureSetId}/versions/{versionId}",
        meta={"featureSetId": featureSetId, "versionId": versionId},
    )
    version = svc.get_feature_version(featureSetId, versionId)
    if not version:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "version_not_found", "message": f"Version '{versionId}' not found"}},
        )
    return FeatureVersionResponse(
        versionId=version["versionId"],
        featureSetId=version["featureSetId"],
        storageRef=version["storageRef"],
        schemaRef=version.get("schemaRef"),
        computedFrom=version.get("computedFrom"),
        digest=version.get("digest"),
        dvcTracked=version.get("dvcTracked", False),
        createdAt=version["createdAt"],
    )
