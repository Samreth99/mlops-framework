"""
Feature Storage & Versioning router — MLOps

Implements:
  POST   /data/features                                      – Register a logical feature set
  GET    /data/features                                      – List feature sets
  GET    /data/features/online/get                           – Retrieve online features by entity keys (Redis)
  POST   /data/features/online/store                         – Write a single entity's features to Redis
  GET    /data/features/offline/get                          – Retrieve offline feature table ref (S3)
  GET    /data/features/{featureSetId}                       – Feature set metadata
  POST   /data/features/{featureSetId}                       – Update feature set metadata
  POST   /data/features/{featureSetId}/versions              – Create/publish a feature version (S3 + DVC)
  GET    /data/features/{featureSetId}/versions              – List feature versions
  GET    /data/features/{featureSetId}/versions/{versionId}  – Fetch a specific version pointer

Online vs Offline store:
  Online  → Redis hash store (O(1) key lookup, sub-millisecond serving).
             Populated via POST /features/online/store after a prediction is made for a new entity.
             On return visits the entity's pre-computed features are served directly from Redis.
  Offline → S3 CSV/Parquet URI returned for bulk training use.
             No data is loaded into memory; only schema and row count are derived.
"""
from __future__ import annotations

import logging
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
    OnlineFeatureStoreRequest,
    OnlineFeatureStoreResponse,
    OnlineMaterializeRequest,
    OnlineMaterializeResponse,
)
from .. import s3_service as svc
from .. import redis_service as redis_svc

logger = logging.getLogger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────
# Note: /online/get and /offline/get MUST be
# declared BEFORE /{featureSetId} so FastAPI
# does not match "online" / "offline" as a
# featureSetId path parameter.
# ─────────────────────────────────────────────
@router.get(
    "/features/online/get",
    response_model=OnlineFeatureGetResponse,
    summary="Retrieve online features for serving by entity keys (Redis)",
)
def get_online_features(
    featureSetId: str = Query(..., description="Feature set ID"),
    versionId: Optional[str] = Query(None, description="Specific version (latest if omitted)"),
    entityKeyValues: str = Query(..., description='JSON-encoded entity key map, e.g. {"patient_id":"42"}'),
    asOfTime: Optional[str] = Query(None, description="ISO timestamp for point-in-time lookup"),
):
    """
    Serve pre-computed feature vectors for real-time inference (returning entities only).

    Looks up `fsv:{featureSetId}:{versionId}:{entityKeyValues}` in Redis.
    Returns `features: {}` with `storeUsed: "miss"` when the entity is not found —
    this means it is a new entity whose features have not been stored yet.

    Use `POST /features/online/store` to write features after a first-visit prediction.
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
        if not target_version:
            raise HTTPException(
                status_code=404,
                detail={"error": {"code": "version_not_found", "message": f"Version '{versionId}' not found"}},
            )
    elif versions:
        target_version = sorted(versions, key=lambda v: v["createdAt"], reverse=True)[0]

    features: Dict[str, Any] = {}
    store_used = "miss"

    if target_version:
        resolved_version_id = target_version["versionId"]
        redis_result = redis_svc.get_features(featureSetId, resolved_version_id, entity_kv)
        if redis_result is not None:
            features = redis_result
            store_used = "redis"
        else:
            logger.debug(
                "Redis miss for fs=%s v=%s entity=%s",
                featureSetId, resolved_version_id, entity_kv,
            )

    return OnlineFeatureGetResponse(
        featureSetId=featureSetId,
        versionId=target_version["versionId"] if target_version else versionId,
        entityKeyValues=entity_kv,
        features=features,
        storeUsed=store_used,
        asOfTime=asOfTime,
        retrievalMetadata={
            "storageRef": target_version.get("storageRef") if target_version else None,
            "featureSetName": fs.get("name"),
        },
    )


@router.post(
    "/features/online/store",
    response_model=OnlineFeatureStoreResponse,
    summary="Store a single entity's features in Redis (write-on-first-visit)",
)
def store_online_features(req: OnlineFeatureStoreRequest):
    """
    Write a single entity's pre-computed features into the Redis online store.

    **When to call this:**
    After a model makes a first-visit prediction for a new entity (e.g. a new patient),
    call this endpoint to cache that entity's features. On subsequent visits,
    `GET /features/online/get` will return the cached features in O(1) without
    recomputing them.

    **`overwrite: false`** skips the write if the entity's key already exists in Redis,
    preserving the original features from the first visit.
    """
    make_log(
        area="Data",
        component="Feature Storage & Versioning",
        endpoint="/data/features/online/store",
        meta={"featureSetId": req.featureSetId, "versionId": req.versionId},
    )

    fs = svc.get_feature_set(req.featureSetId)
    if not fs:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "feature_set_not_found", "message": f"Feature set '{req.featureSetId}' not found"}},
        )

    version = svc.get_feature_version(req.featureSetId, req.versionId)
    if not version:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "version_not_found", "message": f"Version '{req.versionId}' not found"}},
        )

    stored = redis_svc.store_features(
        fs_id=req.featureSetId,
        version_id=req.versionId,
        entity_kv=req.entityKeyValues,
        features=req.features,
        overwrite=req.overwrite,
    )

    if stored:
        msg = "Features stored in Redis online store."
    elif req.overwrite:
        msg = "Redis unavailable — features could not be stored."
    else:
        msg = "Key already exists in Redis; write skipped (overwrite=false)."

    return OnlineFeatureStoreResponse(
        featureSetId=req.featureSetId,
        versionId=req.versionId,
        entityKeyValues=req.entityKeyValues,
        stored=stored,
        message=msg,
    )


@router.post(
    "/features/online/materialize",
    response_model=OnlineMaterializeResponse,
    summary="Bulk load entire feature CSV into Redis online store",
)
def materialize_online_features(req: OnlineMaterializeRequest):
    """
    Bulk materialize a full feature CSV into the Redis online store.

    Called once after POST /features/{id}/versions during pipeline initialization.
    Loads every row from the local CSV into Redis in batches of 500.

    Use POST /features/online/store for single-entity writes at inference time.
    """
    make_log(
        area="Data",
        component="Feature Storage & Versioning",
        endpoint="/data/features/online/materialize",
        meta={"featureSetId": req.featureSetId, "versionId": req.versionId, "localRef": req.localRef},
    )

    fs = svc.get_feature_set(req.featureSetId)
    if not fs:
        raise HTTPException(status_code=404, detail={"error": {"code": "feature_set_not_found", "message": f"Feature set '{req.featureSetId}' not found"}})

    target = svc.get_feature_version(req.featureSetId, req.versionId)
    if not target:
        raise HTTPException(status_code=404, detail={"error": {"code": "version_not_found", "message": f"Version '{req.versionId}' not found"}})

    if not redis_svc.is_redis_available():
        raise HTTPException(status_code=503, detail={"error": {"code": "redis_unavailable", "message": "Redis is not reachable"}})

    try:
        result = redis_svc.materialize_to_redis(
            csv_path=req.localRef,
            fs_id=req.featureSetId,
            version_id=req.versionId,
            entity_keys=list(req.entityKeys),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail={"error": {"code": "materialize_error", "message": str(exc)}})

    return OnlineMaterializeResponse(
        featureSetId=req.featureSetId,
        versionId=req.versionId,
        materialized=result["materialized"],
        errors=result["errors"],
        skippedMissingKeys=result["skipped_missing_keys"],
        message=f"Materialized {result['materialized']} rows into Redis. Errors: {result['errors']}.",
    )


@router.get(
    "/features/offline/get",
    response_model=OfflineFeatureGetResponse,
    summary="Retrieve offline feature table ref for training (S3)",
)
def get_offline_features(
    featureSetId: str = Query(..., description="Feature set ID"),
    versionId: Optional[str] = Query(None, description="Specific version (latest if omitted)"),
    entityKeyColumn: Optional[str] = Query(None, description="Column name of the entity key"),
    timeWindow: Optional[str] = Query(None, description="Time window: e.g. 2024-01-01/2024-12-31"),
    joinConfig: Optional[str] = Query(None, description="JSON-encoded join configuration"),
):
    """
    Return the offline feature table S3 URI for training pipelines.

    Schema and row count are derived without loading the full file into memory.

    **Returns:** S3 storageRef + column schema + row count
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
        if not target_version:
            raise HTTPException(
                status_code=404,
                detail={"error": {"code": "version_not_found", "message": f"Version '{versionId}' not found"}},
            )
    elif versions:
        target_version = sorted(versions, key=lambda v: v["createdAt"], reverse=True)[0]

    if not target_version:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "no_versions", "message": f"No versions found for feature set '{featureSetId}'"}},
        )

    schema: Dict[str, Any] = {}
    row_count: Optional[int] = None
    local_ref = target_version.get("localRef") or target_version.get("storageRef", "")
    try:
        from pathlib import Path
        import pandas as pd

        if Path(local_ref).exists():
            first_chunk = next(pd.read_csv(local_ref, chunksize=1))
            schema = {c: str(t) for c, t in first_chunk.dtypes.items()}
            with open(local_ref, "rb") as fh:
                row_count = sum(1 for _ in fh) - 1  # subtract header line
    except FileNotFoundError as exc:
        logger.warning("Offline feature CSV not found for v=%s: %s", target_version.get("versionId"), exc)
    except Exception as exc:
        logger.warning("Failed to read offline feature schema for v=%s: %s", target_version.get("versionId"), exc)

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
    summary="Create / publish a feature version (S3 + DVC)",
)
def create_feature_version(featureSetId: str, req: FeatureVersionCreateRequest):
    """
    Upload feature table to S3 and track with DVC.

    **Inputs:**
    - `storageRef` — local CSV path or S3 URI
    - `entityKeys` — column names that identify each entity (e.g. `["patient_id"]`).
      Stored as metadata; used by `POST /features/online/store` at serving time.
    - `schemaRef`, `computedFrom`, `ticketId` — optional metadata

    **Returns:** versionId, digest, storageRef, entityKeys
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
            entity_keys=req.entityKeys or None,
            ticket_id=req.ticketId,
        )
    except LookupError as exc:
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
        localRef=version.get("localRef"),
        trainLocalRef=version.get("trainLocalRef"),
        testLocalRef=version.get("testLocalRef"),
        trainStorageRef=version.get("trainStorageRef"),
        testStorageRef=version.get("testStorageRef"),
        schemaRef=version.get("schemaRef"),
        computedFrom=version.get("computedFrom"),
        digest=version.get("digest"),
        dvcTracked=version.get("dvcTracked", False),
        entityKeys=version.get("entityKeys", []),
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
            localRef=v.get("localRef"),
            trainLocalRef=v.get("trainLocalRef"),
            testLocalRef=v.get("testLocalRef"),
            trainStorageRef=v.get("trainStorageRef"),
            testStorageRef=v.get("testStorageRef"),
            schemaRef=v.get("schemaRef"),
            computedFrom=v.get("computedFrom"),
            digest=v.get("digest"),
            dvcTracked=v.get("dvcTracked", False),
            entityKeys=v.get("entityKeys", []),
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
        localRef=version.get("localRef"),
        trainLocalRef=version.get("trainLocalRef"),
        testLocalRef=version.get("testLocalRef"),
        trainStorageRef=version.get("trainStorageRef"),
        testStorageRef=version.get("testStorageRef"),
        schemaRef=version.get("schemaRef"),
        computedFrom=version.get("computedFrom"),
        digest=version.get("digest"),
        dvcTracked=version.get("dvcTracked", False),
        entityKeys=version.get("entityKeys", []),
        createdAt=version["createdAt"],
    )
