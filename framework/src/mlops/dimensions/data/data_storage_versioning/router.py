"""
Data Storage & Versioning router — MLOps

Implements:
  POST   /data/datasets                               – Register a logical dataset
  GET    /data/datasets                               – List all datasets
  GET    /data/datasets/{datasetId}                   – Dataset metadata
  POST   /data/datasets/{datasetId}                   – Update dataset metadata
  POST   /data/datasets/{datasetId}/versions          – Create a dataset version (S3 upload + DVC)
  GET    /data/datasets/{datasetId}/versions          – List dataset versions
  GET    /data/datasets/{datasetId}/versions/{vid}    – Fetch a specific version pointer
  GET    /data/datasets/{datasetId}/lineage           – Provenance graph
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from ....core.logs import make_log
from ..schemas import (
    DatasetCreateRequest,
    DatasetResponse,
    DatasetUpdateRequest,
    DatasetVersionCreateRequest,
    DatasetVersionResponse,
    LineageResponse,
)
from .. import s3_service as svc

router = APIRouter()


# ─────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────
@router.post(
    "/datasets",
    response_model=DatasetResponse,
    summary="Register a logical dataset",
)
def create_dataset(req: DatasetCreateRequest):
    """
    Register a new logical dataset (not an individual version).

    **Inputs:** name, owner, domain, description, schemaRef?
    **Returns:** datasetId + snapshot
    """
    make_log(
        area="Data",
        component="Data Storage & Versioning",
        endpoint="/data/datasets",
        meta={"name": req.name},
    )
    record = svc.create_dataset(
        name=req.name,
        owner=req.owner,
        domain=req.domain,
        description=req.description,
        schema_ref=req.schemaRef,
    )
    return DatasetResponse(**record)


@router.get(
    "/datasets",
    response_model=List[DatasetResponse],
    summary="List all registered datasets",
)
def list_datasets():
    make_log(
        area="Data",
        component="Data Storage & Versioning",
        endpoint="/data/datasets",
    )
    return [DatasetResponse(**d) for d in svc.list_datasets()]


@router.get(
    "/datasets/{datasetId}",
    response_model=DatasetResponse,
    summary="Get dataset metadata",
)
def get_dataset(datasetId: str):
    make_log(
        area="Data",
        component="Data Storage & Versioning",
        endpoint=f"/data/datasets/{datasetId}",
        meta={"datasetId": datasetId},
    )
    record = svc.get_dataset(datasetId)
    if not record:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "dataset_not_found", "message": f"Dataset '{datasetId}' not found"}},
        )
    return DatasetResponse(**record)


@router.post(
    "/datasets/{datasetId}",
    response_model=DatasetResponse,
    summary="Update dataset metadata",
)
def update_dataset(datasetId: str, req: DatasetUpdateRequest):
    make_log(
        area="Data",
        component="Data Storage & Versioning",
        endpoint=f"/data/datasets/{datasetId}",
        meta={"datasetId": datasetId},
    )
    update_kwargs: dict = {}
    if req.description is not None:
        update_kwargs["description"] = req.description
    if req.owner is not None:
        update_kwargs["owner"] = req.owner
    if req.schemaRef is not None:
        update_kwargs["schemaRef"] = req.schemaRef

    record = svc.update_dataset(datasetId, **update_kwargs)
    if not record:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "dataset_not_found", "message": f"Dataset '{datasetId}' not found"}},
        )
    return DatasetResponse(**record)


# ─────────────────────────────────────────────
# DATASET VERSIONS
# ─────────────────────────────────────────────
@router.post(
    "/datasets/{datasetId}/versions",
    response_model=DatasetVersionResponse,
    summary="Create a dataset version (S3 upload + DVC tracking)",
)
def create_dataset_version(datasetId: str, req: DatasetVersionCreateRequest):
    """
    Upload the dataset file to the S3 bucket via boto3 and track it with DVC.

    **Inputs:** storageRef (local path or S3 URI), schemaRef?, lineage?, stats?, ticketId?
    **Returns:** versionId, immutable digest/checksum, storageRef (S3 URI)
    """
    make_log(
        area="Data",
        component="Data Storage & Versioning",
        endpoint=f"/data/datasets/{datasetId}/versions",
        meta={"datasetId": datasetId, "storageRef": req.storageRef},
    )
    try:
        version = svc.create_dataset_version(
            dataset_id=datasetId,
            storage_ref=req.storageRef,
            schema_ref=req.schemaRef,
            lineage=req.lineage,
            stats=req.stats,
            ticket_id=req.ticketId,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "dataset_not_found", "message": str(exc)}},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "version_creation_failed", "message": str(exc)}},
        )
    return DatasetVersionResponse(
        versionId=version["versionId"],
        datasetId=version["datasetId"],
        storageRef=version["storageRef"],
        localRef=version.get("localRef"),
        trainStorageRef=version.get("trainStorageRef"),
        testStorageRef=version.get("testStorageRef"),
        trainLocalRef=version.get("trainLocalRef"),
        testLocalRef=version.get("testLocalRef"),
        digest=version.get("digest"),
        schemaRef=version.get("schemaRef"),
        lineage=version.get("lineage"),
        stats=version.get("stats"),
        dvcTracked=version.get("dvcTracked", False),
        createdBy=version.get("createdBy"),
        createdAt=version["createdAt"],
    )


@router.get(
    "/datasets/{datasetId}/versions",
    response_model=List[DatasetVersionResponse],
    summary="List all versions of a dataset",
)
def list_dataset_versions(datasetId: str):
    make_log(
        area="Data",
        component="Data Storage & Versioning",
        endpoint=f"/data/datasets/{datasetId}/versions",
        meta={"datasetId": datasetId},
    )
    versions = svc.list_dataset_versions(datasetId)
    return [
        DatasetVersionResponse(
            versionId=v["versionId"],
            datasetId=v["datasetId"],
            storageRef=v["storageRef"],
            trainStorageRef=v.get("trainStorageRef"),
            testStorageRef=v.get("testStorageRef"),
            digest=v.get("digest"),
            schemaRef=v.get("schemaRef"),
            lineage=v.get("lineage"),
            stats=v.get("stats"),
            dvcTracked=v.get("dvcTracked", False),
            createdBy=v.get("createdBy"),
            createdAt=v["createdAt"],
        )
        for v in versions
    ]


@router.get(
    "/datasets/{datasetId}/versions/{versionId}",
    response_model=DatasetVersionResponse,
    summary="Fetch a specific dataset version pointer",
)
def get_dataset_version(datasetId: str, versionId: str):
    """
    Returns the S3 storage pointer, schema, lineage, stats and digest for one version.
    """
    make_log(
        area="Data",
        component="Data Storage & Versioning",
        endpoint=f"/data/datasets/{datasetId}/versions/{versionId}",
        meta={"datasetId": datasetId, "versionId": versionId},
    )
    version = svc.get_dataset_version(datasetId, versionId)
    if not version:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "version_not_found", "message": f"Version '{versionId}' not found"}},
        )
    return DatasetVersionResponse(
        versionId=version["versionId"],
        datasetId=version["datasetId"],
        storageRef=version["storageRef"],
        localRef=version.get("localRef"),
        trainStorageRef=version.get("trainStorageRef"),
        testStorageRef=version.get("testStorageRef"),
        trainLocalRef=version.get("trainLocalRef"),
        testLocalRef=version.get("testLocalRef"),
        digest=version.get("digest"),
        schemaRef=version.get("schemaRef"),
        lineage=version.get("lineage"),
        stats=version.get("stats"),
        dvcTracked=version.get("dvcTracked", False),
        createdBy=version.get("createdBy"),
        createdAt=version["createdAt"],
    )


# ─────────────────────────────────────────────
# LINEAGE
# ─────────────────────────────────────────────
@router.get(
    "/datasets/{datasetId}/lineage",
    response_model=LineageResponse,
    summary="Query dataset provenance graph",
)
def get_dataset_lineage(
    datasetId: str,
    fromVersion: Optional[str] = Query(None, description="Start from a specific versionId"),
    depth: int = Query(3, description="Max traversal depth"),
):
    """
    Returns lineage nodes and edges: dataset → versions → parent versions → transforms → sources.
    """
    make_log(
        area="Data",
        component="Data Storage & Versioning",
        endpoint=f"/data/datasets/{datasetId}/lineage",
        meta={"datasetId": datasetId, "fromVersion": fromVersion, "depth": depth},
    )
    lineage = svc.get_dataset_lineage(datasetId, from_version=fromVersion, depth=depth)
    return LineageResponse(**lineage)
