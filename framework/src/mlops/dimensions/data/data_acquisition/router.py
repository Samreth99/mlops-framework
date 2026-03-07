"""
Data Acquisition router — MLOps

Implements:
  POST   /data/sources                            – Register a source connector
  GET    /data/sources                            – List source connectors
  POST   /data/ingestions                         – Trigger ingestion (uploads file to S3 + DVC)
  GET    /data/ingestions                         – List ingestion executions
  GET    /data/ingestions/{ingestionId}           – Fetch ingestion config and run linkage
  GET    /data/ingestions/{ingestionId}/status    – Poll ingestion health / progress / outcome
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from ....core.logs import make_log
from ..schemas import (
    IngestionCreateRequest,
    IngestionDetailResponse,
    IngestionResponse,
    IngestionStatusResponse,
    SourceCreateRequest,
    SourceResponse,
)
from .. import s3_service as svc

router = APIRouter()


# ─────────────────────────────────────────────
# SOURCES
# ─────────────────────────────────────────────
@router.post(
    "/sources",
    response_model=SourceResponse,
    summary="Register a source connector",
)
def create_source(req: SourceCreateRequest):
    """
    Register a data source connector with credentials/connection metadata.

    **Inputs:** name, type, connectionRef, schemaRef?, owner, refreshCadence?
    **Returns:** sourceId, connector status
    """
    make_log(
        area="Data",
        component="Data Acquisition",
        endpoint="/data/sources",
        meta={"name": req.name, "type": req.type},
    )
    record = svc.create_source(
        name=req.name,
        type_=req.type,
        connection_ref=req.connectionRef,
        schema_ref=req.schemaRef,
        owner=req.owner,
        refresh_cadence=req.refreshCadence,
    )
    return SourceResponse(**record)


@router.get(
    "/sources",
    response_model=List[SourceResponse],
    summary="List all registered source connectors",
)
def list_sources():
    make_log(
        area="Data",
        component="Data Acquisition",
        endpoint="/data/sources",
    )
    return [SourceResponse(**s) for s in svc.list_sources()]


# ─────────────────────────────────────────────
# INGESTIONS
# ─────────────────────────────────────────────
@router.post(
    "/ingestions",
    response_model=IngestionResponse,
    summary="Trigger an ingestion job",
)
def create_ingestion(req: IngestionCreateRequest):
    """
    Trigger an ingestion job from a registered source.

    If `params.localFilePath` is supplied and the file exists, it is uploaded
    to the S3 bucket (`mlops-storage`) and tracked with DVC automatically.

    **Inputs:** sourceId, mode (batch|stream), window?, params?, ticketId?
    **Returns:** ingestionId, runId, accepted status
    """
    make_log(
        area="Data",
        component="Data Acquisition",
        endpoint="/data/ingestions",
        meta={"sourceId": req.sourceId, "mode": req.mode},
    )
    source = svc.get_source(req.sourceId)
    if not source:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "source_not_found", "message": f"Source '{req.sourceId}' not found"}},
        )
    try:
        record = svc.create_ingestion(
            source_id=req.sourceId,
            mode=req.mode,
            window=req.window,
            params=req.params,
            ticket_id=req.ticketId,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "invalid_request", "message": str(exc)}},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "ingestion_failed", "message": str(exc)}},
        )
    return IngestionResponse(
        ingestionId=record["ingestionId"],
        runId=record.get("runId"),
        sourceId=record["sourceId"],
        mode=record["mode"],
        status=record["status"],
        accepted=True,
        outputRefs=record.get("outputRefs", []),
        s3Ref=record.get("s3Ref"),
        localRef=record.get("localRef"),
        createdAt=record["createdAt"],
    )


@router.get(
    "/ingestions",
    response_model=List[IngestionResponse],
    summary="List all ingestion executions",
)
def list_ingestions():
    make_log(
        area="Data",
        component="Data Acquisition",
        endpoint="/data/ingestions",
    )
    return [
        IngestionResponse(
            ingestionId=r["ingestionId"],
            runId=r.get("runId"),
            sourceId=r["sourceId"],
            mode=r["mode"],
            status=r["status"],
            accepted=True,
            outputRefs=r.get("outputRefs", []),
            s3Ref=r.get("s3Ref"),
            localRef=r.get("localRef"),
            createdAt=r["createdAt"],
        )
        for r in svc.list_ingestions()
    ]


@router.get(
    "/ingestions/{ingestionId}",
    response_model=IngestionDetailResponse,
    summary="Fetch ingestion configuration and run linkage",
)
def get_ingestion(
    ingestionId: str,
    includeLogs: bool = Query(False, description="Include execution logs in response"),
):
    """
    Returns full config snapshot, linked S3 ref and dataset version (if produced).

    **Returns:** config snapshot, linked dataset version (if produced)
    """
    make_log(
        area="Data",
        component="Data Acquisition",
        endpoint=f"/data/ingestions/{ingestionId}",
        meta={"ingestionId": ingestionId, "includeLogs": includeLogs},
    )
    record = svc.get_ingestion(ingestionId)
    if not record:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "ingestion_not_found", "message": f"Ingestion '{ingestionId}' not found"}},
        )
    return IngestionDetailResponse(
        ingestionId=record["ingestionId"],
        runId=record.get("runId"),
        sourceId=record["sourceId"],
        sourceName=record.get("sourceName"),
        connectionRef=record.get("connectionRef"),
        mode=record["mode"],
        window=record.get("window"),
        params=record.get("params"),
        ticketId=record.get("ticketId"),
        status=record["status"],
        outputDatasetVersionId=record.get("outputDatasetVersionId"),
        s3Ref=record.get("s3Ref"),
        localRef=record.get("localRef"),
        logs=record.get("logs") if includeLogs else None,
        createdAt=record["createdAt"],
    )


@router.get(
    "/ingestions/{ingestionId}/status",
    response_model=IngestionStatusResponse,
    summary="Poll ingestion health / progress / outcome",
)
def get_ingestion_status(
    ingestionId: str,
    verbose: bool = Query(False, description="Include error details"),
):
    """
    **Returns:** state, progress %, error summary, output refs
    """
    make_log(
        area="Data",
        component="Data Acquisition",
        endpoint=f"/data/ingestions/{ingestionId}/status",
        meta={"ingestionId": ingestionId},
    )
    record = svc.get_ingestion(ingestionId)
    if not record:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "ingestion_not_found", "message": f"Ingestion '{ingestionId}' not found"}},
        )
    error_summary: Optional[str] = None
    if verbose and record.get("logs"):
        errors = [l for l in record["logs"] if "error" in l.lower()]
        error_summary = "; ".join(errors) if errors else None

    return IngestionStatusResponse(
        ingestionId=ingestionId,
        state=record["status"],
        progress=record.get("progress", 0.0),
        errorSummary=error_summary,
        outputRefs=record.get("outputRefs", []),
    )
