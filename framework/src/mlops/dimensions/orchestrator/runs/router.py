"""
Orchestrator – Runs endpoints.

  POST  /runs                          – Create an orchestrated execution
  GET   /runs                          – List runs (filter by ticketId, status)
  GET   /runs/{runId}                  – Read run status, timestamps, summary outcome
  GET   /runs/{runId}/trace            – Fetch end-to-end execution trace
  GET   /runs/{runId}/artifacts        – List artifacts produced/consumed during the run
  GET   /runs/{runId}/tickets          – List linked tickets (trigger, sub-tickets, escalations)
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from ....core.logs import make_log
from ..schemas import (
    RunCreate,
    RunResponse,
    RunStatusResponse,
    RunTraceResponse,
    RunArtifactsResponse,
    RunTicketsResponse,
)
from .. import orchestrator_service as svc

router = APIRouter()


# ─────────────────────────────────────────────
# POST /runs  — create
# GET  /runs  — list
# ─────────────────────────────────────────────

@router.post("/runs", response_model=RunResponse,
             summary="Create an orchestrated execution correlated to a ticket/contract")
def create_run(req: RunCreate):
    """
    **API spec inputs (POST):** ticketId, contractVersionId, requestedBy,
    inputArtifacts[], parameters?, targetEnv?
    **Returns:** runId, status, correlation_ids
    """
    make_log(
        area="Orchestrator",
        component="Runs",
        endpoint="/orchestrator/runs",
        meta={
            "ticketId": req.ticketId,
            "contractVersionId": req.contractVersionId,
            "requestedBy": req.requestedBy,
        },
    )
    try:
        result = svc.create_run(
            ticket_id=req.ticketId,
            contract_version_id=req.contractVersionId,
            requested_by=req.requestedBy,
            input_artifacts=req.inputArtifacts,
            parameters=req.parameters,
            target_env=req.targetEnv or "staging",
        )
        return RunResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "run_creation_failed", "message": str(e)}
        })


@router.get("/runs", response_model=List[RunResponse],
            summary="List orchestrated runs with optional filters")
def list_runs(
    ticketId: Optional[str] = Query(None, description="Filter by linked ticket ID"),
    status: Optional[str] = Query(None, description="Filter by run status"),
):
    make_log(
        area="Orchestrator",
        component="Runs",
        endpoint="/orchestrator/runs",
        meta={"ticketId": ticketId, "status": status},
    )
    try:
        return [RunResponse(**r) for r in svc.list_runs(ticket_id=ticketId, status=status)]
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "list_runs_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# GET /runs/{runId}  — status + summary
# ─────────────────────────────────────────────

@router.get("/runs/{runId}", response_model=RunStatusResponse,
            summary="Read run status, timestamps, and summary outcome")
def get_run_status(
    runId: str,
    includeSteps: bool = Query(True, description="Include step-by-step timeline"),
    includeGates: bool = Query(False, description="Include quality gate results"),
):
    """
    **API spec inputs:** includeSteps?, includeGates?
    **Returns:** runId, status_timeline, current_step, summary_metrics, gate_results?
    """
    make_log(
        area="Orchestrator",
        component="Runs",
        endpoint=f"/orchestrator/runs/{runId}",
        meta={"runId": runId, "includeSteps": includeSteps, "includeGates": includeGates},
    )
    run = svc.get_run(runId)
    if run is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "run_not_found", "message": f"Run '{runId}' not found"}
        })
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        timeline = []
        if includeSteps:
            timeline = [
                {"step_name": "Initialization", "status": "completed", "timestamp": now},
                {"step_name": "Data Ingestion",  "status": "completed", "timestamp": now},
            ]

        gate_results = None
        if includeGates:
            gate_results = [
                {"gate": "DataQuality", "passed": True, "approver": "System"}
            ]

        return RunStatusResponse(
            runId=runId,
            status_timeline=timeline,
            current_step=run.get("status", "unknown"),
            summary_metrics=run.get("parameters", {}),
            gate_results=gate_results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "run_status_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# GET /runs/{runId}/trace
# ─────────────────────────────────────────────

@router.get("/runs/{runId}/trace", response_model=RunTraceResponse,
            summary="Fetch end-to-end execution trace (steps, gates, decisions)")
def get_run_trace(
    runId: str,
    format: str = Query("json", description="Output format (json)"),
    includePayloads: bool = Query(False, description="Include full step payloads"),
):
    """
    Provides the full forensic record for audit and reproducibility.

    **API spec inputs:** format? (json), includePayloads?
    **Returns:** ordered trace: steps, decisions, gate results, actor actions
    """
    make_log(
        area="Orchestrator",
        component="Runs",
        endpoint=f"/orchestrator/runs/{runId}/trace",
        meta={"runId": runId, "format": format, "includePayloads": includePayloads},
    )
    if svc.get_run(runId) is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "run_not_found", "message": f"Run '{runId}' not found"}
        })
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        trace = [
            {
                "itemType": "step",
                "name": "Data Validation",
                "actor": "DataSvc-01",
                "timestamp": now,
                "outcome": "Success",
                "payload": {"records_processed": 1000} if includePayloads else None,
            },
            {
                "itemType": "gate",
                "name": "Quality Gate #1",
                "actor": "System",
                "timestamp": now,
                "outcome": "Passed",
                "payload": {"min_accuracy": 0.90, "actual": 0.94} if includePayloads else None,
            },
            {
                "itemType": "decision",
                "name": "Branching Logic",
                "actor": "Orchestrator",
                "timestamp": now,
                "outcome": "Route to Training",
                "payload": None,
            },
        ]
        return RunTraceResponse(runId=runId, format=format, ordered_trace=trace)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "run_trace_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# GET /runs/{runId}/artifacts
# ─────────────────────────────────────────────

@router.get("/runs/{runId}/artifacts", response_model=RunArtifactsResponse,
            summary="List artifacts produced/consumed during the run")
def get_run_artifacts(
    runId: str,
    direction: Optional[str] = Query(
        None,
        pattern="^(inputs|outputs)$",
        description="Filter by artifact role: inputs or outputs",
    ),
    type: Optional[str] = Query(None, description="Filter by artifact type"),
):
    """
    Provides technical lineage for a specific MLOps execution.

    **API spec inputs:** direction? (inputs|outputs), type?
    **Returns:** list of {artifactType, artifactId, versionId, role, producedAt}
    """
    make_log(
        area="Orchestrator",
        component="Runs",
        endpoint=f"/orchestrator/runs/{runId}/artifacts",
        meta={"runId": runId, "direction": direction, "type": type},
    )
    if svc.get_run(runId) is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "run_not_found", "message": f"Run '{runId}' not found"}
        })
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        all_artifacts = [
            {"artifactType": "dataset", "artifactId": "ds-training-01",
             "versionId": "v2024-05-20", "role": "input",  "producedAt": None},
            {"artifactType": "model",   "artifactId": "model-resnet-50",
             "versionId": "sha-7a8b9c", "role": "output", "producedAt": now},
        ]

        filtered = all_artifacts
        if direction:
            # "inputs" → role "input",  "outputs" → role "output"
            role_filter = direction.rstrip("s")
            filtered = [a for a in filtered if a["role"] == role_filter]
        if type:
            filtered = [a for a in filtered if a["artifactType"] == type]

        return RunArtifactsResponse(runId=runId, artifacts=filtered)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "run_artifacts_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# GET /runs/{runId}/tickets
# ─────────────────────────────────────────────

@router.get("/runs/{runId}/tickets", response_model=RunTicketsResponse,
            summary="List linked tickets (trigger, sub-tickets, escalations)")
def get_run_linked_tickets(
    runId: str,
    includeChildren: bool = Query(False, description="Include sub-tickets and escalations"),
):
    """
    Connects the technical execution back to the governance layer.

    **API spec inputs:** includeChildren?
    **Returns:** list of linked ticket refs
    """
    make_log(
        area="Orchestrator",
        component="Runs",
        endpoint=f"/orchestrator/runs/{runId}/tickets",
        meta={"runId": runId, "includeChildren": includeChildren},
    )
    run = svc.get_run(runId)
    if run is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "run_not_found", "message": f"Run '{runId}' not found"}
        })
    try:
        links = [
            {
                "ticketId": run["correlation_ids"]["ticketId"],
                "relationType": "trigger",
                "title": "Orchestrated Run Trigger",
                "state": "in-progress",
            }
        ]
        if includeChildren:
            links.append({
                "ticketId": "tkt-escalation-001",
                "relationType": "escalation",
                "title": "Data Drift Alert",
                "state": "open",
            })
        return RunTicketsResponse(runId=runId, linked_tickets=links)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "run_tickets_failed", "message": str(e)}
        })
