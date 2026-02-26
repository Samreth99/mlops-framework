"""
Orchestrator – Orchestration control endpoints.

  POST  /orchestrator/context          – Set orchestration context (env, defaults, policies)
  POST  /orchestrator/start            – DisPOST initial START messages / bootstrap runs
  POST  /orchestrator/health-checks/run – Trigger periodic health check workflow
  POST  /router/classify               – Classify a ticket into a routing action
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ....core.logs import make_log
from ..schemas import (
    ContextSetRequest,
    ContextResponse,
    StartOrchestrationRequest,
    StartOrchestrationResponse,
    HealthCheckRequest,
    HealthCheckResponse,
    ClassificationRequest,
    ClassificationResponse,
)
from .. import orchestrator_service as svc

router = APIRouter()


# ─────────────────────────────────────────────
# POST /orchestrator/context
# ─────────────────────────────────────────────

@router.post("/orchestrator/context", response_model=ContextResponse,
             summary="Set orchestration context (env, business key, policies, defaults)")
def set_orchestration_context(req: ContextSetRequest):
    """
    Defines the environment and defaults for the entire MLOps lifecycle.
    Aligns the MPO with the Project Management layer.

    **API spec inputs:** businessKey?, systemId?, env, defaultContractRefs?[],
    defaultPolicies?, tags?[]
    **Returns:** contextId, storedContextRef, effectiveDefaults
    """
    make_log(
        area="Orchestrator",
        component="Orchestration",
        endpoint="/orchestrator/orchestrator/context",
        meta={"env": req.env, "businessKey": req.businessKey},
    )
    try:
        result = svc.set_context(
            env=req.env,
            business_key=req.businessKey,
            system_id=req.systemId,
            default_contract_refs=req.defaultContractRefs,
            default_policies=req.defaultPolicies,
            tags=req.tags,
        )
        return ContextResponse(
            contextId=result["contextId"],
            storedContextRef=result["storedContextRef"],
            effectiveDefaults=result["effectiveDefaults"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "context_set_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# POST /orchestrator/start
# ─────────────────────────────────────────────

@router.post("/orchestrator/start", response_model=StartOrchestrationResponse,
             summary="Bootstrap orchestrated runs — dispatch START messages across dimensions")
def start_orchestration(req: StartOrchestrationRequest):
    """
    Triggers the initial START signals for the MLOps pipeline.
    Creates a Run ID for each target dimension (DATA, MODEL, SOFT, OPS).

    **API spec inputs:** contextRef|contextId?, startTargets[] (DATA|MODEL|SOFT|OPS),
    ticketId?, parameters?
    **Returns:** disPOSTId (dispatchId), runId(s), accepted=true, emittedEventIds[]
    """
    make_log(
        area="Orchestrator",
        component="Orchestration",
        endpoint="/orchestrator/orchestrator/start",
        meta={"startTargets": req.startTargets, "ticketId": req.ticketId},
    )
    try:
        result = svc.start_orchestration(
            start_targets=req.startTargets,
            context_id=req.contextId,
            ticket_id=req.ticketId,
            parameters=req.parameters,
        )
        return StartOrchestrationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "orchestration_start_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# POST /orchestrator/health-checks/run
# ─────────────────────────────────────────────

@router.post("/orchestrator/health-checks/run", response_model=HealthCheckResponse,
             summary="Trigger a periodic health check workflow")
def trigger_health_check(req: HealthCheckRequest):
    """
    Validates the operational environment before MLOps activities begin.
    Ensures Ops and Infra dimensions are ready.

    **API spec inputs:** contextRef|contextId?, scope (services|pipelines|infra), ticketId?
    **Returns:** healthCheckRunId, status, emittedEventIds[]
    """
    make_log(
        area="Orchestrator",
        component="Orchestration",
        endpoint="/orchestrator/orchestrator/health-checks/run",
        meta={"scope": req.scope, "contextId": req.contextId},
    )
    try:
        result = svc.trigger_health_check(
            scope=req.scope,
            context_id=req.contextId,
            ticket_id=req.ticketId,
        )
        return HealthCheckResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "health_check_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# POST /router/classify
# ─────────────────────────────────────────────

@router.post("/router/classify", response_model=ClassificationResponse,
             summary="Classify a ticket into a routing action (e.g. PERF_INCIDENT, NEW_DATA)")
def classify_ticket_routing(req: ClassificationRequest):
    """
    Connects Ops Monitoring signals to the correct governance process.
    Uses a signal-based classifier (or manual override) to assign a routing label.

    **API spec inputs:** ticketId, signals? (alerts/metrics), manualOverride?,
    classifierVersion?
    **Returns:** routingLabel, confidence, rationale, recommendedAction, nextStepRef
    """
    make_log(
        area="Orchestrator",
        component="Orchestration",
        endpoint="/orchestrator/router/classify",
        meta={"ticketId": req.ticketId, "classifierVersion": req.classifierVersion},
    )
    try:
        result = svc.classify_ticket(
            ticket_id=req.ticketId,
            signals=req.signals,
            manual_override=req.manualOverride,
            classifier_version=req.classifierVersion,
        )
        return ClassificationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "classification_failed", "message": str(e)}
        })
