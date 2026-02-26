"""
Orchestrator – Tickets endpoints.

  POST  /tickets                         – Create a governance ticket
  GET   /tickets                         – List tickets (filter by type, priority)
  POST  /tickets/{ticketId}              – Update core fields
  POST  /tickets/{ticketId}/assign       – Set / transfer assignee and ownership
  POST  /tickets/{ticketId}/state        – Transition ticket status
  POST  /tickets/{ticketId}/evidence     – Attach validation report / approval / log
  POST  /tickets/{ticketId}/artifacts    – Link dataset / model / service versions
  GET   /tickets/{ticketId}/history      – Retrieve state transitions and actions log
  POST  /tickets/{ticketId}/routing      – Persist routing decision + rationale
"""
from __future__ import annotations

import json
import hashlib
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from ....core.logs import make_log
from ..schemas import (
    TicketCreate,
    TicketResponse,
    TicketUpdate,
    AssignmentRequest,
    TicketAssignmentResponse,
    StateTransitionRequest,
    TicketStateResponse,
    EvidenceCreate,
    EvidenceResponse,
    ArtifactLinkRequest,
    ArtifactLinkResponse,
    TicketHistoryResponse,
    RoutingDecisionRequest,
    RoutingDecisionResponse,
)
from .. import orchestrator_service as svc

router = APIRouter()


# ─────────────────────────────────────────────
# POST /tickets  — create
# GET  /tickets  — list
# ─────────────────────────────────────────────

@router.post("/tickets", response_model=TicketResponse,
             summary="Create a governance ticket (issue, alert, or request)")
def create_ticket(req: TicketCreate):
    """
    Create a new governance ticket.

    **API spec inputs:** type, category, severity, priority, title, description,
    requester, assignee?, relatedArtifacts?[], contractRef?, tags?[]
    **Returns:** ticketId, state, timestamps, links, sla
    """
    make_log(
        area="Orchestrator",
        component="Tickets",
        endpoint="/orchestrator/tickets",
        meta={"type": req.type, "title": req.title, "requester": req.requester},
    )
    try:
        result = svc.create_ticket(
            type=req.type,
            category=req.category,
            severity=req.severity,
            priority=req.priority,
            title=req.title,
            description=req.description,
            requester=req.requester,
            assignee=req.assignee,
            related_artifacts=req.relatedArtifacts,
            contract_ref=req.contractRef,
            tags=req.tags,
        )
        return TicketResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "ticket_creation_failed", "message": str(e)}
        })


@router.get("/tickets", response_model=List[TicketResponse],
            summary="List governance tickets with optional filters")
def list_tickets(
    type: Optional[str] = Query(None, description="Filter by ticket type"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
):
    make_log(
        area="Orchestrator",
        component="Tickets",
        endpoint="/orchestrator/tickets",
        meta={"type": type, "priority": priority},
    )
    try:
        return [TicketResponse(**t) for t in svc.list_tickets(type=type, priority=priority)]
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "list_tickets_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# POST /tickets/{ticketId}  — update core fields
# ─────────────────────────────────────────────

@router.post("/tickets/{ticketId}",
             summary="Read / update a ticket's core fields")
def update_ticket(ticketId: str, update: TicketUpdate):
    """
    **API spec inputs (POST):** title?, description?, priority?, severity?,
    labels?[], dueDate?, relatedArtifacts?[]
    **Returns:** updated ticket snapshot + version/etag
    """
    make_log(
        area="Orchestrator",
        component="Tickets",
        endpoint=f"/orchestrator/tickets/{ticketId}",
        meta={"ticketId": ticketId},
    )
    ticket = svc.get_ticket(ticketId)
    if ticket is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "ticket_not_found", "message": f"Ticket '{ticketId}' not found"}
        })
    try:
        update_data = update.model_dump(exclude_unset=True)
        result = svc.update_ticket(ticketId, update_data)
        ticket_json = json.dumps(result, sort_keys=True, default=str)
        etag = hashlib.md5(ticket_json.encode()).hexdigest()
        return {
            "snapshot": result,
            "version": "1.0.1",
            "etag": f'W/"{etag}"',
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "ticket_update_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# POST /tickets/{ticketId}/assign
# ─────────────────────────────────────────────

@router.post("/tickets/{ticketId}/assign", response_model=TicketAssignmentResponse,
             summary="Set or transfer assignee and ownership")
def assign_ticket(ticketId: str, assignment: AssignmentRequest):
    """
    **API spec inputs:** assignee, team?, reason?
    **Returns:** ticketId, assignee, state, auditEntryId
    """
    make_log(
        area="Orchestrator",
        component="Tickets",
        endpoint=f"/orchestrator/tickets/{ticketId}/assign",
        meta={"ticketId": ticketId, "assignee": assignment.assignee},
    )
    result = svc.assign_ticket(
        ticket_id=ticketId,
        assignee=assignment.assignee,
        team=assignment.team,
        reason=assignment.reason,
    )
    if result is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "ticket_not_found", "message": f"Ticket '{ticketId}' not found"}
        })
    try:
        import uuid
        return TicketAssignmentResponse(
            ticketId=ticketId,
            assignee=result["assignee"],
            state=result["state"],
            auditEntryId=str(uuid.uuid4()),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "assign_ticket_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# POST /tickets/{ticketId}/state
# ─────────────────────────────────────────────

@router.post("/tickets/{ticketId}/state", response_model=TicketStateResponse,
             summary="Transition ticket status (open → in-progress → resolved, etc.)")
def transition_ticket_state(ticketId: str, transition: StateTransitionRequest):
    """
    **API spec inputs:** from, to, reason, gateResult?, approvalRef?
    **Returns:** ticketId, new_state, transition_record, next_required_actions
    """
    make_log(
        area="Orchestrator",
        component="Tickets",
        endpoint=f"/orchestrator/tickets/{ticketId}/state",
        meta={"ticketId": ticketId, "to": transition.to_state},
    )
    result = svc.transition_ticket_state(
        ticket_id=ticketId,
        from_state=transition.from_state,
        to_state=transition.to_state,
        reason=transition.reason,
        gate_result=transition.gateResult,
        approval_ref=transition.approvalRef,
    )
    if result is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "ticket_not_found", "message": f"Ticket '{ticketId}' not found"}
        })
    try:
        next_actions: List[str] = []
        if transition.to_state == "in-progress":
            next_actions = ["D3. Preprocess Data", "D4. Validate Data"]
        elif transition.to_state == "resolved":
            next_actions = ["O5. Deploy to Production"]

        from datetime import datetime, timezone
        return TicketStateResponse(
            ticketId=ticketId,
            new_state=transition.to_state,
            transition_record={
                "transitioned_at": datetime.now(timezone.utc).isoformat(),
                "from": transition.from_state,
                "reason": transition.reason,
                "gate_passed": str(transition.gateResult) if transition.gateResult is not None else "N/A",
            },
            next_required_actions=next_actions,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "state_transition_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# POST /tickets/{ticketId}/evidence
# ─────────────────────────────────────────────

@router.post("/tickets/{ticketId}/evidence", response_model=EvidenceResponse,
             summary="Attach validation reports, approvals, or audit evidence")
def attach_evidence(ticketId: str, evidence: EvidenceCreate):
    """
    **API spec inputs:** evidenceType (report|approval|log), artifactRefs?[],
    uri|blobRef, summary, checksum?
    **Returns:** evidenceId, stored_reference, linked_ticket
    """
    make_log(
        area="Orchestrator",
        component="Tickets",
        endpoint=f"/orchestrator/tickets/{ticketId}/evidence",
        meta={"ticketId": ticketId, "evidenceType": evidence.evidenceType},
    )
    if svc.get_ticket(ticketId) is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "ticket_not_found", "message": f"Ticket '{ticketId}' not found"}
        })
    try:
        result = svc.attach_evidence(
            ticket_id=ticketId,
            evidence_type=evidence.evidenceType,
            uri_or_blob_ref=evidence.uri_or_blobRef,
            summary=evidence.summary,
            artifact_refs=evidence.artifactRefs,
            checksum=evidence.checksum,
        )
        return EvidenceResponse(
            evidenceId=result["evidenceId"],
            stored_reference=result["uri_or_blobRef"],
            linked_ticket=ticketId,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "evidence_attach_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# POST /tickets/{ticketId}/artifacts
# ─────────────────────────────────────────────

@router.post("/tickets/{ticketId}/artifacts", response_model=ArtifactLinkResponse,
             summary="Link provided/required artifacts (dataset/feature/model/service/manifest)")
def link_artifacts(ticketId: str, artifact: ArtifactLinkRequest):
    """
    **API spec inputs:** artifactType, artifactId, versionId, role (input|output), provenance?
    **Returns:** updated artifact links, linkageIds
    """
    make_log(
        area="Orchestrator",
        component="Tickets",
        endpoint=f"/orchestrator/tickets/{ticketId}/artifacts",
        meta={"ticketId": ticketId, "artifactId": artifact.artifactId},
    )
    if svc.get_ticket(ticketId) is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "ticket_not_found", "message": f"Ticket '{ticketId}' not found"}
        })
    try:
        entry = svc.link_artifact(
            ticket_id=ticketId,
            artifact_type=artifact.artifactType,
            artifact_id=artifact.artifactId,
            version_id=artifact.versionId,
            role=artifact.role,
            provenance=artifact.provenance,
        )
        ticket = svc.get_ticket(ticketId)
        return ArtifactLinkResponse(
            updated_artifact_links=ticket["links"].get("artifacts", []),
            linkageIds=[entry["linkageId"]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "artifact_link_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# GET /tickets/{ticketId}/history
# ─────────────────────────────────────────────

@router.get("/tickets/{ticketId}/history", response_model=TicketHistoryResponse,
            summary="Retrieve state transitions and actions log")
def get_ticket_history(
    ticketId: str,
    since: Optional[str] = Query(None, description="ISO-8601 datetime filter (inclusive)"),
    limit: int = Query(default=50, ge=1, description="Maximum entries to return"),
    includeSystem: bool = Query(default=True, description="Include system-generated entries"),
):
    """
    **API spec inputs:** since?, limit?, includeSystem?
    **Returns:** ordered list of transitions/actions with actors and timestamps
    """
    make_log(
        area="Orchestrator",
        component="Tickets",
        endpoint=f"/orchestrator/tickets/{ticketId}/history",
        meta={"ticketId": ticketId, "limit": limit},
    )
    history = svc.get_ticket_history(ticketId)
    if history is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "ticket_not_found", "message": f"Ticket '{ticketId}' not found"}
        })
    try:
        filtered = history
        if since:
            filtered = [h for h in filtered if h["timestamp"] >= since]
        if not includeSystem:
            filtered = [h for h in filtered if h["actor"] != "System"]

        ordered = sorted(filtered, key=lambda x: x["timestamp"], reverse=True)
        return TicketHistoryResponse(ticketId=ticketId, history=ordered[:limit])
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "history_fetch_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# POST /tickets/{ticketId}/routing
# ─────────────────────────────────────────────

@router.post("/tickets/{ticketId}/routing", response_model=RoutingDecisionResponse,
             summary="Persist routing decision + rationale onto the ticket")
def persist_ticket_routing(ticketId: str, decision: RoutingDecisionRequest):
    """
    **API spec inputs:** routingLabel, actionType, rationale, decidedBy (human|system),
    evidenceRefs?[]
    **Returns:** ticketId, routingRecordId, updatedTicketSnapshot
    """
    make_log(
        area="Orchestrator",
        component="Tickets",
        endpoint=f"/orchestrator/tickets/{ticketId}/routing",
        meta={"ticketId": ticketId, "routingLabel": decision.routingLabel},
    )
    if svc.get_ticket(ticketId) is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "ticket_not_found", "message": f"Ticket '{ticketId}' not found"}
        })
    try:
        result = svc.persist_routing_decision(
            ticket_id=ticketId,
            routing_label=decision.routingLabel,
            action_type=decision.actionType,
            rationale=decision.rationale,
            decided_by=decision.decidedBy,
            evidence_refs=decision.evidenceRefs,
        )
        return RoutingDecisionResponse(
            ticketId=ticketId,
            routingRecordId=result["routing_record_id"],
            updatedTicketSnapshot=result["ticket"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "routing_persist_failed", "message": str(e)}
        })
