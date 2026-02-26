"""
Orchestrator – Events endpoints.

  POST  /events/publish    – Publish typed events for orchestration routing
  POST  /events/subscribe  – Consume/stream events for handlers and routers
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ....core.logs import make_log
from ..schemas import (
    EventPublishRequest,
    EventPublishResponse,
    EventSubscribeRequest,
    EventBatchResponse,
)
from .. import orchestrator_service as svc

router = APIRouter()


# ─────────────────────────────────────────────
# POST /events/publish
# ─────────────────────────────────────────────

@router.post("/events/publish", response_model=EventPublishResponse,
             summary="Publish a typed event for orchestration routing")
def publish_event(req: EventPublishRequest):
    """
    Operationalises strategic decisions by emitting typed events that the MPO
    routes to the correct downstream worker or workflow.

    **API spec inputs:** eventType, sourceComponent, ticketId?, runId?,
    artifactRefs?[], payload, timestamp
    **Returns:** eventId, accepted=true, routing_metadata
    """
    make_log(
        area="Orchestrator",
        component="Events",
        endpoint="/orchestrator/events/publish",
        meta={
            "eventType": req.eventType,
            "sourceComponent": req.sourceComponent,
            "runId": req.runId,
        },
    )
    try:
        result = svc.publish_event(
            event_type=req.eventType,
            source_component=req.sourceComponent,
            payload=req.payload,
            timestamp=req.timestamp,
            ticket_id=req.ticketId,
            run_id=req.runId,
            artifact_refs=req.artifactRefs,
        )
        return EventPublishResponse(
            eventId=result["eventId"],
            accepted=True,
            routing_metadata=result["routing_metadata"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "event_publish_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# POST /events/subscribe
# ─────────────────────────────────────────────

@router.post("/events/subscribe", response_model=EventBatchResponse,
             summary="Consume / stream events for handlers and routers")
def consume_events(req: EventSubscribeRequest):
    """
    Allows MLOps workers to pull their assigned tasks from the MPO event log.
    Supports cursor-based pagination for replay and reproducibility.

    **API spec inputs:** topics[]/eventTypes[], consumerId, offset|cursor?, ackMode?
    **Returns:** stream/batch of events + next_cursor
    """
    make_log(
        area="Orchestrator",
        component="Events",
        endpoint="/orchestrator/events/subscribe",
        meta={"consumerId": req.consumerId, "eventTypes": req.eventTypes},
    )
    try:
        result = svc.consume_events(
            event_types=req.eventTypes,
            consumer_id=req.consumerId,
            offset=req.offset,
            ack_mode=req.ackMode,
        )
        return EventBatchResponse(
            events=result["events"],
            next_cursor=result["next_cursor"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "event_consume_failed", "message": str(e)}
        })
