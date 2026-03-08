"""
Ticket Generation API endpoints.
  POST   /tickets  – Generate MPO tickets from planning decisions
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ....core.logs import make_log
from ..schemas import TicketGenerationRequest, TicketGenerationResponse
from .. import plan_service as svc

router = APIRouter()


@router.post("/tickets", response_model=TicketGenerationResponse,
             summary="Generate MPO tickets from planning decisions")
def generate_tickets(req: TicketGenerationRequest):
    """
    Convert a list of planning work items into orchestration tickets.

    Each ticket is auto-routed to a dimension (data/model/ops/soft/orchestrator)
    based on keyword matching in the work item description.

    **Inputs:** projectId, workItems[] (type, description, priority, owner), targetContract?
    **Returns:** created ticketIds[] + routing hints per ticket
    """
    log = make_log(
        area="Plan", component="Ticket Generation",
        endpoint="/plan/tickets",
        meta={"projectId": req.projectId, "workItems": len(req.workItems)},
    )
    try:
        work_items_dicts = [w.model_dump() for w in req.workItems]
        result = svc.generate_tickets(
            project_id=req.projectId,
            work_items=work_items_dicts,
            target_contract=req.targetContract,
        )
        return TicketGenerationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "ticket_generation_failed", "message": str(e)}
        })
