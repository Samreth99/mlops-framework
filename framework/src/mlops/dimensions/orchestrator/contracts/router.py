"""
Orchestrator – Contracts endpoints.

  POST  /contracts                         – Register a new coordination contract
  GET   /contracts                         – List contracts (filter by domain)
  GET   /contracts/{contractId}            – Read contract snapshot + version pointers
  POST  /contracts/{contractId}            – Update contract metadata
  POST  /contracts/{contractId}/versions   – Create a versioned contract definition
  POST  /contracts/{contractId}/validate   – Validate inputs/outputs/gates vs contract
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from ....core.logs import make_log
from ..schemas import (
    ContractCreate,
    ContractResponse,
    ContractUpdate,
    ContractSnapshotResponse,
    ContractVersionCreate,
    ContractVersionResponse,
    ContractValidationRequest,
    ContractValidationResponse,
)
from .. import orchestrator_service as svc

router = APIRouter()


# ─────────────────────────────────────────────
# POST /contracts  — register
# GET  /contracts  — list
# ─────────────────────────────────────────────

@router.post("/contracts", response_model=ContractResponse,
             summary="Register a new coordination contract (process spec)")
def register_contract(req: ContractCreate):
    """
    **API spec inputs (POST):** name, description, owner, domain, defaultGates?[]
    **API spec inputs (GET):** filters?
    **Returns:** contractId, currentVersionPointer
    """
    make_log(
        area="Orchestrator",
        component="Contracts",
        endpoint="/orchestrator/contracts",
        meta={"name": req.name, "domain": req.domain, "owner": req.owner},
    )
    try:
        result = svc.create_contract(
            name=req.name,
            description=req.description,
            owner=req.owner,
            domain=req.domain,
            default_gates=req.defaultGates,
        )
        return ContractResponse(
            contractId=result["contractId"],
            currentVersionPointer=result["currentVersionPointer"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "contract_creation_failed", "message": str(e)}
        })


@router.get("/contracts", response_model=List[ContractResponse],
            summary="List coordination contracts with optional filters")
def list_contracts(
    domain: Optional[str] = Query(None, description="Filter by MLOps sub-domain"),
):
    make_log(
        area="Orchestrator",
        component="Contracts",
        endpoint="/orchestrator/contracts",
        meta={"domain": domain},
    )
    try:
        return [
            ContractResponse(
                contractId=c["contractId"],
                currentVersionPointer=c["currentVersionPointer"],
            )
            for c in svc.list_contracts(domain=domain)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "list_contracts_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# GET  /contracts/{contractId}  — read snapshot
# POST /contracts/{contractId}  — update metadata
# ─────────────────────────────────────────────

@router.get("/contracts/{contractId}", response_model=ContractSnapshotResponse,
            summary="Read contract metadata snapshot and version pointers")
def read_contract(contractId: str):
    """
    **Returns:** contract snapshot + version_pointers
    """
    make_log(
        area="Orchestrator",
        component="Contracts",
        endpoint=f"/orchestrator/contracts/{contractId}",
        meta={"contractId": contractId},
    )
    contract = svc.get_contract(contractId)
    if contract is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "contract_not_found", "message": f"Contract '{contractId}' not found"}
        })
    return ContractSnapshotResponse(
        snapshot=contract,
        version_pointers={"current": contract["currentVersionPointer"]},
    )


@router.post("/contracts/{contractId}", response_model=ContractSnapshotResponse,
             summary="Update contract metadata (description, owners, status)")
def update_contract(contractId: str, update: ContractUpdate):
    """
    **API spec inputs:** description?, owners?[], status?
    **Returns:** contract snapshot + version pointers
    """
    make_log(
        area="Orchestrator",
        component="Contracts",
        endpoint=f"/orchestrator/contracts/{contractId}",
        meta={"contractId": contractId},
    )
    if svc.get_contract(contractId) is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "contract_not_found", "message": f"Contract '{contractId}' not found"}
        })
    try:
        update_data = update.model_dump(exclude_unset=True)
        result = svc.update_contract(contractId, update_data)
        return ContractSnapshotResponse(
            snapshot=result,
            version_pointers={"current": result["currentVersionPointer"]},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "contract_update_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# POST /contracts/{contractId}/versions
# ─────────────────────────────────────────────

@router.post("/contracts/{contractId}/versions", response_model=ContractVersionResponse,
             summary="Create a versioned contract definition")
def create_contract_version(contractId: str, req: ContractVersionCreate):
    """
    Links a high-level contract to specific BPMN logic, IO rules, and gate policies.

    **API spec inputs:** version, definitionRef (e.g. BPMN), ioSchema, gatePolicies?,
    routingRules?
    **Returns:** contractVersionId, immutable definitionDigest
    """
    make_log(
        area="Orchestrator",
        component="Contracts",
        endpoint=f"/orchestrator/contracts/{contractId}/versions",
        meta={"contractId": contractId, "version": req.version},
    )
    if svc.get_contract(contractId) is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "contract_not_found", "message": f"Contract '{contractId}' not found"}
        })
    try:
        result = svc.create_contract_version(
            contract_id=contractId,
            version=req.version,
            definition_ref=req.definitionRef,
            io_schema=req.ioSchema,
            gate_policies=req.gatePolicies,
            routing_rules=req.routingRules,
        )
        return ContractVersionResponse(
            contractVersionId=result["contractVersionId"],
            definitionDigest=result["definitionDigest"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "contract_version_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# POST /contracts/{contractId}/validate
# ─────────────────────────────────────────────

@router.post("/contracts/{contractId}/validate", response_model=ContractValidationResponse,
             summary="Validate that inputs/outputs/gates satisfy the contract")
def validate_contract(contractId: str, req: ContractValidationRequest):
    """
    Ensures engineering rigor before moving to the next MLOps stage.

    **API spec inputs:** contractVersionId, providedInputs, expectedOutputs?,
    gateEvidence?[]
    **Returns:** valid=true/false, violations list, required_fixes
    """
    make_log(
        area="Orchestrator",
        component="Contracts",
        endpoint=f"/orchestrator/contracts/{contractId}/validate",
        meta={"contractId": contractId, "contractVersionId": req.contractVersionId},
    )
    if svc.get_contract(contractId) is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "contract_not_found", "message": f"Contract '{contractId}' not found"}
        })
    try:
        result = svc.validate_contract(
            contract_id=contractId,
            contract_version_id=req.contractVersionId,
            provided_inputs=req.providedInputs,
            gate_evidence=req.gateEvidence,
        )
        if result.get("not_found"):
            raise HTTPException(status_code=404, detail={
                "error": {
                    "code": "contract_version_not_found",
                    "message": f"Version '{req.contractVersionId}' not found",
                }
            })
        return ContractValidationResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "validation_failed", "message": str(e)}
        })
