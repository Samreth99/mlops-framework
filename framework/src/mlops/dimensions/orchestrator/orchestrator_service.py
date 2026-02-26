"""
Orchestrator dimension service layer.

Provides an in-memory store for:
  - Tickets (governance, state, evidence, artifacts, history, routing)
  - Events  (publish / consume)
  - Contracts (CRUD, versioned definitions, compliance validation)
  - Runs   (orchestrated executions, traces, artifacts, linked tickets)
  - Orchestration context (context, start, health-check, classify)

All IDs use uuid4 hex prefixed for readability.
Timestamps are ISO-8601 UTC strings.
The store resets on process restart — swap _store with a DB backend for production.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
# In-memory store
# ─────────────────────────────────────────────
_store: Dict[str, Dict[str, Any]] = {
    "tickets": {},
    "events": [],        # append-only log
    "contracts": {},
    "runs": {},
    "context": {},       # single active context (keyed by contextId)
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


# ─────────────────────────────────────────────
# Tickets
# ─────────────────────────────────────────────

def create_ticket(
    type: str,
    category: str,
    severity: str,
    priority: str,
    title: str,
    description: str,
    requester: str,
    assignee: Optional[str] = None,
    related_artifacts: Optional[List[str]] = None,
    contract_ref: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    ticket_id = _new_id("tkt-")
    now = _now()
    record = {
        "ticketId": ticket_id,
        "type": type,
        "category": category,
        "severity": severity,
        "priority": priority,
        "title": title,
        "description": description,
        "requester": requester,
        "assignee": assignee,
        "relatedArtifacts": related_artifacts or [],
        "contractRef": contract_ref,
        "tags": tags or [],
        "state": "open",
        "timestamps": {"created_at": now, "updated_at": now},
        "links": {"artifacts": [], "evidence": []},
        "sla": "48h",
        "history": [],
        "routing_history": [],
    }
    _store["tickets"][ticket_id] = record
    _append_history(record, action="Ticket Created", actor=requester)
    return record


def get_ticket(ticket_id: str) -> Optional[Dict[str, Any]]:
    return _store["tickets"].get(ticket_id)


def list_tickets(
    type: Optional[str] = None,
    priority: Optional[str] = None,
) -> List[Dict[str, Any]]:
    tickets = list(_store["tickets"].values())
    if type:
        tickets = [t for t in tickets if t["type"] == type]
    if priority:
        tickets = [t for t in tickets if t["priority"] == priority]
    return tickets


def update_ticket(
    ticket_id: str,
    update_data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    record = _store["tickets"].get(ticket_id)
    if record is None:
        return None
    for key, value in update_data.items():
        record[key] = value
    record["timestamps"]["updated_at"] = _now()
    _append_history(record, action="Ticket Updated", actor="System", details=update_data)
    return record


def assign_ticket(
    ticket_id: str,
    assignee: str,
    team: Optional[str] = None,
    reason: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    record = _store["tickets"].get(ticket_id)
    if record is None:
        return None
    record["assignee"] = assignee
    record["state"] = "assigned"
    record["timestamps"]["updated_at"] = _now()
    details: Dict[str, str] = {"assignee": assignee}
    if team:
        details["team"] = team
    if reason:
        details["reason"] = reason
    _append_history(record, action="Ticket Assigned", actor=assignee, details=details)
    return record


def transition_ticket_state(
    ticket_id: str,
    from_state: str,
    to_state: str,
    reason: str,
    gate_result: Optional[bool] = None,
    approval_ref: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    record = _store["tickets"].get(ticket_id)
    if record is None:
        return None
    record["state"] = to_state
    record["timestamps"]["updated_at"] = _now()
    details: Dict[str, str] = {
        "from": from_state,
        "to": to_state,
        "reason": reason,
    }
    if gate_result is not None:
        details["gate_passed"] = str(gate_result)
    if approval_ref:
        details["approvalRef"] = approval_ref
    _append_history(record, action="State Transition", actor="System", details=details)
    return record


def attach_evidence(
    ticket_id: str,
    evidence_type: str,
    uri_or_blob_ref: str,
    summary: str,
    artifact_refs: Optional[List[str]] = None,
    checksum: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    record = _store["tickets"].get(ticket_id)
    if record is None:
        return None
    evidence_id = _new_id("ev-")
    evidence_record = {
        "evidenceId": evidence_id,
        "evidenceType": evidence_type,
        "artifactRefs": artifact_refs or [],
        "uri_or_blobRef": uri_or_blob_ref,
        "summary": summary,
        "checksum": checksum,
        "timestamp": _now(),
    }
    record["links"].setdefault("evidence", []).append(evidence_record)
    _append_history(record, action="Evidence Attached", actor="System",
                    details={"evidenceId": evidence_id, "type": evidence_type})
    return evidence_record


def link_artifact(
    ticket_id: str,
    artifact_type: str,
    artifact_id: str,
    version_id: str,
    role: str,
    provenance: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    record = _store["tickets"].get(ticket_id)
    if record is None:
        return None
    linkage_id = _new_id("lnk-")
    entry = {
        "linkageId": linkage_id,
        "artifactType": artifact_type,
        "artifactId": artifact_id,
        "versionId": version_id,
        "role": role,
        "provenance": provenance,
        "linked_at": _now(),
    }
    record["links"].setdefault("artifacts", []).append(entry)
    _append_history(record, action="Artifact Linked", actor="System",
                    details={"artifactId": artifact_id, "role": role})
    return entry


def get_ticket_history(ticket_id: str) -> Optional[List[Dict[str, Any]]]:
    record = _store["tickets"].get(ticket_id)
    if record is None:
        return None
    return record.get("history", [])


def persist_routing_decision(
    ticket_id: str,
    routing_label: str,
    action_type: str,
    rationale: str,
    decided_by: str,
    evidence_refs: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    record = _store["tickets"].get(ticket_id)
    if record is None:
        return None
    routing_record_id = _new_id("rt-")
    entry = {
        "routingRecordId": routing_record_id,
        "routingLabel": routing_label,
        "actionType": action_type,
        "rationale": rationale,
        "decidedBy": decided_by,
        "evidenceRefs": evidence_refs or [],
        "timestamp": _now(),
    }
    record.setdefault("routing_history", []).append(entry)
    record["state"] = f"routed:{routing_label}"
    record["timestamps"]["updated_at"] = _now()
    _append_history(record, action="Routing Decision Persisted", actor=decided_by,
                    details={"routingLabel": routing_label, "actionType": action_type})
    return {"routing_record_id": routing_record_id, "ticket": record}


def _append_history(
    ticket: Dict[str, Any],
    action: str,
    actor: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Internal helper — appends a structured entry to the ticket's history log."""
    ticket.setdefault("history", []).append({
        "action": action,
        "actor": actor,
        "timestamp": _now(),
        "details": {k: str(v) for k, v in (details or {}).items()},
    })


# ─────────────────────────────────────────────
# Events
# ─────────────────────────────────────────────

def publish_event(
    event_type: str,
    source_component: str,
    payload: Any,
    timestamp: str,
    ticket_id: Optional[str] = None,
    run_id: Optional[str] = None,
    artifact_refs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    event_id = _new_id("evt-")

    # Simple routing logic based on event type keywords
    if "DRIFT" in event_type.upper():
        route_target = "Trigger_Retraining_Workflow"
    elif "COMPLETED" in event_type.upper():
        route_target = "Quality_Gate_Validator"
    elif "FAILED" in event_type.upper():
        route_target = "Incident_Handler"
    else:
        route_target = "Default_Handler"

    record = {
        "eventId": event_id,
        "eventType": event_type,
        "sourceComponent": source_component,
        "ticketId": ticket_id,
        "runId": run_id,
        "artifactRefs": artifact_refs or [],
        "payload": payload,
        "timestamp": timestamp,
        "stored_at": _now(),
        "routing_metadata": {
            "target_queue": route_target,
            "processed_by": "MPO_Event_Router",
            "correlation_id": run_id or ticket_id or "global",
        },
    }
    _store["events"].append(record)
    return record


def consume_events(
    event_types: List[str],
    consumer_id: str,
    offset: Optional[str] = None,
    ack_mode: str = "auto",
) -> Dict[str, Any]:
    """Return a batch of events matching the requested types."""
    matching = [e for e in _store["events"] if e["eventType"] in event_types]
    batch = matching[-10:]  # last 10 matching events
    return {
        "events": batch,
        "next_cursor": _now(),
    }


# ─────────────────────────────────────────────
# Contracts
# ─────────────────────────────────────────────

def create_contract(
    name: str,
    description: str,
    owner: str,
    domain: str,
    default_gates: Optional[List[str]] = None,
) -> Dict[str, Any]:
    contract_id = _new_id("ctr-")
    record = {
        "contractId": contract_id,
        "name": name,
        "description": description,
        "owner": owner,
        "domain": domain,
        "defaultGates": default_gates or [],
        "currentVersionPointer": "v1.0.0",
        "status": "draft",
        "versions": [],
        "created_at": _now(),
        "updated_at": None,
    }
    _store["contracts"][contract_id] = record
    return record


def get_contract(contract_id: str) -> Optional[Dict[str, Any]]:
    return _store["contracts"].get(contract_id)


def list_contracts(domain: Optional[str] = None) -> List[Dict[str, Any]]:
    contracts = list(_store["contracts"].values())
    if domain:
        contracts = [c for c in contracts if c["domain"] == domain]
    return contracts


def update_contract(
    contract_id: str,
    update_data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    record = _store["contracts"].get(contract_id)
    if record is None:
        return None
    for key, value in update_data.items():
        record[key] = value
    record["updated_at"] = _now()
    return record


def create_contract_version(
    contract_id: str,
    version: str,
    definition_ref: str,
    io_schema: Dict[str, Any],
    gate_policies: Optional[List[str]] = None,
    routing_rules: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    contract = _store["contracts"].get(contract_id)
    if contract is None:
        return None

    # Immutable digest — hash of definitionRef + ioSchema
    raw = f"{definition_ref}-{str(io_schema)}"
    digest = hashlib.sha256(raw.encode()).hexdigest()

    version_id = f"{contract_id}-{version}"
    new_version = {
        "contractVersionId": version_id,
        "version": version,
        "definitionRef": definition_ref,
        "ioSchema": io_schema,
        "gatePolicies": gate_policies or [],
        "routingRules": routing_rules or {},
        "definitionDigest": digest,
        "created_at": _now(),
    }
    contract["versions"].append(new_version)
    contract["currentVersionPointer"] = version
    contract["updated_at"] = _now()
    return new_version


def validate_contract(
    contract_id: str,
    contract_version_id: str,
    provided_inputs: Dict[str, Any],
    gate_evidence: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    contract = _store["contracts"].get(contract_id)
    if contract is None:
        return None

    version = next(
        (v for v in contract.get("versions", [])
         if v["contractVersionId"] == contract_version_id),
        None,
    )
    if version is None:
        return {"not_found": True}

    violations: List[str] = []
    required_fixes: List[str] = []

    # Check required inputs against ioSchema
    for key in version["ioSchema"].get("required_inputs", []):
        if key not in provided_inputs:
            violations.append(f"Missing required input: {key}")
            required_fixes.append(f"Provide value for '{key}'")

    # Check gate policy evidence
    for policy in version.get("gatePolicies", []):
        if policy not in (gate_evidence or []):
            violations.append(f"Gate policy violation: '{policy}' evidence missing")
            required_fixes.append(f"Attach evidence for '{policy}'")

    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "required_fixes": required_fixes,
    }


# ─────────────────────────────────────────────
# Runs
# ─────────────────────────────────────────────

def create_run(
    ticket_id: str,
    contract_version_id: str,
    requested_by: str,
    input_artifacts: List[str],
    parameters: Optional[Dict[str, Any]] = None,
    target_env: str = "staging",
) -> Dict[str, Any]:
    run_id = _new_id("run-")
    record = {
        "runId": run_id,
        "ticketId": ticket_id,
        "contractVersionId": contract_version_id,
        "requestedBy": requested_by,
        "inputArtifacts": input_artifacts,
        "parameters": parameters or {},
        "targetEnv": target_env,
        "status": "pending",
        "correlation_ids": {
            "ticketId": ticket_id,
            "contractVersionId": contract_version_id,
        },
        "created_at": _now(),
        "updated_at": None,
    }
    _store["runs"][run_id] = record
    return record


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    return _store["runs"].get(run_id)


def list_runs(
    ticket_id: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    runs = list(_store["runs"].values())
    if ticket_id:
        runs = [r for r in runs if r["correlation_ids"]["ticketId"] == ticket_id]
    if status:
        runs = [r for r in runs if r["status"] == status]
    return runs


# ─────────────────────────────────────────────
# Orchestration context
# ─────────────────────────────────────────────

def set_context(
    env: str,
    business_key: Optional[str] = None,
    system_id: Optional[str] = None,
    default_contract_refs: Optional[List[str]] = None,
    default_policies: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    context_id = _new_id("ctx-")
    stored_ref = f"ctx-ref-{context_id[:6]}"
    effective = {
        "env": env,
        "policies": default_policies or {},
        "contracts": default_contract_refs or [],
    }
    record = {
        "contextId": context_id,
        "businessKey": business_key,
        "systemId": system_id,
        "env": env,
        "defaultContractRefs": default_contract_refs or [],
        "defaultPolicies": default_policies or {},
        "tags": tags or [],
        "storedContextRef": stored_ref,
        "effectiveDefaults": effective,
        "created_at": _now(),
    }
    _store["context"][context_id] = record
    return record


def start_orchestration(
    start_targets: List[str],
    context_id: Optional[str] = None,
    ticket_id: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    dispatch_id = _new_id("dp-")
    run_ids = [_new_id(f"run-{t.lower()}-") for t in start_targets]
    event_ids = [_new_id("evt-start-") for _ in run_ids]
    return {
        "dispatchId": dispatch_id,
        "runIds": run_ids,
        "accepted": True,
        "emittedEventIds": event_ids,
    }


def trigger_health_check(
    scope: str,
    context_id: Optional[str] = None,
    ticket_id: Optional[str] = None,
) -> Dict[str, Any]:
    run_id = _new_id("hc-")
    event_id = _new_id(f"evt-hc-{scope}-")
    return {
        "healthCheckRunId": run_id,
        "status": "triggered",
        "emittedEventIds": [event_id],
    }


def classify_ticket(
    ticket_id: str,
    signals: Dict[str, Any],
    manual_override: Optional[str] = None,
    classifier_version: str = "v1-llm-classifier",
) -> Dict[str, Any]:
    label = "GENERAL_MAINTENANCE"
    rationale = "No high-severity signals detected."
    recommended_action = "Monitor"
    confidence: float = 0.55  # low confidence when no strong signal

    if manual_override:
        label = manual_override
        rationale = "Manual override applied by operator."
        recommended_action = "Follow manual routing label."
        confidence = 1.0  # human decision — always certain

    elif signals.get("accuracy_drop", 0) > 0.1:
        # numeric float → scale confidence from how far above the 10% threshold
        drop = float(signals["accuracy_drop"])
        label = "PERF_INCIDENT"
        rationale = f"Accuracy drop of {drop:.0%} exceeds 10% threshold."
        recommended_action = "Trigger Retraining Workflow"
        confidence = round(min(0.99, 0.75 + drop * 1.2), 2)

    elif signals.get("drift_score") is not None:
        # numeric float 0–1 provided directly → use it as confidence base
        score = float(signals["drift_score"])
        label = "DATA_QUALITY_ISSUE"
        severity = "high" if score >= 0.7 else "moderate"
        rationale = f"Feature drift score {score:.2f} detected ({severity})."
        recommended_action = "Trigger Data Re-ingestion" if score >= 0.7 else "Schedule Data Re-ingestion"
        confidence = round(min(0.99, 0.55 + score * 0.6), 2)

    elif signals.get("data_drift") in ("high", "medium"):
        # categorical fallback when no drift_score is provided
        level = signals["data_drift"]
        label = "DATA_QUALITY_ISSUE"
        rationale = f"Categorical drift signal '{level}' received — no numeric score provided."
        recommended_action = "Trigger Data Re-ingestion" if level == "high" else "Schedule Data Re-ingestion"
        # no numeric magnitude → mark confidence as low to flag incomplete signal
        confidence = 0.60

    elif signals.get("infra_error_count") is not None:
        # numeric count → scale confidence by error frequency
        count = int(signals["infra_error_count"])
        label = "INFRA_FAILURE"
        rationale = f"{count} infrastructure error(s) detected."
        recommended_action = "Escalate to Ops team"
        confidence = round(min(0.99, 0.65 + count * 0.05), 2)

    elif signals.get("infra_error"):
        # boolean fallback when no count provided
        label = "INFRA_FAILURE"
        rationale = "Infrastructure error flag set — no error count provided."
        recommended_action = "Escalate to Ops team"
        confidence = 0.60

    return {
        "routingLabel": label,
        "confidence": confidence,
        "rationale": rationale,
        "recommendedAction": recommended_action,
        "nextStepRef": f"/orchestrator/tickets/{ticket_id}/state",
    }
