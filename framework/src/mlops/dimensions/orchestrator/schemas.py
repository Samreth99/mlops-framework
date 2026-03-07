"""
Orchestrator dimension Pydantic schemas.

Covers:
  - Tickets  (governance, state, evidence, artifacts, history, routing)
  - Events   (publish, subscribe/consume)
  - Contracts (CRUD, versioning, compliance validation)
  - Runs      (create, status, trace, artifacts, linked-tickets)
  - Orchestration context (context, start, health-check, classify)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ─────────────────────────────────────────────
# Tickets
# ─────────────────────────────────────────────

class TicketCreate(BaseModel):
    type: str
    category: str
    severity: str
    priority: str
    title: str
    description: str
    requester: str
    assignee: Optional[str] = None
    relatedArtifacts: List[str] = []
    contractRef: Optional[str] = None
    tags: List[str] = []
    routingHint: Optional[str] = None   # pre-set by Plan dimension (e.g. NEW_DATA, NEW_REQUIREMENT)


class TicketResponse(BaseModel):
    ticketId: str
    type: str
    category: str
    severity: str
    priority: str
    title: str
    description: str
    requester: str
    assignee: Optional[str] = None
    relatedArtifacts: List[str] = []
    contractRef: Optional[str] = None
    tags: List[str] = []
    state: str
    timestamps: Dict[str, str]
    links: Dict[str, List]
    sla: Optional[str] = "48h"
    routingHint: Optional[str] = None


class TicketUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[str] = None
    severity: Optional[str] = None
    labels: List[str] = []
    dueDate: Optional[str] = None
    relatedArtifacts: List[str] = []


class AssignmentRequest(BaseModel):
    assignee: str
    team: Optional[str] = None
    reason: Optional[str] = None


class TicketAssignmentResponse(BaseModel):
    ticketId: str
    assignee: str
    state: str
    auditEntryId: str


class StateTransitionRequest(BaseModel):
    from_state: str
    to_state: str
    reason: str
    gateResult: Optional[bool] = None
    approvalRef: Optional[str] = None


class TicketStateResponse(BaseModel):
    ticketId: str
    new_state: str
    transition_record: Dict[str, str]
    next_required_actions: List[str]


class EvidenceCreate(BaseModel):
    evidenceType: str                  # report | approval | log
    artifactRefs: List[str] = []
    uri_or_blobRef: str
    summary: str
    checksum: Optional[str] = None


class EvidenceResponse(BaseModel):
    evidenceId: str
    stored_reference: str
    linked_ticket: str


class ArtifactLinkRequest(BaseModel):
    artifactType: str                  # dataset | feature | model | service | manifest
    artifactId: str
    versionId: str
    role: str                          # input | output
    provenance: Optional[str] = None


class ArtifactLinkResponse(BaseModel):
    updated_artifact_links: List[Dict[str, Any]]
    linkageIds: List[str]


class HistoryEntry(BaseModel):
    action: str
    actor: str
    timestamp: str
    details: Dict[str, str]


class TicketHistoryResponse(BaseModel):
    ticketId: str
    history: List[HistoryEntry]


class RoutingDecisionRequest(BaseModel):
    routingLabel: str                  # e.g. PERF_INCIDENT
    actionType: str                    # e.g. RETRAIN | ROLLBACK | INVESTIGATE
    rationale: str
    decidedBy: str                     # human | system
    evidenceRefs: List[str] = []


class RoutingDecisionResponse(BaseModel):
    ticketId: str
    routingRecordId: str
    updatedTicketSnapshot: Dict[str, Any]


# ─────────────────────────────────────────────
# Events
# ─────────────────────────────────────────────

class EventPublishRequest(BaseModel):
    eventType: str                     # e.g. TRAINING_COMPLETED | DATA_DRIFT_DETECTED
    sourceComponent: str               # e.g. ModelTraining | RuntimeMonitoring
    ticketId: Optional[str] = None
    runId: Optional[str] = None
    artifactRefs: List[str] = []
    payload: Any                       # Flexible JSON payload
    timestamp: str                     # ISO-8601 string provided by the caller


class EventPublishResponse(BaseModel):
    eventId: str
    accepted: bool = True
    routing_metadata: Dict[str, str]


class EventSubscribeRequest(BaseModel):
    eventTypes: List[str]              # Topics to listen for
    consumerId: str                    # Unique ID of the worker
    offset: Optional[str] = None      # Current cursor position in the stream
    ackMode: str = "auto"              # auto | manual


class EventBatchResponse(BaseModel):
    events: List[Any]
    next_cursor: str


# ─────────────────────────────────────────────
# Contracts
# ─────────────────────────────────────────────

class ContractCreate(BaseModel):
    name: str                          # e.g. Standard-Retraining-Loop
    description: str
    owner: str
    domain: str                        # MLOps sub-domain
    defaultGates: List[str] = []


class ContractResponse(BaseModel):
    contractId: str
    currentVersionPointer: str


class ContractUpdate(BaseModel):
    description: Optional[str] = None
    owners: Optional[List[str]] = None
    status: Optional[str] = None      # active | deprecated | draft


class ContractSnapshotResponse(BaseModel):
    snapshot: Dict[str, Any]
    version_pointers: Dict[str, str]  # e.g. {"latest": "v1.2", "stable": "v1.0"}


class ContractVersionCreate(BaseModel):
    version: str                       # e.g. v1.0.0 or v2-stable
    definitionRef: str                 # URI or name of the BPMN process file
    ioSchema: Dict[str, Any]           # Required inputs/outputs
    gatePolicies: Optional[List[str]] = []
    routingRules: Optional[Dict[str, Any]] = {}


class ContractVersionResponse(BaseModel):
    contractVersionId: str
    definitionDigest: str              # Immutable SHA-256 hash of the definition


class ContractValidationRequest(BaseModel):
    contractVersionId: str
    providedInputs: Dict[str, Any]
    expectedOutputs: Optional[Dict[str, Any]] = None
    gateEvidence: List[str] = []       # List of evidenceIds


class ContractValidationResponse(BaseModel):
    valid: bool
    violations: List[str]
    required_fixes: List[str]


# ─────────────────────────────────────────────
# Runs
# ─────────────────────────────────────────────

class RunCreate(BaseModel):
    ticketId: str
    contractVersionId: str
    requestedBy: str
    inputArtifacts: List[str]
    parameters: Optional[Dict[str, Any]] = {}
    targetEnv: Optional[str] = "staging"


class RunResponse(BaseModel):
    runId: str
    status: str
    correlation_ids: Dict[str, str]


class StatusEvent(BaseModel):
    step_name: str
    status: str
    timestamp: str
    details: Optional[str] = None


class RunStatusResponse(BaseModel):
    runId: str
    status_timeline: List[StatusEvent]
    current_step: str
    summary_metrics: Dict[str, Any]
    gate_results: Optional[List[Dict[str, Any]]] = None


class TraceItem(BaseModel):
    itemType: str                      # step | gate | decision | action
    name: str
    actor: str
    timestamp: str
    outcome: str
    payload: Optional[Dict[str, Any]] = None


class RunTraceResponse(BaseModel):
    runId: str
    format: str
    ordered_trace: List[TraceItem]


class RunArtifactEntry(BaseModel):
    artifactType: str
    artifactId: str
    versionId: str
    role: str                          # input | output
    producedAt: Optional[str] = None


class RunArtifactsResponse(BaseModel):
    runId: str
    artifacts: List[RunArtifactEntry]


class LinkedTicketRef(BaseModel):
    ticketId: str
    relationType: str                  # trigger | sub-ticket | escalation
    title: str
    state: str


class RunTicketsResponse(BaseModel):
    runId: str
    linked_tickets: List[LinkedTicketRef]


# ─────────────────────────────────────────────
# Orchestration (context, start, health-check, classify)
# ─────────────────────────────────────────────

class ContextSetRequest(BaseModel):
    businessKey: Optional[str] = None
    systemId: Optional[str] = None
    env: str                           # dev | staging | prod
    defaultContractRefs: List[str] = []
    defaultPolicies: Dict[str, Any] = {}
    tags: List[str] = []


class ContextResponse(BaseModel):
    contextId: str
    storedContextRef: str
    effectiveDefaults: Dict[str, Any]


class StartOrchestrationRequest(BaseModel):
    contextId: Optional[str] = None
    startTargets: List[str]            # e.g. ["DATA", "MODEL", "SOFT", "OPS"]
    ticketId: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = {}


class StartOrchestrationResponse(BaseModel):
    dispatchId: str
    runIds: List[str]
    accepted: bool = True
    emittedEventIds: List[str]


class HealthCheckRequest(BaseModel):
    contextId: Optional[str] = None
    scope: str = "services"            # services | pipelines | infra
    ticketId: Optional[str] = None


class HealthCheckResponse(BaseModel):
    healthCheckRunId: str
    status: str
    emittedEventIds: List[str]


class ClassificationRequest(BaseModel):
    ticketId: str
    signals: Dict[str, Any] = {}       # e.g. {"accuracy_drop": 0.15, "data_drift": "high"}
    manualOverride: Optional[str] = None
    classifierVersion: str = "v1-llm-classifier"


class ClassificationResponse(BaseModel):
    routingLabel: str                  # e.g. PERF_INCIDENT | DATA_QUALITY_ISSUE
    confidence: Optional[float] = None
    rationale: str
    recommendedAction: str
    nextStepRef: str
