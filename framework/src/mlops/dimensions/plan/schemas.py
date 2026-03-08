from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# PROJECTS
# ─────────────────────────────────────────────
class ProjectCreateRequest(BaseModel):
    name: str = Field(..., description="Project name")
    owner: str = Field(..., description="Project owner / team")
    domain: str = Field(..., description="Business or ML domain (e.g. healthcare, fraud-detection)")
    objective: str = Field(..., description="High-level project objective")
    stakeholders: Optional[List[str]] = Field(None, description="Stakeholder names or emails")


class ProjectUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, description="Updated project name")
    objective: Optional[str] = Field(None, description="Updated objective")
    stakeholders: Optional[List[str]] = Field(None, description="Updated stakeholder list")
    status: Optional[str] = Field(None, description="active | archived | on-hold")


class ProjectResponse(BaseModel):
    projectId: str
    name: str
    owner: str
    domain: str
    objective: str
    stakeholders: List[str] = []
    status: str = "active"
    createdAt: str


# ─────────────────────────────────────────────
# REQUIREMENTS
# ─────────────────────────────────────────────
class RequirementItem(BaseModel):
    id: Optional[str] = Field(None, description="Auto-assigned if omitted")
    text: str = Field(..., description="Requirement description")
    priority: str = Field("medium", description="low | medium | high")
    constraints: Optional[List[str]] = Field(None, description="Technical or regulatory constraints")
    acceptanceCriteria: Optional[List[str]] = Field(None, description="Acceptance criteria list")


class RequirementsRequest(BaseModel):
    requirements: List[RequirementItem]
    ticketId: Optional[str] = Field(None, description="Orchestration ticket reference")


class RequirementsResponse(BaseModel):
    projectId: str
    version: str
    requirements: List[RequirementItem]
    storedAt: str
    diff: Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────────
# ML PROBLEM DEFINITION
# ─────────────────────────────────────────────
class MLProblemRequest(BaseModel):
    taskType: str = Field(..., description="classification | regression | clustering | nlp | anomaly-detection")
    target: str = Field(..., description="Target column or outcome variable")
    metrics: List[str] = Field(..., description="Evaluation metrics (e.g. accuracy, f1_macro, rmse)")
    fairnessOrSafetyConstraints: Optional[List[str]] = Field(None, description="Fairness or safety requirements")
    assumptions: Optional[List[str]] = Field(None, description="Modelling assumptions")
    ticketId: Optional[str] = Field(None, description="Orchestration ticket reference")


class MLGateIdentifier(BaseModel):
    gateId: str
    gateType: str
    threshold: Optional[float] = None


class MLProblemResponse(BaseModel):
    projectId: str
    taskType: str
    target: str
    metrics: List[str]
    fairnessOrSafetyConstraints: List[str] = []
    assumptions: List[str] = []
    gateIdentifiers: List[MLGateIdentifier] = []
    createdAt: str


# ─────────────────────────────────────────────
# PROJECT DATA SOURCES (planning-level)
# ─────────────────────────────────────────────
class PlannedDataSource(BaseModel):
    name: str = Field(..., description="Data source name")
    owner: str = Field(..., description="Source owner / team")
    accessMode: str = Field(..., description="public | internal | restricted")
    refreshCadence: Optional[str] = Field(None, description="How often data refreshes (e.g. daily, weekly)")
    risk: Optional[str] = Field(None, description="Data risk level: low | medium | high")


class ProjectDataSourcesRequest(BaseModel):
    sources: List[PlannedDataSource]
    ticketId: Optional[str] = Field(None, description="Orchestration ticket reference")


class ProjectDataSourcesResponse(BaseModel):
    projectId: str
    sources: List[PlannedDataSource]
    refs: List[str]
    registeredAt: str


# ─────────────────────────────────────────────
# PROJECT PLAN (Roadmap / Milestones)
# ─────────────────────────────────────────────
class Milestone(BaseModel):
    id: Optional[str] = Field(None, description="Auto-assigned if omitted")
    name: str = Field(..., description="Milestone name")
    dueDate: Optional[str] = Field(None, description="ISO-8601 due date")
    status: str = Field("pending", description="pending | in-progress | done")


class Deliverable(BaseModel):
    id: Optional[str] = Field(None, description="Auto-assigned if omitted")
    name: str = Field(..., description="Deliverable name")
    type: str = Field(..., description="dataset | model | report | api | docs")
    milestoneId: Optional[str] = Field(None, description="Parent milestone ID")


class ProjectPlanRequest(BaseModel):
    milestones: List[Milestone]
    deliverables: List[Deliverable]
    timeline: Optional[str] = Field(None, description="Free-text or ISO timeline description")
    roles: Optional[List[str]] = Field(None, description="Roles or team members on this plan")
    ticketId: Optional[str] = Field(None, description="Orchestration ticket reference")


class ProjectPlanResponse(BaseModel):
    projectId: str
    milestoneIds: List[str]
    milestones: List[Milestone]
    deliverables: List[Deliverable]
    timeline: Optional[str] = None
    roles: List[str] = []
    snapshot: Dict[str, Any] = {}
    updatedAt: str


# ─────────────────────────────────────────────
# REPOSITORY BOOTSTRAP
# ─────────────────────────────────────────────
class RepoBootstrapRequest(BaseModel):
    projectId: str = Field(..., description="Project to bootstrap repositories for")
    repoTypes: List[str] = Field(..., description="Types of repos to create: code | data | model")
    provider: str = Field(..., description="Provider: github | gitlab | s3 | mlflow")
    naming: Optional[Dict[str, str]] = Field(None, description="Custom names per repo type, e.g. {code: my-repo}")
    accessPolicy: Optional[str] = Field(None, description="Access policy: private | internal | public")


class RepoInfo(BaseModel):
    repoType: str
    uri: str
    id: str


class RepoBootstrapResponse(BaseModel):
    projectId: str
    repos: List[RepoInfo]
    accessMetadata: Dict[str, Any] = {}
    bootstrappedAt: str


# ─────────────────────────────────────────────
# TICKET GENERATION
# ─────────────────────────────────────────────
class WorkItem(BaseModel):
    type: str = Field(..., description="task | bug | epic | story")
    description: str = Field(..., description="Work item description")
    priority: str = Field("medium", description="low | medium | high")
    owner: Optional[str] = Field(None, description="Assigned owner")


class TicketGenerationRequest(BaseModel):
    projectId: str = Field(..., description="Project the tickets belong to")
    workItems: List[WorkItem]
    targetContract: Optional[str] = Field(None, description="Target contract or SLA reference")


class TicketHint(BaseModel):
    ticketId: str
    workItemType: str
    description: str
    routingHint: Optional[str] = None


class TicketGenerationResponse(BaseModel):
    projectId: str
    ticketIds: List[str]
    tickets: List[TicketHint]
    routingHints: Dict[str, str] = {}
