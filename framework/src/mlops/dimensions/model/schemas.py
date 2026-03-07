from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# ─────────────────────────────────────────────
# Experiments
# ─────────────────────────────────────────────
class ExperimentCreateRequest(BaseModel):
    projectId: Optional[str] = Field(None, description="Associated project identifier")
    name: str = Field(..., description="Experiment name")
    objective: Optional[str] = Field(None, description="Experiment objective or description")
    ticketId: Optional[str] = Field(None, description="Linked orchestration ticket")


class ExperimentResponse(BaseModel):
    experimentId: str
    name: str
    objective: Optional[str] = None
    ticketId: Optional[str] = None
    artifact_location: Optional[str] = None
    lifecycle_stage: Optional[str] = None
    creation_time: Optional[str] = None


# ─────────────────────────────────────────────
# Runs – Execute Training
# ─────────────────────────────────────────────
class ExecuteTrainingRequest(BaseModel):
    experimentId: Optional[str] = Field(None, description="MLflow experiment ID or name")
    datasetVersion: Optional[str] = Field(None, description="Dataset version ref (path or URL)")
    featureSetVersion: Optional[str] = Field(None, description="Feature set version ref")
    trainingConfig: Dict[str, Any] = Field(default_factory=dict, description="Training hyperparameters")
    target_column: str = Field("Class", description="Target column name")
    cv_folds: int = Field(10, description="Number of cross-validation folds")
    seed: Optional[int] = Field(42, description="Random seed for reproducibility")
    ticketId: str = Field(..., description="Linked orchestration ticket")
    registered_model_name: Optional[str] = Field(None, description="If set, register model under this name")


class ExecuteTrainingResponse(BaseModel):
    runId: str
    experimentId: str
    modelArtifactRef: str
    metrics: Dict[str, Any] = {}
    status: str = "FINISHED"
    resolvedDatasetPath: Optional[str] = None


# ─────────────────────────────────────────────
# Runs – Execute Tuning
# ─────────────────────────────────────────────
class ExecuteTuningRequest(BaseModel):
    experimentId: Optional[str] = Field(None)
    baseConfig: Dict[str, Any] = Field(default_factory=dict, description="Base hyperparameters")
    searchSpace: Dict[str, List[Any]] = Field(
        default_factory=dict,
        description="Param name → list of candidate values for grid search",
    )
    budget: int = Field(10, description="Max number of tuning candidates to try")
    datasetVersion: Optional[str] = Field(None, description="Dataset path or URL")
    featureSetVersion: Optional[str] = Field(None)
    target_column: str = Field("Class")
    test_size: float = 0.2
    seed: Optional[int] = 42
    ticketId: str = Field(..., description="Linked orchestration ticket")


class TuningCandidate(BaseModel):
    runId: str
    params: Dict[str, Any]
    metrics: Dict[str, Any]


class ExecuteTuningResponse(BaseModel):
    runId: str
    experimentId: str
    bestParams: Dict[str, Any]
    candidateLeaderboard: List[TuningCandidate]
    resolvedDatasetPath: Optional[str] = None


# ─────────────────────────────────────────────
# Runs – Execute Validation
# ─────────────────────────────────────────────
class ValidationRule(BaseModel):
    metric: str
    operator: str = Field(">=", description="Comparison operator: >=, <=, >, <, ==")
    threshold: float


class ExecuteValidationRequest(BaseModel):
    modelCandidateRef: str = Field(..., description="Model URI (runs:/ or models:/)")
    validationSuiteRef: Optional[str] = Field(None, description="Optional external validation suite")
    rules: List[ValidationRule] = Field(default_factory=list, description="Validation threshold rules")
    thresholds: Optional[Dict[str, float]] = Field(None, description="Simple metric→threshold map (alternative to rules)")
    ticketId: str = Field(..., description="Linked orchestration ticket")


class ExecuteValidationResponse(BaseModel):
    passed: bool
    validationReportRef: Optional[str] = None
    details: List[Dict[str, Any]] = []
    runId: Optional[str] = None


# ─────────────────────────────────────────────
# Runs – Execute Evaluation
# ─────────────────────────────────────────────
class ExecuteEvaluationRequest(BaseModel):
    modelCandidateRef: str = Field(..., description="Model URI")
    evalDatasetVersion: Optional[str] = Field(None, description="Evaluation dataset (path or URL). Defaults to configured test dataset.")
    metrics: List[str] = Field(default_factory=lambda: ["accuracy", "f1_macro"])
    fairnessChecks: Optional[List[Dict[str, Any]]] = None
    target_column: str = Field("Class")
    ticketId: str = Field(..., description="Linked orchestration ticket")


class ExecuteEvaluationResponse(BaseModel):
    evaluationReportRef: Optional[str] = None
    runId: str
    metrics: Dict[str, Any]
    passPerCheck: Dict[str, bool] = {}
    resolvedDatasetPath: Optional[str] = None


# ─────────────────────────────────────────────
# Models (Registry)
# ─────────────────────────────────────────────
class ModelCreateRequest(BaseModel):
    name: str = Field(..., description="Registered model name")
    owner: Optional[str] = None
    intendedUse: Optional[str] = None
    riskNotes: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


class ModelResponse(BaseModel):
    modelId: str
    name: str
    owner: Optional[str] = None
    intendedUse: Optional[str] = None
    riskNotes: Optional[str] = None
    creation_timestamp: Optional[str] = None
    last_updated_timestamp: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


class ModelUpdateRequest(BaseModel):
    intendedUse: Optional[str] = None
    owners: Optional[List[str]] = None
    riskNotes: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


# ─────────────────────────────────────────────
# Model Versions
# ─────────────────────────────────────────────
class ModelVersionCreateRequest(BaseModel):
    artifactRef: str = Field(..., description="Source artifact URI (runs:/<run_id>/model)")
    trainingRunId: Optional[str] = None
    metricsRef: Optional[str] = None
    datasetRef: Optional[str] = None
    featureRef: Optional[str] = None
    ticketId: str = Field(..., description="Linked orchestration ticket")
    stage: Optional[str] = Field(None, description="Target stage: None, Staging, Production, Archived")
    description: Optional[str] = None


class ModelVersionResponse(BaseModel):
    modelId: str
    versionId: str
    version: str
    artifactRef: str
    status: Optional[str] = None
    stage: Optional[str] = None
    creation_timestamp: Optional[str] = None
    description: Optional[str] = None
    run_id: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


# ─────────────────────────────────────────────
# Model Register (convenience)
# ─────────────────────────────────────────────
class ModelRegisterRequest(BaseModel):
    artifactRef: str = Field(..., description="Model artifact URI to register")
    runId: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    evidenceRefs: Optional[List[str]] = None
    ticketId: str = Field(..., description="Linked orchestration ticket")
    description: Optional[str] = None


class ModelRegisterResponse(BaseModel):
    modelId: str
    versionId: str
    version: str
    artifactRef: str
    stage: Optional[str] = None


# ─────────────────────────────────────────────
# Evidence
# ─────────────────────────────────────────────
class EvidenceAttachRequest(BaseModel):
    evidenceType: str = Field(..., description="report | approval | log")
    uri: Optional[str] = Field(None, description="URI or blob reference for the evidence")
    blobRef: Optional[str] = None
    summary: str = Field("", description="Human-readable summary")
    approver: Optional[str] = None
    ticketId: Optional[str] = None
    artifactRefs: Optional[List[str]] = None


class EvidenceResponse(BaseModel):
    evidenceId: str
    modelId: str
    versionId: str
    evidenceType: str
    summary: str
    uri: Optional[str] = None
    approver: Optional[str] = None
    created_at: str
    tags: Optional[Dict[str, str]] = None
