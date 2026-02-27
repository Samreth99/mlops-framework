from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Sources
# ─────────────────────────────────────────────
class SourceCreateRequest(BaseModel):
    name: str = Field(..., description="Human-readable connector name")
    type: str = Field(..., description="Connector type: csv, parquet, json, db, api, s3")
    connectionRef: str = Field(..., description="Connection string, path, or URI")
    schemaRef: Optional[str] = Field(None, description="Optional schema reference")
    owner: Optional[str] = Field(None, description="Owner identifier")
    refreshCadence: Optional[str] = Field(None, description="Cron or interval (e.g. @daily)")


class SourceResponse(BaseModel):
    sourceId: str
    name: str
    type: str
    connectionRef: str
    schemaRef: Optional[str] = None
    owner: Optional[str] = None
    refreshCadence: Optional[str] = None
    status: str
    createdAt: str


# ─────────────────────────────────────────────
# Ingestions
# ─────────────────────────────────────────────
class IngestionCreateRequest(BaseModel):
    sourceId: str = Field(..., description="Source connector ID")
    mode: str = Field(..., description="batch or stream")
    window: Optional[str] = Field(None, description="Time window for batch: e.g. 2024-01-01/2024-01-31")
    params: Optional[Dict[str, Any]] = Field(None, description="Extra params, e.g. {'localFilePath': '/path/to/file.csv'}")
    ticketId: Optional[str] = Field(None, description="Linked orchestration ticket")


class IngestionResponse(BaseModel):
    ingestionId: str
    runId: Optional[str] = None
    sourceId: str
    mode: str
    status: str
    accepted: bool = True
    createdAt: str


class IngestionDetailResponse(BaseModel):
    ingestionId: str
    runId: Optional[str] = None
    sourceId: str
    sourceName: Optional[str] = None
    connectionRef: Optional[str] = None
    mode: str
    window: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    ticketId: Optional[str] = None
    status: str
    outputDatasetVersionId: Optional[str] = None
    s3Ref: Optional[str] = None
    logs: Optional[List[str]] = None
    createdAt: str


class IngestionStatusResponse(BaseModel):
    ingestionId: str
    state: str
    progress: float = Field(0.0, description="Progress 0–100")
    errorSummary: Optional[str] = None
    outputRefs: List[str] = []


# ─────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────
class DatasetCreateRequest(BaseModel):
    name: str = Field(..., description="Logical dataset name")
    owner: Optional[str] = None
    domain: Optional[str] = Field(None, description="Business domain: medical, finance, etc.")
    description: Optional[str] = None
    schemaRef: Optional[str] = None


class DatasetUpdateRequest(BaseModel):
    description: Optional[str] = None
    owners: Optional[List[str]] = None
    schemaRef: Optional[str] = None


class DatasetResponse(BaseModel):
    datasetId: str
    name: str
    owner: Optional[str] = None
    domain: Optional[str] = None
    description: Optional[str] = None
    schemaRef: Optional[str] = None
    createdAt: str


class DatasetVersionCreateRequest(BaseModel):
    storageRef: str = Field(..., description="Local file path or existing S3 URI to the dataset")
    schemaRef: Optional[str] = Field(None, description="Schema reference or inline schema JSON")
    lineage: Optional[Dict[str, Any]] = Field(
        None, description="Provenance: {parentVersions: [], transformRef: str}"
    )
    stats: Optional[Dict[str, Any]] = Field(None, description="Pre-computed statistics snapshot")
    ticketId: Optional[str] = None


class DatasetVersionResponse(BaseModel):
    versionId: str
    datasetId: str
    storageRef: str
    trainStorageRef: Optional[str] = None
    testStorageRef: Optional[str] = None
    digest: Optional[str] = None
    schemaRef: Optional[str] = None
    lineage: Optional[Dict[str, Any]] = None
    stats: Optional[Dict[str, Any]] = None
    dvcTracked: bool = False
    createdBy: Optional[str] = None
    createdAt: str


class LineageResponse(BaseModel):
    datasetId: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


# ─────────────────────────────────────────────
# Preprocessing / Validation / Analysis / Label / Engineer
# ─────────────────────────────────────────────
class PreprocessRequest(BaseModel):
    inputDatasetVersion: str = Field(..., description="versionId of input dataset")
    transformSpecRef: Optional[str] = Field(None, description="Path to transform spec file")
    inlineSpec: Optional[Dict[str, Any]] = Field(
        None, description="Inline transform spec: {drop_na: true, fillna: {}, encode_labels: []}"
    )
    outputDatasetId: Optional[str] = Field(None, description="Target dataset ID (created if absent)")
    ticketId: Optional[str] = None


class PreprocessResponse(BaseModel):
    outputDatasetVersion: str
    outputDatasetId: str
    storageRef: str
    preprocessingReportRef: str
    eventRef: str


class ValidateRequest(BaseModel):
    datasetVersion: str = Field(..., description="versionId to validate")
    expectationsRef: Optional[str] = Field(None, description="Path to expectations file (JSON/YAML)")
    rules: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Inline rules: [{column, rule, value}], e.g. {column:'age', rule:'not_null'}"
    )
    severityPolicy: Optional[str] = Field("warn", description="warn | error | strict")
    ticketId: Optional[str] = None


class ValidateResponse(BaseModel):
    passed: bool
    validationReportRef: str
    violations: List[Dict[str, Any]] = []
    summary: Optional[str] = None


class AnalyzeRequest(BaseModel):
    datasetVersion: str = Field(..., description="versionId to profile")
    profileConfig: Optional[Dict[str, Any]] = Field(
        None, description="Profile config: {include_correlations: true, sample_size: 1000}"
    )
    ticketId: Optional[str] = None


class AnalyzeResponse(BaseModel):
    datasetVersion: str
    stats: Dict[str, Any]
    driftBaselineRef: Optional[str] = None
    reportRef: str


class LabelRequest(BaseModel):
    inputDatasetVersion: str = Field(..., description="versionId of dataset to label")
    labelSpec: Dict[str, Any] = Field(..., description="Label spec: {target_column, label_map}")
    toolingRef: Optional[str] = Field(None, description="External labeling tool reference")
    ticketId: Optional[str] = None


class LabelResponse(BaseModel):
    labeledDatasetVersionId: str
    storageRef: str
    labelQualityReportRef: str


class EngineerRequest(BaseModel):
    datasetVersion: str = Field(..., description="versionId of source dataset")
    featureDefinitionsRef: Optional[str] = Field(None, description="Path to feature definitions file")
    inlineSpec: Optional[Dict[str, Any]] = Field(
        None, description="Inline spec: {features: [{name, expression}], entity_keys: []}"
    )
    entityKeys: List[str] = Field(default_factory=list, description="Entity key columns")
    outputFeatureSetId: Optional[str] = Field(None, description="Target feature set ID")
    ticketId: Optional[str] = None


class EngineerResponse(BaseModel):
    featureSetVersionId: str
    featureSetId: str
    storageRef: str
    featureSchema: Dict[str, Any]
    computationReportRef: str


# ─────────────────────────────────────────────
# Feature Sets
# ─────────────────────────────────────────────
class FeatureSetCreateRequest(BaseModel):
    name: str = Field(..., description="Feature set name")
    owner: Optional[str] = None
    entitySchema: Optional[str] = Field(None, description="Entity key schema description")
    description: Optional[str] = None


class FeatureSetUpdateRequest(BaseModel):
    description: Optional[str] = None
    owner: Optional[str] = None
    entitySchema: Optional[str] = None


class FeatureSetResponse(BaseModel):
    featureSetId: str
    name: str
    owner: Optional[str] = None
    entitySchema: Optional[str] = None
    description: Optional[str] = None
    createdAt: str


class FeatureVersionCreateRequest(BaseModel):
    storageRef: str = Field(..., description="Local file path or S3 URI to feature table")
    schemaRef: Optional[str] = Field(None, description="Feature schema reference")
    computedFrom: Optional[str] = Field(None, description="Source datasetVersionId")
    ticketId: Optional[str] = None


class FeatureVersionResponse(BaseModel):
    versionId: str
    featureSetId: str
    storageRef: str
    schemaRef: Optional[str] = None
    computedFrom: Optional[str] = None
    digest: Optional[str] = None
    dvcTracked: bool = False
    createdAt: str


class OnlineFeatureGetResponse(BaseModel):
    featureSetId: str
    versionId: Optional[str] = None
    entityKeyValues: Dict[str, Any]
    features: Dict[str, Any]
    asOfTime: Optional[str] = None
    retrievalMetadata: Dict[str, Any] = {}


class OfflineFeatureGetResponse(BaseModel):
    featureSetId: str
    versionId: Optional[str] = None
    storageRef: str
    columnSchema: Dict[str, Any]
    rowCount: Optional[int] = None
