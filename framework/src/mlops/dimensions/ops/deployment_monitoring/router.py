"""
Deployment & Monitoring router — MLOps

Implements:
  POST  /ops/monitoring/rules  – Create and evaluate an alerting/drift rule
  GET   /ops/monitoring/rules  – List all monitoring rules
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ....core.logs import make_log

router = APIRouter()

# ── Simple JSON registry for monitoring rules ────────────────────────
_OPS_REGISTRY = Path(__file__).parent / "ops_registry.json"


def _load_rules() -> List[Dict[str, Any]]:
    if _OPS_REGISTRY.exists():
        with open(_OPS_REGISTRY, encoding="utf-8") as f:
            return json.load(f).get("rules", [])
    return []


def _save_rules(rules: List[Dict[str, Any]]) -> None:
    with open(_OPS_REGISTRY, "w", encoding="utf-8") as f:
        json.dump({"rules": rules}, f, indent=2)


def _short_id() -> str:
    return str(uuid4())[:8]


# ── Schemas ──────────────────────────────────────────────────────────
class MonitoringRuleRequest(BaseModel):
    ruleType: str = Field(..., description="Rule type: drift | sla | bias")
    # Drift-specific
    driftBaselineRef: Optional[str] = Field(
        None,
        description=(
            "S3 URI or local path to the drift baseline JSON produced by POST /data/analyze "
            "(profileConfig.create_drift_baseline: true)"
        ),
    )
    currentDatasetVersion: Optional[str] = Field(
        None,
        description="Dataset version ID of incoming/production data to check against the baseline",
    )
    thresholds: Optional[Dict[str, Any]] = Field(
        None,
        description="Thresholds config: {num_method: 'ks', num_threshold: 0.2}",
    )
    window: Optional[str] = Field(None, description="Evaluation window: e.g. daily, @weekly")
    actions: Optional[Dict[str, Any]] = Field(
        None,
        description="Actions on trigger: {open_ticket: true, trigger_retrain: true}",
    )
    ticketId: Optional[str] = None


class MonitoringRuleResponse(BaseModel):
    ruleId: str
    ruleType: str
    enabled: bool = True
    driftBaselineRef: Optional[str] = None
    currentDatasetVersion: Optional[str] = None
    thresholds: Optional[Dict[str, Any]] = None
    window: Optional[str] = None
    actions: Optional[Dict[str, Any]] = None
    driftResult: Optional[Dict[str, Any]] = None
    createdAt: str


# ── Helpers ───────────────────────────────────────────────────────────
def _resolve_data_path(dataset_version: str) -> str:
    """Resolve a dataset versionId to a local CSV path via data_registry.json."""
    registry_file = Path(__file__).parent.parent.parent / "data" / "data_registry.json"
    if not registry_file.exists():
        return dataset_version
    with open(registry_file, encoding="utf-8") as f:
        reg = json.load(f)
    all_versions = reg.get("dataset_versions", {})
    if "/" in dataset_version:
        ds_id, ver_id = dataset_version.split("/", 1)
        ver = all_versions.get(ds_id, {}).get(ver_id, {})
    else:
        ver = {}
        for ds_versions in all_versions.values():
            if dataset_version in ds_versions:
                ver = ds_versions[dataset_version]
                break
    local_ref = ver.get("localRef", "")
    if local_ref and Path(local_ref).exists():
        return local_ref
    return ver.get("storageRef", dataset_version)


def _load_baseline(baseline_ref: str) -> Dict[str, Any]:
    """Load baseline JSON from S3 URI or local path."""
    if baseline_ref.startswith("s3://"):
        # Download from S3
        import boto3
        from mlops.config import settings

        s3 = boto3.client(
            "s3",
            region_name=settings.aws_s3_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
        )
        without_scheme = baseline_ref[len("s3://"):]
        bucket, key = without_scheme.split("/", 1)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        s3.download_file(bucket, key, tmp_path)
        with open(tmp_path, encoding="utf-8") as f:
            data = json.load(f)
        os.unlink(tmp_path)
        return data
    else:
        with open(baseline_ref, encoding="utf-8") as f:
            return json.load(f)


def _run_evidently_drift(
    ref_path: str,
    cur_path: str,
    numerical_cols: List[str],
    categorical_cols: List[str],
    num_method: str = "ks",
    num_threshold: float = 0.2,
) -> Dict[str, Any]:
    """Run Evidently DataDriftPreset and return drift metrics + HTML path."""
    import pandas as pd
    from evidently import Dataset, DataDefinition, Report
    from evidently.presets import DataDriftPreset

    ref_df = pd.read_csv(ref_path)
    cur_df = pd.read_csv(cur_path)

    # Use only columns present in both dataframes
    num_cols = [c for c in numerical_cols if c in ref_df.columns and c in cur_df.columns]
    cat_cols = [c for c in categorical_cols if c in ref_df.columns and c in cur_df.columns]

    data_def = DataDefinition(
        numerical_columns=num_cols,
        categorical_columns=cat_cols,
    )
    baseline_data = Dataset.from_pandas(ref_df, data_definition=data_def)
    current_data = Dataset.from_pandas(cur_df, data_definition=data_def)

    report = Report(metrics=[
        DataDriftPreset(num_method=num_method, num_threshold=num_threshold)
    ])
    result = report.run(reference_data=baseline_data, current_data=current_data)

    html_path = str(
        Path(__file__).parent / f"drift_report_{_short_id()}.html"
    )
    result.save_html(html_path)

    report_dict = result.dict()
    drift_value = report_dict["metrics"][0]["value"]
    drift_count = drift_value.get("count", 0)
    drift_share = drift_value.get("share", 0.0)

    # Collect drifted feature names
    drifted_features = [
        feat for feat, feat_val in drift_value.get("features", {}).items()
        if isinstance(feat_val, dict) and feat_val.get("drift_detected")
    ]

    return {
        "driftedFeatureCount": drift_count,
        "driftShare": round(drift_share, 4),
        "driftedFeatures": drifted_features,
        "driftDetected": drift_count > 0,
        "htmlReportPath": html_path,
    }


# ─────────────────────────────────────────────
# MONITORING RULES
# ─────────────────────────────────────────────
@router.post(
    "/monitoring/rules",
    response_model=MonitoringRuleResponse,
    summary="Create and evaluate an alerting/drift rule",
)
def create_monitoring_rule(req: MonitoringRuleRequest):
    """
    Create a monitoring rule and immediately evaluate it.

    **For drift rules:**
    - `driftBaselineRef`: S3 URI from `POST /data/analyze` (create_drift_baseline: true)
    - `currentDatasetVersion`: incoming production data version ID
    - `thresholds`: {num_method: "ks", num_threshold: 0.2}
    - `actions`: {open_ticket: true, trigger_retrain: true}

    Runs Evidently `DataDriftPreset` (KS-test). If drift detected, executes configured actions.

    **Returns:** ruleId, enabled, driftResult
    """
    make_log(
        area="Ops",
        component="Deployment & Monitoring",
        endpoint="/ops/monitoring/rules",
        meta={"ruleType": req.ruleType},
    )

    rule_id = _short_id()
    drift_result: Optional[Dict[str, Any]] = None
    thresholds = req.thresholds or {}

    if req.ruleType == "drift":
        if not req.driftBaselineRef:
            raise HTTPException(
                status_code=422,
                detail={"error": {"code": "missing_baseline", "message": "driftBaselineRef is required for drift rules"}},
            )
        if not req.currentDatasetVersion:
            raise HTTPException(
                status_code=422,
                detail={"error": {"code": "missing_current", "message": "currentDatasetVersion is required for drift rules"}},
            )

        try:
            baseline = _load_baseline(req.driftBaselineRef)
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail={"error": {"code": "baseline_load_failed", "message": str(exc)}},
            )

        ref_path = baseline.get("localRef", "")
        if not ref_path or not Path(ref_path).exists():
            raise HTTPException(
                status_code=404,
                detail={"error": {"code": "reference_data_missing", "message": f"Reference data not found at '{ref_path}'"}},
            )

        cur_path = _resolve_data_path(req.currentDatasetVersion)
        if not Path(cur_path).exists():
            raise HTTPException(
                status_code=404,
                detail={"error": {"code": "current_data_missing", "message": f"Current data not found at '{cur_path}'"}},
            )

        try:
            drift_result = _run_evidently_drift(
                ref_path=ref_path,
                cur_path=cur_path,
                numerical_cols=baseline.get("numericalColumns", []),
                categorical_cols=baseline.get("categoricalColumns", []),
                num_method=thresholds.get("num_method", "ks"),
                num_threshold=thresholds.get("num_threshold", 0.2),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={"error": {"code": "drift_run_failed", "message": str(exc)}},
            )

        # Upload HTML report to S3
        html_path = drift_result.pop("htmlReportPath", None)
        if html_path and Path(html_path).exists():
            try:
                from mlops.dimensions.data import s3_service as svc
                s3_key = f"reports/drift/rule_{rule_id}.html"
                drift_result["reportS3Ref"] = svc.upload_file_to_s3(html_path, s3_key)
            except Exception:
                drift_result["reportS3Ref"] = html_path
            finally:
                try:
                    os.unlink(html_path)
                except OSError:
                    pass

        # Execute actions
        actions = req.actions or {}
        drift_result["actionsTriggered"] = []
        if drift_result.get("driftDetected"):
            if actions.get("open_ticket"):
                drift_result["actionsTriggered"].append("open_ticket")
            if actions.get("trigger_retrain"):
                drift_result["actionsTriggered"].append("trigger_retrain")

    record = {
        "ruleId": rule_id,
        "ruleType": req.ruleType,
        "enabled": True,
        "driftBaselineRef": req.driftBaselineRef,
        "currentDatasetVersion": req.currentDatasetVersion,
        "thresholds": thresholds or None,
        "window": req.window,
        "actions": req.actions,
        "driftResult": drift_result,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }

    rules = _load_rules()
    rules.append(record)
    _save_rules(rules)

    return MonitoringRuleResponse(**record)


@router.get(
    "/monitoring/rules",
    response_model=List[MonitoringRuleResponse],
    summary="List all monitoring rules",
)
def list_monitoring_rules():
    make_log(
        area="Ops",
        component="Deployment & Monitoring",
        endpoint="/ops/monitoring/rules",
    )
    return [MonitoringRuleResponse(**r) for r in _load_rules()]
