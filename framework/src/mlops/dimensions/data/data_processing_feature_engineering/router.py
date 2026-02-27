"""
Data Processing & Feature Engineering router — MLOps

Implements:
  POST   /data/preprocess   – Execute preprocessing pipeline → new dataset version (S3 + DVC)
  POST   /data/validate     – Run data validation checks
  POST   /data/analyze      – Profiling / EDA summaries
  POST   /data/label        – Labeling steps producing labeled dataset versions
  POST   /data/engineer     – Build feature sets from datasets and publish versions
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from ....core.logs import make_log
from ..schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    EngineerRequest,
    EngineerResponse,
    LabelRequest,
    LabelResponse,
    PreprocessRequest,
    PreprocessResponse,
    ValidateRequest,
    ValidateResponse,
)
from .. import s3_service as svc

router = APIRouter()


def _short_id() -> str:
    return str(uuid4())[:8]


def _resolve_version_file(version_ref: str) -> str:
    """
    Resolve a dataset version reference to a readable file path.

    Accepts two formats:
      - "{datasetId}/v{n}"  e.g. "0e0e2043/v1"  (unambiguous, preferred)
      - "v{n}"              e.g. "v1"             (finds the first dataset that
                                                    has this versionId — OK when
                                                    only one dataset exists)
    Falls back to the storageRef when no local file is present.
    """
    import json
    registry_file = Path(__file__).parent.parent / "data_registry.json"
    if not registry_file.exists():
        return version_ref

    with open(registry_file, encoding="utf-8") as f:
        reg = json.load(f)

    all_ds_versions: dict = reg.get("dataset_versions", {})

    # ── Format 1: "datasetId/versionId" ─────────────────────────────
    if "/" in version_ref:
        dataset_id, version_id = version_ref.split("/", 1)
        ver = all_ds_versions.get(dataset_id, {}).get(version_id)
        if ver:
            local_ref = ver.get("localRef", "")
            if local_ref and Path(local_ref).exists():
                return local_ref
            return ver.get("storageRef", version_ref)
        return version_ref

    # ── Format 2: bare "versionId" — scan all datasets ───────────────
    for version_id_key, ds_versions in all_ds_versions.items():
        if version_ref in ds_versions:
            ver = ds_versions[version_ref]
            local_ref = ver.get("localRef", "")
            if local_ref and Path(local_ref).exists():
                return local_ref
            return ver.get("storageRef", version_ref)

    return version_ref


# ─────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────
@router.post(
    "/preprocess",
    response_model=PreprocessResponse,
    summary="Execute preprocessing pipeline → new dataset version",
)
def preprocess(req: PreprocessRequest):
    """
    Load an existing dataset version, apply transform spec, produce a new version
    and upload the result to S3 (with DVC tracking).

    **Inputs:** inputDatasetVersion, transformSpecRef|inlineSpec, outputDatasetId?, ticketId?
    **Returns:** outputDatasetVersion, preprocessing report ref, event ref
    """
    make_log(
        area="Data",
        component="Data Processing & Feature Engineering",
        endpoint="/data/preprocess",
        meta={"inputDatasetVersion": req.inputDatasetVersion},
    )
    try:
        import pandas as pd

        input_path = _resolve_version_file(req.inputDatasetVersion)
        df = pd.read_csv(input_path)

        # Build transform spec
        spec: Dict[str, Any] = req.inlineSpec or {}

        if spec.get("drop_na", True):
            df = df.dropna().reset_index(drop=True)

        fill_na = spec.get("fillna", {})
        if fill_na:
            df = df.fillna(fill_na)

        encode_labels = spec.get("encode_labels", [])
        if encode_labels:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in encode_labels:
                if col in df.columns:
                    df[col] = le.fit_transform(df[col].astype(str))

        drop_cols = spec.get("drop_columns", [])
        if drop_cols:
            df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Write processed file to a temp location
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
        df.to_csv(tmp_path, index=False)

        # Determine output dataset
        output_dataset_id = req.outputDatasetId
        if not output_dataset_id:
            ds = svc.create_dataset(
                name=f"preprocessed_{req.inputDatasetVersion}",
                owner=None,
                domain=None,
                description=f"Auto-created by /data/preprocess from version {req.inputDatasetVersion}",
                schema_ref=None,
            )
            output_dataset_id = ds["datasetId"]

        version = svc.create_dataset_version(
            dataset_id=output_dataset_id,
            storage_ref=tmp_path,
            schema_ref=None,
            lineage={"parentVersions": [req.inputDatasetVersion], "transformRef": req.transformSpecRef},
            stats={"rows": len(df), "columns": len(df.columns)},
            ticket_id=req.ticketId,
        )

        os.unlink(tmp_path)

        event_ref = f"events/preprocess/{version['versionId']}"
        report_ref = f"reports/preprocess/{version['versionId']}"

        return PreprocessResponse(
            outputDatasetVersion=version["versionId"],
            outputDatasetId=output_dataset_id,
            storageRef=version["storageRef"],
            preprocessingReportRef=report_ref,
            eventRef=event_ref,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "preprocess_failed", "message": str(exc)}},
        )


# ─────────────────────────────────────────────
# VALIDATE
# ─────────────────────────────────────────────
@router.post(
    "/validate",
    response_model=ValidateResponse,
    summary="Run data validation checks and return evidence",
)
def validate(req: ValidateRequest):
    """
    Run validation rules against a dataset version and return a pass/fail report.

    **Inputs:** datasetVersion, expectationsRef|rules, severityPolicy?, ticketId?
    **Returns:** passed, validationReportRef, violations list
    """
    make_log(
        area="Data",
        component="Data Processing & Feature Engineering",
        endpoint="/data/validate",
        meta={"datasetVersion": req.datasetVersion, "severityPolicy": req.severityPolicy},
    )
    try:
        import pandas as pd

        input_path = _resolve_version_file(req.datasetVersion)
        df = pd.read_csv(input_path)

        rules = req.rules or []
        violations = []
        all_passed = True

        for rule in rules:
            col = rule.get("column")
            check = rule.get("rule", "not_null")
            value = rule.get("value")

            if col and col not in df.columns:
                violations.append({"column": col, "rule": check, "message": f"Column '{col}' not found"})
                all_passed = False
                continue

            if check == "not_null" and col:
                null_count = int(df[col].isnull().sum())
                if null_count > 0:
                    violations.append({"column": col, "rule": check, "nullCount": null_count, "passed": False})
                    if req.severityPolicy in ("error", "strict"):
                        all_passed = False

            elif check == "min_value" and col and value is not None:
                failing = int((df[col] < float(value)).sum())
                if failing > 0:
                    violations.append({"column": col, "rule": check, "threshold": value, "failingRows": failing, "passed": False})
                    all_passed = False

            elif check == "max_value" and col and value is not None:
                failing = int((df[col] > float(value)).sum())
                if failing > 0:
                    violations.append({"column": col, "rule": check, "threshold": value, "failingRows": failing, "passed": False})
                    all_passed = False

            elif check == "unique" and col:
                dupes = int(df[col].duplicated().sum())
                if dupes > 0:
                    violations.append({"column": col, "rule": check, "duplicateCount": dupes, "passed": False})
                    all_passed = False

            elif check == "row_count_min" and value is not None:
                if len(df) < int(value):
                    violations.append({"rule": check, "expected": value, "actual": len(df), "passed": False})
                    all_passed = False

        report_ref = f"reports/validation/{req.datasetVersion}/{_short_id()}"

        return ValidateResponse(
            passed=all_passed,
            validationReportRef=report_ref,
            violations=violations,
            summary=f"{len(violations)} violation(s) found" if violations else "All checks passed",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "validation_failed", "message": str(exc)}},
        )


# ─────────────────────────────────────────────
# ANALYZE
# ─────────────────────────────────────────────
@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Run profiling / EDA summaries for decisions and monitoring baselines",
)
def analyze(req: AnalyzeRequest):
    """
    Profile a dataset version and return descriptive statistics.

    **Inputs:** datasetVersion, profileConfig?, ticketId?
    **Returns:** stats summaries, drift baseline refs?, report ref
    """
    make_log(
        area="Data",
        component="Data Processing & Feature Engineering",
        endpoint="/data/analyze",
        meta={"datasetVersion": req.datasetVersion},
    )
    try:
        import pandas as pd

        input_path = _resolve_version_file(req.datasetVersion)
        cfg = req.profileConfig or {}

        df = pd.read_csv(input_path)
        sample_size = cfg.get("sample_size")
        if sample_size and len(df) > sample_size:
            df = df.sample(n=int(sample_size), random_state=42)

        desc = df.describe(include="all").to_dict()
        # Make it JSON-safe (replace NaN with None)
        safe_desc: Dict[str, Any] = {}
        for col, stats in desc.items():
            safe_desc[col] = {k: (None if (isinstance(v, float) and v != v) else v)
                               for k, v in stats.items()}

        stats: Dict[str, Any] = {
            "rows": len(df),
            "columns": len(df.columns),
            "columnNames": list(df.columns),
            "nullCounts": df.isnull().sum().to_dict(),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "descriptive": safe_desc,
        }

        if cfg.get("include_correlations", False):
            num_df = df.select_dtypes(include="number")
            if not num_df.empty:
                stats["correlations"] = num_df.corr().to_dict()

        report_ref = f"reports/analyze/{req.datasetVersion}/{_short_id()}"
        drift_ref = f"baselines/drift/{req.datasetVersion}" if cfg.get("create_drift_baseline") else None

        return AnalyzeResponse(
            datasetVersion=req.datasetVersion,
            stats=stats,
            driftBaselineRef=drift_ref,
            reportRef=report_ref,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "analyze_failed", "message": str(exc)}},
        )


# ─────────────────────────────────────────────
# LABEL
# ─────────────────────────────────────────────
@router.post(
    "/label",
    response_model=LabelResponse,
    summary="Execute labeling steps producing labeled dataset versions",
)
def label(req: LabelRequest):
    """
    Apply a label spec to a dataset version and register a new labeled version in S3.

    **Inputs:** inputDatasetVersion, labelSpec, toolingRef?, ticketId?
    **Returns:** labeled dataset version, label quality report ref
    """
    make_log(
        area="Data",
        component="Data Processing & Feature Engineering",
        endpoint="/data/label",
        meta={"inputDatasetVersion": req.inputDatasetVersion},
    )
    try:
        import pandas as pd

        input_path = _resolve_version_file(req.inputDatasetVersion)
        df = pd.read_csv(input_path)

        target_col = req.labelSpec.get("target_column")
        label_map: Dict[str, Any] = req.labelSpec.get("label_map", {})

        if target_col and target_col in df.columns and label_map:
            df[target_col] = df[target_col].map(label_map).fillna(df[target_col])

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
        df.to_csv(tmp_path, index=False)

        labeled_ds = svc.create_dataset(
            name=f"labeled_{req.inputDatasetVersion}",
            owner=None,
            domain=None,
            description=f"Labeled version of {req.inputDatasetVersion}",
            schema_ref=None,
        )
        version = svc.create_dataset_version(
            dataset_id=labeled_ds["datasetId"],
            storage_ref=tmp_path,
            schema_ref=None,
            lineage={"parentVersions": [req.inputDatasetVersion], "toolingRef": req.toolingRef},
            stats={"rows": len(df), "labeled_column": target_col},
            ticket_id=req.ticketId,
        )
        os.unlink(tmp_path)

        report_ref = f"reports/label/{version['versionId']}"
        return LabelResponse(
            labeledDatasetVersionId=version["versionId"],
            storageRef=version["storageRef"],
            labelQualityReportRef=report_ref,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "label_failed", "message": str(exc)}},
        )


# ─────────────────────────────────────────────
# ENGINEER
# ─────────────────────────────────────────────
@router.post(
    "/engineer",
    response_model=EngineerResponse,
    summary="Build feature sets from datasets and publish versions",
)
def engineer(req: EngineerRequest):
    """
    Apply feature engineering spec to a dataset version, publish a new feature set version
    and upload it to S3 (with DVC tracking).

    **Inputs:** datasetVersion, featureDefinitionsRef|inlineSpec, entityKeys, outputFeatureSetId?, ticketId?
    **Returns:** featureSetVersionId, featureSchema, computationReportRef
    """
    make_log(
        area="Data",
        component="Data Processing & Feature Engineering",
        endpoint="/data/engineer",
        meta={"datasetVersion": req.datasetVersion, "entityKeys": req.entityKeys},
    )
    try:
        import pandas as pd

        input_path = _resolve_version_file(req.datasetVersion)
        df = pd.read_csv(input_path)

        spec: Dict[str, Any] = req.inlineSpec or {}
        feature_defs = spec.get("features", [])
        feature_schema: Dict[str, Any] = {}

        # Select/compute features from spec
        selected_cols = list(req.entityKeys)
        for feat in feature_defs:
            name = feat.get("name")
            expr = feat.get("expression")
            if name and expr:
                try:
                    df[name] = df.eval(expr)
                    selected_cols.append(name)
                    feature_schema[name] = {"expression": expr, "dtype": str(df[name].dtype)}
                except Exception:
                    pass
            elif name and name in df.columns:
                selected_cols.append(name)
                feature_schema[name] = {"dtype": str(df[name].dtype)}

        # If no features specified, use all numeric columns
        if not feature_defs:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            selected_cols = list(set(selected_cols + num_cols))
            feature_schema = {c: {"dtype": str(df[c].dtype)} for c in selected_cols if c in df.columns}

        feature_df = df[[c for c in selected_cols if c in df.columns]]

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
        feature_df.to_csv(tmp_path, index=False)

        # Get or create the output feature set
        output_fs_id = req.outputFeatureSetId
        if not output_fs_id:
            fs = svc.create_feature_set(
                name=f"features_{req.datasetVersion}",
                owner=None,
                entity_schema=",".join(req.entityKeys),
                description=f"Auto-engineered from dataset version {req.datasetVersion}",
            )
            output_fs_id = fs["featureSetId"]

        fs_version = svc.create_feature_version(
            fs_id=output_fs_id,
            storage_ref=tmp_path,
            schema_ref=None,
            computed_from=req.datasetVersion,
            ticket_id=req.ticketId,
        )
        os.unlink(tmp_path)

        report_ref = f"reports/engineer/{fs_version['versionId']}"
        return EngineerResponse(
            featureSetVersionId=fs_version["versionId"],
            featureSetId=output_fs_id,
            storageRef=fs_version["storageRef"],
            featureSchema=feature_schema,
            computationReportRef=report_ref,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "engineer_failed", "message": str(exc)}},
        )
