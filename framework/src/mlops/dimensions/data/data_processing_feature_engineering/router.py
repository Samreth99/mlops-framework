"""
Data Processing & Feature Engineering router — MLOps

Implements:
  POST   /data/preprocess   – Execute preprocessing pipeline → new dataset version (S3 + DVC)
  POST   /data/validate     – Run data validation checks
  POST   /data/analyze      – Profiling / EDA summaries + drift baseline
  POST   /data/label        – Labeling steps producing labeled dataset versions
  POST   /data/engineer     – Build feature sets from datasets and publish versions

Production fixes:
  - Temp files wrapped in try/finally (no leak on exception)
  - Registry loaded via mtime-based cache (no full scan per request)
  - df.eval() expressions validated with ast before execution (injection guard)
  - Silent eval() failures now log a warning
  - Drift baseline raises 500 on S3 upload failure (never returns deleted path)
"""
from __future__ import annotations

import ast
import json
import logging
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, Optional
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

logger = logging.getLogger(__name__)
router = APIRouter()


def _short_id() -> str:
    return str(uuid4())[:8]


# ── Mtime-based registry cache (issue #25) ──────────────────────────
_registry_cache: Dict[str, Any] = {}
_registry_cache_mtime: float = 0.0
_registry_cache_lock = threading.Lock()
_REGISTRY_FILE = Path(__file__).parent.parent / "data_registry.json"
_DATA_STORAGE = Path(__file__).parent.parent / "data-storage"
_DATA_STORAGE.mkdir(exist_ok=True)


def _get_registry() -> Dict[str, Any]:
    global _registry_cache, _registry_cache_mtime
    with _registry_cache_lock:
        try:
            mtime = _REGISTRY_FILE.stat().st_mtime
            if mtime != _registry_cache_mtime:
                with open(_REGISTRY_FILE, encoding="utf-8") as f:
                    _registry_cache = json.load(f)
                _registry_cache_mtime = mtime
        except Exception as exc:
            logger.warning("Failed to load registry cache: %s", exc)
        return _registry_cache


# ── Temp CSV helper (issue #24 / #27) ───────────────────────────────
@contextmanager
def _temp_csv() -> Generator[str, None, None]:
    """Context manager that yields a temp CSV path and deletes it on exit."""
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
        yield tmp_path
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ── Persistent CSV helper ────────────────────────────────────────────
@contextmanager
def _persistent_csv(prefix: str = "proc") -> Generator[str, None, None]:
    """
    Save to data-storage/ so localRef remains valid after the response.
    Deletes the file only on exception (keeps it on success).
    """
    path = _DATA_STORAGE / f"{prefix}_{uuid4().hex[:8]}.csv"
    try:
        yield str(path)
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ── Expression safety guard (issue #29) ─────────────────────────────
def _is_safe_expression(expr: str) -> bool:
    """
    Validate a pandas eval() expression using the AST.
    Rejects expressions containing function calls, imports, or attribute access
    that could be used for code injection.
    """
    try:
        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Call, ast.Import, ast.ImportFrom, ast.Attribute)):
                return False
        return True
    except SyntaxError:
        return False


# ── Version file resolver ────────────────────────────────────────────
def _resolve_version_file(version_ref: str) -> str:
    """
    Resolve a dataset version reference to a readable local file path.

    Accepts:
      - "{datasetId}/{versionId}"  e.g. "22d278f1/v1"  (preferred)
      - "v{n}"                     bare versionId (scans all datasets)
    Falls back to storageRef when no local file is present.
    """
    reg = _get_registry()
    all_ds_versions: dict = reg.get("dataset_versions", {})

    if "/" in version_ref:
        dataset_id, version_id = version_ref.split("/", 1)
        ver = all_ds_versions.get(dataset_id, {}).get(version_id)
        if ver:
            local_ref = ver.get("localRef", "")
            if local_ref and Path(local_ref).exists():
                return local_ref
            return ver.get("storageRef", version_ref)
        return version_ref

    for ds_versions in all_ds_versions.values():
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

        spec: Dict[str, Any] = req.inlineSpec or {}

        if spec.get("drop_na", True):
            df = df.dropna().reset_index(drop=True)

        # Add a stable row-level entity key if requested (enables online feature serving)
        if spec.get("add_sample_id", False) and "sample_id" not in df.columns:
            df.insert(0, "sample_id", range(len(df)))

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

        with _persistent_csv(prefix=f"preprocess_{output_dataset_id}") as tmp_path:
            df.to_csv(tmp_path, index=False)
            version = svc.create_dataset_version(
                dataset_id=output_dataset_id,
                storage_ref=tmp_path,
                schema_ref=None,
                lineage={"parentVersions": [req.inputDatasetVersion], "transformRef": req.transformSpecRef},
                stats={"rows": len(df), "columns": len(df.columns)},
                ticket_id=req.ticketId,
            )

        event_ref = f"events/preprocess/{version['versionId']}"
        report_ref = f"reports/preprocess/{version['versionId']}"

        return PreprocessResponse(
            outputDatasetVersion=version["versionId"],
            outputDatasetId=output_dataset_id,
            storageRef=version["storageRef"],
            preprocessingReportRef=report_ref,
            eventRef=event_ref,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "version_not_found", "message": str(exc)}},
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "file_not_found", "message": str(exc)}},
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
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "file_not_found", "message": str(exc)}},
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
    When profileConfig.create_drift_baseline is true, saves an Evidently-compatible
    baseline JSON to S3 — the returned driftBaselineRef is used by
    POST /ops/monitoring/rules for drift detection.

    **profileConfig keys:**
      - create_drift_baseline: bool — save baseline to S3
      - sample_size: int
      - include_correlations: bool

    **Returns:** stats summaries, driftBaselineRef (S3 URI), reportRef
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
        safe_desc: Dict[str, Any] = {}
        for col, col_stats in desc.items():
            safe_desc[col] = {k: (None if (isinstance(v, float) and v != v) else v)
                               for k, v in col_stats.items()}

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
        drift_ref: Optional[str] = None

        # ── Save drift baseline to S3 (issue #28: raise on failure, never return deleted path)
        if cfg.get("create_drift_baseline"):
            # Look up storageRef / localRef from registry
            reg = _get_registry()
            all_versions = reg.get("dataset_versions", {})
            ver_rec: Dict[str, Any] = {}
            if "/" in req.datasetVersion:
                ds_id, ver_id = req.datasetVersion.split("/", 1)
                ver_rec = all_versions.get(ds_id, {}).get(ver_id, {})
            else:
                for ds_versions in all_versions.values():
                    if req.datasetVersion in ds_versions:
                        ver_rec = ds_versions[req.datasetVersion]
                        break

            baseline_meta = {
                "datasetVersion": req.datasetVersion,
                "numericalColumns": df.select_dtypes(include="number").columns.tolist(),
                "categoricalColumns": df.select_dtypes(include="object").columns.tolist(),
                "localRef": ver_rec.get("localRef", input_path),
                "storageRef": ver_rec.get("storageRef", input_path),
                "rowCount": len(df),
                "createdAt": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
            }

            baseline_id = _short_id()
            s3_key = f"baselines/drift/{req.datasetVersion.replace('/', '_')}/{baseline_id}.json"

            # Save locally to data-storage/ (persistent, never deleted)
            local_baseline_path = _DATA_STORAGE / f"baseline_drift_{req.datasetVersion.replace('/', '_')}_{baseline_id}.json"
            with open(local_baseline_path, "w", encoding="utf-8") as _f:
                json.dump(baseline_meta, _f, indent=2)

            try:
                # Raise on S3 failure
                drift_ref = svc.upload_file_to_s3(str(local_baseline_path), s3_key)
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail={"error": {"code": "baseline_upload_failed", "message": str(exc)}},
                )

        return AnalyzeResponse(
            datasetVersion=req.datasetVersion,
            stats=stats,
            driftBaselineRef=drift_ref,
            reportRef=report_ref,
        )
    except HTTPException:
        raise
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
            # Coerce map keys to match actual column dtype so int columns
            # match string JSON keys (e.g. {"2": 0, "4": 1} maps int values 2, 4)
            col_dtype = df[target_col].dtype
            try:
                coerced_map = {col_dtype.type(k): v for k, v in label_map.items()}
            except (ValueError, TypeError):
                coerced_map = label_map
            df[target_col] = df[target_col].map(coerced_map).fillna(df[target_col])

        labeled_ds = svc.create_dataset(
            name=f"labeled_{req.inputDatasetVersion}",
            owner=None,
            domain=None,
            description=f"Labeled version of {req.inputDatasetVersion}",
            schema_ref=None,
        )

        with _persistent_csv(prefix=f"label_{labeled_ds['datasetId']}") as tmp_path:
            df.to_csv(tmp_path, index=False)
            version = svc.create_dataset_version(
                dataset_id=labeled_ds["datasetId"],
                storage_ref=tmp_path,
                schema_ref=None,
                lineage={"parentVersions": [req.inputDatasetVersion], "toolingRef": req.toolingRef},
                stats={"rows": len(df), "labeled_column": target_col},
                ticket_id=req.ticketId,
            )

        full_version_ref = f"{labeled_ds['datasetId']}/{version['versionId']}"
        report_ref = f"reports/label/{full_version_ref}"
        return LabelResponse(
            labeledDatasetId=labeled_ds["datasetId"],
            labeledDatasetVersionId=full_version_ref,
            storageRef=version["storageRef"],
            labelQualityReportRef=report_ref,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "version_not_found", "message": str(exc)}},
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

    Feature expressions are validated with the AST before execution to prevent injection.
    """
    make_log(
        area="Data",
        component="Data Processing & Feature Engineering",
        endpoint="/data/engineer",
        meta={"datasetVersion": req.datasetVersion, "entityKeys": req.entityKeys},
    )
    try:
        import pandas as pd
        import numpy as np

        input_path = _resolve_version_file(req.datasetVersion)
        df = pd.read_csv(input_path)

        spec = req.inlineSpec
        feature_schema: Dict[str, Any] = {}
        failed_features: list = []

        # Start with entity keys + explicit passthrough columns
        selected_cols: list = list(req.entityKeys)
        for col in spec.includeColumns:
            if col in df.columns and col not in selected_cols:
                selected_cols.append(col)
                feature_schema[col] = {
                    "dtype": str(df[col].dtype),
                    "nullCount": int(df[col].isnull().sum()),
                    "min": float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                    "max": float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                    "mean": float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                }

        for feat in spec.features:
            name = feat.name
            expr = feat.expression
            passthrough = feat.passthrough

            if expr:
                if not _is_safe_expression(expr):
                    logger.warning("Rejected unsafe feature expression '%s' for '%s'", expr, name)
                    failed_features.append({"name": name, "reason": "unsafe_expression"})
                    continue
                try:
                    df[name] = df.eval(expr)
                    selected_cols.append(name)
                    feature_schema[name] = {
                        "expression": expr,
                        "dtype": str(df[name].dtype),
                        "nullCount": int(df[name].isnull().sum()),
                        "min": float(df[name].min()) if pd.api.types.is_numeric_dtype(df[name]) else None,
                        "max": float(df[name].max()) if pd.api.types.is_numeric_dtype(df[name]) else None,
                        "mean": float(df[name].mean()) if pd.api.types.is_numeric_dtype(df[name]) else None,
                    }
                except Exception as exc:
                    logger.warning("Feature '%s' eval failed: %s", name, exc)
                    failed_features.append({"name": name, "reason": str(exc)})

            elif passthrough and passthrough in df.columns:
                df[name] = df[passthrough]
                selected_cols.append(name)
                feature_schema[name] = {
                    "passthrough": passthrough,
                    "dtype": str(df[name].dtype),
                    "nullCount": int(df[name].isnull().sum()),
                    "min": float(df[name].min()) if pd.api.types.is_numeric_dtype(df[name]) else None,
                    "max": float(df[name].max()) if pd.api.types.is_numeric_dtype(df[name]) else None,
                    "mean": float(df[name].mean()) if pd.api.types.is_numeric_dtype(df[name]) else None,
                }

            elif not expr and not passthrough and name in df.columns:
                selected_cols.append(name)
                feature_schema[name] = {
                    "dtype": str(df[name].dtype),
                    "nullCount": int(df[name].isnull().sum()),
                    "min": float(df[name].min()) if pd.api.types.is_numeric_dtype(df[name]) else None,
                    "max": float(df[name].max()) if pd.api.types.is_numeric_dtype(df[name]) else None,
                    "mean": float(df[name].mean()) if pd.api.types.is_numeric_dtype(df[name]) else None,
                }
            else:
                failed_features.append({"name": name, "reason": "column_not_found_and_no_expression"})

        feature_df = df[[c for c in dict.fromkeys(selected_cols) if c in df.columns]]

        # Save engineered CSV — registration (feature set + version) is done by the caller
        # (e.g. BPMN RegisterFeatureSet → VersionFeature tasks) to avoid double-registration.
        prefix = f"engineer_{req.outputFeatureSetId or uuid4().hex[:8]}"
        with _persistent_csv(prefix=prefix) as tmp_path:
            feature_df.to_csv(tmp_path, index=False)
            local_ref = tmp_path

        return EngineerResponse(
            featureSetVersionId=None,
            featureSetId=req.outputFeatureSetId,
            storageRef=None,
            localRef=local_ref,
            featureSchema=feature_schema,
            failedFeatures=failed_features,
            computationReportRef=None,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "feature_set_not_found", "message": str(exc)}},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "engineer_failed", "message": str(exc)}},
        )
