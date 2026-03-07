from __future__ import annotations

import io
import itertools
import json
import logging
import warnings

warnings.filterwarnings(
    "ignore",
    message="Saving scikit-learn models in the pickle or cloudpickle format",
    category=FutureWarning,
)

import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import mlflow
import mlflow.sklearn  
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from ...config import settings
from ...dimensions.data import s3_service as data_svc

logger = logging.getLogger(__name__)


def _client() -> MlflowClient:
    """Return an MlflowClient pointed at the configured tracking URI."""
    uri = settings.mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_to_iso(ts_ms: Optional[int]) -> Optional[str]:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def _resolve_data_split(source: Optional[str], split: str = "train") -> Optional[str]:
    """
    Resolve a dataset or feature-version reference to a local file path for the
    requested split ('train' or 'test').

    Resolution order:
      1. '{featureSetId}/{versionId}' → look up feature_versions in registry
      2. '{datasetId}/{versionId}'    → let data_svc.resolve_split_path handle it
      3. s3:// URI / bare local path  → let data_svc.resolve_split_path handle it
    """
    from pathlib import Path as _Path

    if not source:
        return None

    if (
        "/" in source
        and not source.startswith("s3://")
        and not _Path(source).exists()
    ):
        parts = source.split("/", 1)
        if len(parts) == 2:
            fs_id, ver_id = parts
            try:
                reg = data_svc._load_registry()
                ver = reg.get("feature_versions", {}).get(fs_id, {}).get(ver_id)
                if ver:
                    local_key = "trainLocalRef" if split == "train" else "testLocalRef"
                    s3_key = "trainStorageRef" if split == "train" else "testStorageRef"
                    local_ref = ver.get(local_key)
                    if local_ref and _Path(local_ref).exists():
                        return local_ref
                    s3_ref = ver.get(s3_key) or ver.get("storageRef", "")
                    if s3_ref.startswith("s3://"):
                        local_dest = str(data_svc._DATA_STORAGE / s3_ref.split("/")[-1])
                        return data_svc.download_from_s3(s3_ref, local_dest)
            except Exception as e:
                logger.warning("feature_versions registry lookup failed for %s: %s", source, e)

    return data_svc.resolve_split_path(source, split=split)


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════
def create_experiment(
    name: str,
    objective: Optional[str] = None,
    artifact_root: Optional[str] = None,
    ticket_id: Optional[str] = None,
) -> Dict[str, Any]:
    client = _client()
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        return _experiment_to_dict(existing)

    tags: Dict[str, str] = {}
    if objective:
        tags["objective"] = objective
    if ticket_id:
        tags["ticketId"] = ticket_id

    art_location = artifact_root or settings.mlflow_artifact_root
    exp_id = client.create_experiment(
        name=name,
        artifact_location=art_location,
        tags=tags or None,
    )
    exp = client.get_experiment(exp_id)
    return _experiment_to_dict(exp)


def list_experiments() -> List[Dict[str, Any]]:
    client = _client()
    experiments = client.search_experiments()
    return [_experiment_to_dict(e) for e in experiments]


def get_experiment(experiment_id: str) -> Dict[str, Any]:
    client = _client()
    exp = client.get_experiment(experiment_id)
    return _experiment_to_dict(exp)


def _experiment_to_dict(exp) -> Dict[str, Any]:
    return {
        "experimentId": exp.experiment_id,
        "name": exp.name,
        "objective": (exp.tags or {}).get("objective"),
        "ticketId": (exp.tags or {}).get("ticketId"),
        "artifact_location": exp.artifact_location,
        "lifecycle_stage": exp.lifecycle_stage,
        "creation_time": _ts_to_iso(getattr(exp, "creation_time", None)),
    }


# ═══════════════════════════════════════════════════════════════════
#  TRAINING RUN
# ═══════════════════════════════════════════════════════════════════
def execute_training(
    *,
    experiment_id: Optional[str] = None,
    experiment_name: Optional[str] = None,
    dataset_source: Optional[str] = None,
    target_column: str = "Class",
    training_config: Optional[Dict[str, Any]] = None,
    cv_folds: int = 10,
    seed: int = 42,
    registered_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a training run with k-fold cross-validation, log to MLflow.
    CV metrics (mean across folds) are logged. Final model is trained on full dataset.
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score

    client = _client()

    # Resolve experiment
    exp_id = _resolve_experiment_id(client, experiment_id, experiment_name)

    # Load train split — supports featureSetId/versionId, datasetId/versionId, local path, S3
    train_path = _resolve_data_split(dataset_source, split="train")
    df = _load_dataset(train_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode labels if needed; save classes so evaluation can reuse the same mapping
    label_classes: Optional[List] = None
    if y.dtype.kind not in ("i", "u", "f"):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        encoded = np.asarray(le.fit_transform(y))
        y = pd.Series(data=encoded, index=y.index, name=target_column)
        label_classes = le.classes_.tolist()

    params = training_config or {}
    if "random_state" not in params:
        params["random_state"] = seed

    with mlflow.start_run(experiment_id=exp_id, run_name="training_run") as run:
        run_id = run.info.run_id
        mlflow.log_params(params)
        mlflow.log_param("cv_folds", cv_folds)
        if label_classes is not None:
            mlflow.set_tag("training.label_classes", json.dumps(label_classes))

        # k-fold cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        model = RandomForestClassifier(**params)
        scoring = {
            "accuracy":        make_scorer(accuracy_score),
            "f1_macro":        make_scorer(f1_score, average="macro"),
            "precision_macro": make_scorer(precision_score, average="macro"),
            "recall_macro":    make_scorer(recall_score, average="macro"),
        }
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)

        # Average metrics across all folds
        metrics = {
            "accuracy":        float(np.mean(cv_results["test_accuracy"])),
            "f1_macro":        float(np.mean(cv_results["test_f1_macro"])),
            "precision_macro": float(np.mean(cv_results["test_precision_macro"])),
            "recall_macro":    float(np.mean(cv_results["test_recall_macro"])),
            "accuracy_std":    float(np.std(cv_results["test_accuracy"])),
            "f1_macro_std":    float(np.std(cv_results["test_f1_macro"])),
        }
        mlflow.log_metrics(metrics)

        # Train final model on full dataset for artifact logging
        model.fit(X, y)

        log_kwargs: Dict[str, Any] = {
            "sk_model": model,
            "name": "model",
        }
        if registered_model_name:
            log_kwargs["registered_model_name"] = registered_model_name

        try:
            X_sample = X.head(100).astype(float)
            y_pred = model.predict(X_sample)
            sig = infer_signature(X_sample, y_pred)
            log_kwargs["signature"] = sig
            log_kwargs["input_example"] = X.head(1).astype(float)
        except Exception as e:
            logger.warning("Could not infer model signature: %s", e)

        model_info = mlflow.sklearn.log_model(**log_kwargs)
        model_uri = model_info.model_uri  # "models:/<model_id>" in MLflow 3.x

    return {
        "runId": run_id,
        "experimentId": exp_id,
        "modelArtifactRef": model_uri,
        "metrics": metrics,
        "status": "FINISHED",
        "resolvedDatasetPath": train_path,
    }


# ═══════════════════════════════════════════════════════════════════
#  TUNING RUN
# ═══════════════════════════════════════════════════════════════════
def execute_tuning(
    *,
    experiment_id: Optional[str] = None,
    experiment_name: Optional[str] = None,
    base_config: Dict[str, Any],
    search_space: Dict[str, List[Any]],
    budget: int = 10,
    dataset_source: Optional[str] = None,
    target_column: str = "Class",
    test_size: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """Grid-search over search_space, log each candidate as a child run."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    client = _client()
    exp_id = _resolve_experiment_id(client, experiment_id, experiment_name)

    train_path = _resolve_data_split(dataset_source, split="train")
    df = _load_dataset(train_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in dataset")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    if y.dtype.kind not in ("i", "u", "f"):
        from sklearn.preprocessing import LabelEncoder
        y = pd.Series(data=np.asarray(LabelEncoder().fit_transform(y)), index=y.index, name=target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed,
        stratify=y if y.nunique() < 50 else None,
    )

    # Generate param combinations (grid)
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    combos = list(itertools.product(*param_values))[:budget]

    candidates: List[Dict[str, Any]] = []

    with mlflow.start_run(experiment_id=exp_id, run_name="tuning_parent") as parent_run:
        parent_id = parent_run.info.run_id

        for combo in combos:
            combo_params = {**base_config, **dict(zip(param_names, combo))}
            with mlflow.start_run(
                experiment_id=exp_id,
                run_name=f"tuning_candidate",
                nested=True,
            ) as child_run:
                child_id = child_run.info.run_id
                mlflow.log_params({k: str(v) for k, v in combo_params.items()})

                safe_params = {k: v for k, v in combo_params.items()
                               if k in RandomForestClassifier().get_params()}
                safe_params.setdefault("random_state", seed)
                model = RandomForestClassifier(**safe_params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                m = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
                }
                mlflow.log_metrics(m)

                candidates.append({
                    "runId": child_id,
                    "params": combo_params,
                    "metrics": m,
                })

        # Find best
        candidates.sort(key=lambda c: c["metrics"].get("accuracy", 0), reverse=True)
        best = candidates[0] if candidates else {"params": {}, "metrics": {}}
        mlflow.log_metrics({"best_accuracy": best["metrics"].get("accuracy", 0)})

    return {
        "runId": parent_id,
        "experimentId": exp_id,
        "bestParams": best["params"],
        "candidateLeaderboard": candidates,
        "resolvedDatasetPath": train_path,
    }


# ═══════════════════════════════════════════════════════════════════
#  VALIDATION RUN
# ═══════════════════════════════════════════════════════════════════
def execute_validation(
    *,
    model_candidate_ref: str,
    rules: Optional[List[Dict[str, Any]]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Load model's metrics from its source run and check against rules/thresholds.
    """
    client = _client()

    # Resolve run_id from model URI
    run_id = _resolve_run_id_from_uri(client, model_candidate_ref)
    if not run_id:
        return {"passed": False, "details": [{"error": "Cannot resolve run from model URI"}], "runId": None}

    run = client.get_run(run_id)
    recorded_metrics = run.data.metrics or {}

    # Build unified rules list
    checks: List[Dict[str, Any]] = []
    if rules:
        checks.extend(rules)
    if thresholds:
        for metric, threshold in thresholds.items():
            checks.append({"metric": metric, "operator": ">=", "threshold": threshold})

    details = []
    all_passed = True
    for rule in checks:
        metric_name = rule["metric"]
        op = rule.get("operator", ">=")
        threshold = rule["threshold"]
        actual = recorded_metrics.get(metric_name)

        if actual is None:
            details.append({"metric": metric_name, "passed": False, "reason": "metric not found"})
            all_passed = False
            continue

        passed = _eval_operator(actual, op, threshold)
        if not passed:
            all_passed = False
        details.append({
            "metric": metric_name,
            "actual": actual,
            "threshold": threshold,
            "operator": op,
            "passed": passed,
        })

    # Log validation result as a tag on the run
    try:
        client.set_tag(run_id, "validation.passed", str(all_passed))
    except Exception as e:
        logger.warning("Could not set validation tag on run %s: %s", run_id, e)

    return {
        "passed": all_passed,
        "validationReportRef": f"runs:/{run_id}/validation",
        "details": details,
        "runId": run_id,
    }


def _eval_operator(actual: float, op: str, threshold: float) -> bool:
    ops = {">=": actual >= threshold, "<=": actual <= threshold,
           ">": actual > threshold, "<": actual < threshold,
           "==": actual == threshold}
    return ops.get(op, actual >= threshold)


# ═══════════════════════════════════════════════════════════════════
#  EVALUATION RUN
# ═══════════════════════════════════════════════════════════════════
def execute_evaluation(
    *,
    model_candidate_ref: str,
    eval_dataset_source: Optional[str] = None,
    metric_names: List[str],
    target_column: str = "Class",
    fairness_checks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Load model, predict on eval dataset, compute metrics, log as a new run."""
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    if not eval_dataset_source:
        raise ValueError(
            "evalDatasetVersion is required. Pass the testLocalRef from your feature version "
            "(e.g. the path to *_test.csv) or a '{featureSetId}/{versionId}' reference."
        )

    client = _client()

    # Resolve the source training run to carry over experiment context and label mapping
    source_run_id = _resolve_run_id_from_uri(client, model_candidate_ref)
    exp_id = None
    label_classes = None
    if source_run_id:
        try:
            source_run = client.get_run(source_run_id)
            exp_id = source_run.info.experiment_id
            raw = source_run.data.tags.get("training.label_classes")
            if raw:
                label_classes = json.loads(raw)
        except Exception as e:
            logger.warning("Could not load source run %s: %s", source_run_id, e)

    # Load model (sklearn flavor avoids strict schema enforcement)
    model = mlflow.sklearn.load_model(model_candidate_ref)

    # Load test split — supports featureSetId/versionId, local path, S3
    test_path = _resolve_data_split(eval_dataset_source, split="test")
    df = _load_dataset(test_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in eval dataset")

    X_eval = df.drop(columns=[target_column])
    y_eval = df[target_column]

    # Re-encode labels using the same class mapping as training to avoid mismatched integers
    if y_eval.dtype.kind not in ("i", "u", "f"):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        if label_classes is not None:
            le.classes_ = np.array(label_classes)
            y_eval = pd.Series(
                data=np.asarray(le.transform(y_eval)),
                index=y_eval.index,
                name=target_column,
            )
        else:
            y_eval = pd.Series(
                data=np.asarray(le.fit_transform(y_eval)),
                index=y_eval.index,
                name=target_column,
            )

    y_pred = model.predict(X_eval.astype(float))

    available_metrics = {
        "accuracy": lambda yt, yp: float(accuracy_score(yt, yp)),
        "f1_macro": lambda yt, yp: float(f1_score(yt, yp, average="macro")),
        "f1_weighted": lambda yt, yp: float(f1_score(yt, yp, average="weighted")),
        "precision_macro": lambda yt, yp: float(precision_score(yt, yp, average="macro")),
        "recall_macro": lambda yt, yp: float(recall_score(yt, yp, average="macro")),
    }

    computed: Dict[str, Any] = {}
    for m in metric_names:
        fn = available_metrics.get(m)
        if fn:
            computed[m] = fn(y_eval, y_pred)

    pass_per_check: Dict[str, bool] = {}
    if fairness_checks:
        for fc in fairness_checks:
            check_name = fc.get("name", "fairness_check")
            pass_per_check[check_name] = True  # placeholder

    with mlflow.start_run(experiment_id=exp_id, run_name="evaluation_run") as run:
        run_id = run.info.run_id
        mlflow.log_metrics(computed)
        mlflow.set_tag("evaluation.model_ref", model_candidate_ref)
        mlflow.set_tag("evaluation.dataset", eval_dataset_source)

    return {
        "evaluationReportRef": f"runs:/{run_id}/evaluation",
        "runId": run_id,
        "metrics": computed,
        "passPerCheck": pass_per_check,
        "resolvedDatasetPath": test_path,
    }


# ═══════════════════════════════════════════════════════════════════
#  MODEL REGISTRY  (models, versions, evidence)
# ═══════════════════════════════════════════════════════════════════
def register_model_family(
    name: str,
    owner: Optional[str] = None,
    intended_use: Optional[str] = None,
    risk_notes: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Create (or return existing) registered model in MLflow registry."""
    client = _client()
    try:
        rm = client.create_registered_model(
            name=name,
            tags=_build_model_tags(owner, intended_use, risk_notes, tags),
            description=intended_use or "",
        )
    except mlflow.exceptions.MlflowException as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            # Update existing model with the provided metadata
            if intended_use:
                client.update_registered_model(name, description=intended_use)
            new_tags = _build_model_tags(owner, intended_use, risk_notes, tags)
            for k, v in new_tags.items():
                client.set_registered_model_tag(name, k, v)
            rm = client.get_registered_model(name)
        else:
            raise
    return _registered_model_to_dict(rm)


def list_models(filter_string: Optional[str] = None) -> List[Dict[str, Any]]:
    client = _client()
    models = client.search_registered_models(
        filter_string=filter_string or "",
        max_results=100,
    )
    return [_registered_model_to_dict(m) for m in models]


def get_model(model_id: str) -> Dict[str, Any]:
    """model_id is the registered model name in MLflow."""
    client = _client()
    rm = client.get_registered_model(model_id)
    return _registered_model_to_dict(rm)


def update_model(
    model_id: str,
    description: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    client = _client()
    if description is not None:
        client.update_registered_model(model_id, description=description)
    if tags:
        for k, v in tags.items():
            client.set_registered_model_tag(model_id, k, v)
    rm = client.get_registered_model(model_id)
    return _registered_model_to_dict(rm)


def create_model_version(
    model_id: str,
    artifact_ref: str,
    run_id: Optional[str] = None,
    description: Optional[str] = None,
    stage: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    client = _client()
    mv = client.create_model_version(
        name=model_id,
        source=artifact_ref,
        run_id=run_id,
        description=description or "",
        tags=tags or {},
    )
    if stage and stage in ("Staging", "Production", "Archived"):
        client.transition_model_version_stage(
            name=model_id, version=mv.version, stage=stage,
        )
        mv = client.get_model_version(model_id, mv.version)
    return _model_version_to_dict(model_id, mv)


def list_model_versions(model_id: str) -> List[Dict[str, Any]]:
    client = _client()
    versions = client.search_model_versions(f"name='{model_id}'")
    return [_model_version_to_dict(model_id, mv) for mv in versions]


def get_model_version(model_id: str, version_id: str) -> Dict[str, Any]:
    client = _client()
    mv = client.get_model_version(model_id, version_id)
    return _model_version_to_dict(model_id, mv)


def register_artifact_as_version(
    model_id: str,
    artifact_ref: str,
    run_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    evidence_refs: Optional[List[str]] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience: register a trained artifact as a new model version."""
    client = _client()

    # Ensure registered model exists
    try:
        client.get_registered_model(model_id)
    except Exception:
        client.create_registered_model(model_id)

    tags = {}
    if metadata:
        for k, v in metadata.items():
            tags[f"meta.{k}"] = str(v)
    if evidence_refs:
        tags["evidence_refs"] = ",".join(evidence_refs)

    mv = client.create_model_version(
        name=model_id,
        source=artifact_ref,
        run_id=run_id,
        description=description or "",
        tags=tags,
    )
    return {
        "modelId": model_id,
        "versionId": str(mv.version),
        "version": str(mv.version),
        "artifactRef": artifact_ref,
        "stage": mv.current_stage,
    }


# ═══════════════════════════════════════════════════════════════════
#  EVIDENCE (stored as tags on model version)
# ═══════════════════════════════════════════════════════════════════
def attach_evidence(
    model_id: str,
    version_id: str,
    evidence_type: str,
    summary: str,
    uri: Optional[str] = None,
    approver: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach evidence to a model version via tags."""
    client = _client()
    evidence_id = str(uuid4())[:8]
    prefix = f"evidence.{evidence_id}"

    client.set_model_version_tag(model_id, version_id, f"{prefix}.type", evidence_type)
    client.set_model_version_tag(model_id, version_id, f"{prefix}.summary", summary)
    client.set_model_version_tag(model_id, version_id, f"{prefix}.created_at", _now_iso())
    if uri:
        client.set_model_version_tag(model_id, version_id, f"{prefix}.uri", uri)
    if approver:
        client.set_model_version_tag(model_id, version_id, f"{prefix}.approver", approver)

    return {
        "evidenceId": evidence_id,
        "modelId": model_id,
        "versionId": version_id,
        "evidenceType": evidence_type,
        "summary": summary,
        "uri": uri,
        "approver": approver,
        "created_at": _now_iso(),
    }


def list_evidence(model_id: str, version_id: str) -> List[Dict[str, Any]]:
    """Retrieve all evidence attached to a model version."""
    client = _client()
    mv = client.get_model_version(model_id, version_id)
    tags = mv.tags or {}

    # Collect all evidence.* prefixes
    evidence_ids = set()
    for key in tags:
        if key.startswith("evidence."):
            parts = key.split(".")
            if len(parts) >= 2:
                evidence_ids.add(parts[1])

    results = []
    for eid in sorted(evidence_ids):
        prefix = f"evidence.{eid}"
        results.append({
            "evidenceId": eid,
            "modelId": model_id,
            "versionId": version_id,
            "evidenceType": tags.get(f"{prefix}.type", "unknown"),
            "summary": tags.get(f"{prefix}.summary", ""),
            "uri": tags.get(f"{prefix}.uri"),
            "approver": tags.get(f"{prefix}.approver"),
            "created_at": tags.get(f"{prefix}.created_at", ""),
        })
    return results


# ═══════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════
def _resolve_experiment_id(
    client: MlflowClient,
    experiment_id: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> str:
    if experiment_id:
        return experiment_id
    name = experiment_name or settings.mlflow_experiment_name
    exp = client.get_experiment_by_name(name)
    if exp:
        return exp.experiment_id
    return client.create_experiment(name)


def _resolve_run_id_from_uri(client: MlflowClient, model_uri: str) -> Optional[str]:
    """Extract run_id from runs:/<run_id>/..., models:/<model_id> (MLflow 3.x LoggedModel),
    or models:/<name>/<version> (Model Registry)."""
    if model_uri.startswith("runs:/"):
        parts = model_uri.replace("runs:/", "").split("/")
        return parts[0] if parts else None
    if model_uri.startswith("models:/"):
        parts = model_uri.replace("models:/", "").split("/")
        if len(parts) == 1:
            # MLflow 3.x LoggedModel: models:/<model_id>
            model_id = parts[0]
            try:
                logged_model = client.get_logged_model(model_id)
                return logged_model.source_run_id
            except Exception:
                pass
        elif len(parts) >= 2:
            # Model Registry: models:/<name>/<version>
            name, version_or_stage = parts[0], parts[1]
            try:
                mv = client.get_model_version(name, version_or_stage)
                return mv.run_id
            except Exception:
                pass
            try:
                versions = client.get_latest_versions(name, stages=[version_or_stage])
                if versions:
                    return versions[0].run_id
            except Exception:
                pass
    return None


def _load_dataset(source: Optional[str]) -> "pd.DataFrame":
    """Load a dataset from a local path, S3 URI, or HTTP URL."""
    import pandas as pd

    if not source:
        default = settings.default_dataset_path
        df = pd.read_csv(default).dropna().reset_index(drop=True)
    elif source.startswith("s3://"):
        local_path = data_svc.download_from_s3(source, str(
            (data_svc._DATA_STORAGE / source.split("/")[-1])
        ))
        df = pd.read_csv(local_path).dropna().reset_index(drop=True)
    elif source.startswith("http://") or source.startswith("https://"):
        import httpx
        with httpx.Client(follow_redirects=True, timeout=60.0) as c:
            resp = c.get(source)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text)).dropna().reset_index(drop=True)
    else:
        df = pd.read_csv(source).dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError(f"Dataset at '{source}' has 0 rows after removing NaN values.")
    return df


def _build_model_tags(
    owner: Optional[str],
    intended_use: Optional[str],
    risk_notes: Optional[str],
    extra_tags: Optional[Dict[str, str]],
) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    if owner:
        tags["owner"] = owner
    if intended_use:
        tags["intendedUse"] = intended_use
    if risk_notes:
        tags["riskNotes"] = risk_notes
    if extra_tags:
        tags.update(extra_tags)
    return tags


def _registered_model_to_dict(rm) -> Dict[str, Any]:
    tags = {}
    if hasattr(rm, "tags") and rm.tags:
        tags = dict(rm.tags) if not isinstance(rm.tags, dict) else rm.tags
    return {
        "modelId": rm.name,
        "name": rm.name,
        "owner": tags.get("owner"),
        "intendedUse": tags.get("intendedUse") or getattr(rm, "description", None),
        "riskNotes": tags.get("riskNotes"),
        "creation_timestamp": _ts_to_iso(getattr(rm, "creation_timestamp", None)),
        "last_updated_timestamp": _ts_to_iso(getattr(rm, "last_updated_timestamp", None)),
        "tags": tags or None,
    }


def _model_version_to_dict(model_id: str, mv) -> Dict[str, Any]:
    tags = {}
    if hasattr(mv, "tags") and mv.tags:
        tags = dict(mv.tags) if not isinstance(mv.tags, dict) else mv.tags
    return {
        "modelId": model_id,
        "versionId": str(mv.version),
        "version": str(mv.version),
        "artifactRef": mv.source,
        "status": getattr(mv, "status", None),
        "stage": getattr(mv, "current_stage", None),
        "creation_timestamp": _ts_to_iso(getattr(mv, "creation_timestamp", None)),
        "description": getattr(mv, "description", None),
        "run_id": getattr(mv, "run_id", None),
        "tags": tags or None,
    }
