"""
S3 + DVC service layer for the Data dimension.

Responsibilities:
  - Upload / download files to the S3 bucket via boto3
  - Track files with DVC (dvc add) and push metadata to S3 remote (dvc push)
  - Maintain a JSON registry for sources, ingestions, datasets and features
    so that metadata survives process restarts without needing a database.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from ...config import settings

logger = logging.getLogger(__name__)

# ── DVC project root (walk up until we find the .dvc/ folder) ──────
_HERE = Path(__file__).resolve()
_DVC_ROOT: Path = _HERE
while _DVC_ROOT != _DVC_ROOT.parent:
    if (_DVC_ROOT / ".dvc").is_dir():
        break
    _DVC_ROOT = _DVC_ROOT.parent

# ── Local data-storage folder (train / test splits land here) ───────
_DATA_STORAGE = Path(__file__).parent / "data-storage"
_DATA_STORAGE.mkdir(exist_ok=True)

# ── Registry file (JSON) ────────────────────────────────────────────
_REGISTRY_FILE = Path(__file__).parent / "data_registry.json"


def _load_registry() -> Dict[str, Any]:
    if _REGISTRY_FILE.exists():
        with open(_REGISTRY_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {
        "sources": {},
        "ingestions": {},
        "datasets": {},
        "dataset_versions": {},
        "features": {},
        "feature_versions": {},
    }


def _save_registry(reg: Dict[str, Any]) -> None:
    with open(_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, default=str)


# ── Helpers ─────────────────────────────────────────────────────────
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id() -> str:
    return str(uuid4())[:8]


def _s3_client():
    return boto3.client(
        "s3",
        region_name=settings.aws_s3_region,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )


def compute_digest(file_path: str) -> str:
    """Return the MD5 hex-digest of a local file."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


# ── S3 operations ────────────────────────────────────────────────────
def upload_file_to_s3(local_path: str, s3_key: str) -> str:
    """Upload a local file to S3 and return its s3:// URI."""
    s3 = _s3_client()
    s3.upload_file(local_path, settings.aws_s3_bucket, s3_key)
    uri = f"s3://{settings.aws_s3_bucket}/{s3_key}"
    logger.info("Uploaded %s → %s", local_path, uri)
    return uri


def download_from_s3(s3_uri: str, local_dest: str) -> str:
    """Download a file from an S3 URI to local_dest and return the local path."""
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    Path(local_dest).parent.mkdir(parents=True, exist_ok=True)
    _s3_client().download_file(bucket, key, local_dest)
    logger.info("Downloaded %s → %s", s3_uri, local_dest)
    return local_dest


def resolve_split_path(dataset_source: Optional[str], split: str = "train") -> Optional[str]:
    """
    Resolve a dataset reference to a local file path for the requested split.

    Handles:
      - "{datasetId}/{versionId}" → registry lookup → trainLocalRef / testLocalRef
      - "s3://..."                → download to data-storage/, return local path
      - bare versionId            → scan all datasets in registry
      - local path                → return as-is
    """
    if not dataset_source:
        return None

    train_key = "trainLocalRef" if split == "train" else "testLocalRef"
    s3_key    = "trainStorageRef" if split == "train" else "testStorageRef"

    def _from_version(ver: Dict[str, Any]) -> Optional[str]:
        local_ref = ver.get(train_key)
        if local_ref and Path(local_ref).exists():
            return local_ref
        s3_ref = ver.get(s3_key) or ver.get("storageRef", "")
        if s3_ref.startswith("s3://"):
            filename = s3_ref.split("/")[-1]
            local_dest = str(_DATA_STORAGE / filename)
            if Path(local_dest).exists():
                return local_dest
            return download_from_s3(s3_ref, local_dest)
        return None

    reg = _load_registry()

    # Format: "datasetId/versionId"
    if (
        "/" in dataset_source
        and not dataset_source.startswith("s3://")
        and not Path(dataset_source).exists()
    ):
        parts = dataset_source.split("/", 1)
        if len(parts) == 2:
            ds_id, ver_id = parts
            ver = reg.get("dataset_versions", {}).get(ds_id, {}).get(ver_id)
            if ver:
                result = _from_version(ver)
                if result:
                    return result

    # S3 URI
    if dataset_source.startswith("s3://"):
        filename = dataset_source.split("/")[-1]
        local_dest = str(_DATA_STORAGE / filename)
        if Path(local_dest).exists():
            return local_dest
        return download_from_s3(dataset_source, local_dest)

    # Existing local path
    if Path(dataset_source).exists():
        return dataset_source

    # Bare versionId: scan all datasets
    for ds_versions in reg.get("dataset_versions", {}).values():
        ver = ds_versions.get(dataset_source)
        if ver:
            result = _from_version(ver)
            if result:
                return result

    return dataset_source  # fallback


def list_s3_objects(prefix: str = "") -> List[Dict[str, Any]]:
    """List objects in the S3 bucket under the given prefix."""
    s3 = _s3_client()
    resp = s3.list_objects_v2(Bucket=settings.aws_s3_bucket, Prefix=prefix)
    return [
        {
            "key": obj["Key"],
            "size": obj["Size"],
            "lastModified": obj["LastModified"].isoformat(),
        }
        for obj in resp.get("Contents", [])
    ]


# ── DVC operations ───────────────────────────────────────────────────
def _run_dvc(args: List[str]) -> subprocess.CompletedProcess:
    """Run a DVC command from the DVC project root."""
    return subprocess.run(
        ["dvc"] + args,
        cwd=str(_DVC_ROOT),
        capture_output=True,
        text=True,
    )


def configure_dvc_s3_remote() -> Dict[str, Any]:
    """
    Add/overwrite an S3 DVC remote called 's3remote' pointing at
    s3://<bucket>/dvc and configure AWS credentials for it.
    """
    bucket = settings.aws_s3_bucket
    remote_url = f"s3://{bucket}/dvc"

    _run_dvc(["remote", "add", "--default", "-f", "s3remote", remote_url])

    if settings.aws_access_key_id:
        _run_dvc(["remote", "modify", "--local", "s3remote",
                  "access_key_id", settings.aws_access_key_id])
    if settings.aws_secret_access_key:
        _run_dvc(["remote", "modify", "--local", "s3remote",
                  "secret_access_key", settings.aws_secret_access_key])

    _run_dvc(["remote", "modify", "s3remote", "region", settings.aws_s3_region])
    logger.info("DVC remote configured → %s", remote_url)
    return {"remote": "s3remote", "url": remote_url}


def dvc_add_file(local_path: str) -> Dict[str, Any]:
    """
    Run `dvc add <file>` to track a file with DVC.
    Always resolves to an absolute path so DVC (which runs from _DVC_ROOT)
    can locate the file regardless of the server's working directory.
    """
    abs_path = str(Path(local_path).resolve())
    result = _run_dvc(["add", abs_path])
    if result.returncode != 0:
        logger.warning("dvc add failed: %s", result.stderr)
        return {"success": False, "error": result.stderr}

    dvc_file = abs_path + ".dvc"
    dvc_meta: Dict[str, Any] = {}
    if Path(dvc_file).exists():
        try:
            import yaml
            with open(dvc_file, encoding="utf-8") as fh:
                dvc_meta = yaml.safe_load(fh) or {}
        except Exception:
            pass

    return {"success": True, "dvcFile": dvc_file, "metadata": dvc_meta}


def dvc_push(target_dvc_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Push DVC-tracked files to the S3 remote.
    If target_dvc_file is given, push only that file's cache.
    """
    configure_dvc_s3_remote()
    args = ["push"]
    if target_dvc_file:
        # Resolve to absolute so _run_dvc can locate the .dvc file
        args.append(str(Path(target_dvc_file).resolve()))
    result = _run_dvc(args)
    success = result.returncode == 0
    if not success:
        logger.warning("dvc push failed: %s", result.stderr)
    return {"success": success, "output": result.stdout, "error": result.stderr}


def track_and_push(local_path: str) -> Dict[str, Any]:
    """Convenience: dvc add + dvc push a single local file."""
    abs_path = str(Path(local_path).resolve())
    add_result = dvc_add_file(abs_path)
    if not add_result["success"]:
        return add_result
    push_result = dvc_push(abs_path + ".dvc")
    return {**add_result, "push": push_result}


# ── Source registry ──────────────────────────────────────────────────
def create_source(
    name: str,
    type_: str,
    connection_ref: str,
    schema_ref: Optional[str],
    owner: Optional[str],
    refresh_cadence: Optional[str],
) -> Dict[str, Any]:
    reg = _load_registry()
    source_id = _short_id()
    record = {
        "sourceId": source_id,
        "name": name,
        "type": type_,
        "connectionRef": connection_ref,
        "schemaRef": schema_ref,
        "owner": owner,
        "refreshCadence": refresh_cadence,
        "status": "active",
        "createdAt": _now_iso(),
    }
    reg["sources"][source_id] = record
    _save_registry(reg)
    return record


def list_sources() -> List[Dict[str, Any]]:
    return list(_load_registry()["sources"].values())


def get_source(source_id: str) -> Optional[Dict[str, Any]]:
    return _load_registry()["sources"].get(source_id)


# ── Ingestion registry ───────────────────────────────────────────────
def create_ingestion(
    source_id: str,
    mode: str,
    window: Optional[str],
    params: Optional[Dict[str, Any]],
    ticket_id: Optional[str],
) -> Dict[str, Any]:
    reg = _load_registry()
    ingestion_id = _short_id()
    run_id = _short_id()
    source = reg["sources"].get(source_id, {})

    # If the caller passes a localFilePath, upload it to S3 immediately
    s3_ref: Optional[str] = None
    output_dataset_version_id: Optional[str] = None
    logs: List[str] = []

    local_file = (params or {}).get("localFilePath")
    if local_file and Path(local_file).exists():
        try:
            filename = Path(local_file).name
            s3_key = f"ingestions/{ingestion_id}/{filename}"
            s3_ref = upload_file_to_s3(local_file, s3_key)
            logs.append(f"Uploaded {local_file} → {s3_ref}")
            # Also track with DVC
            dvc_result = track_and_push(local_file)
            logs.append(f"DVC tracking: {dvc_result.get('success', False)}")
        except Exception as exc:
            logs.append(f"Upload error: {exc}")

    record = {
        "ingestionId": ingestion_id,
        "runId": run_id,
        "sourceId": source_id,
        "sourceName": source.get("name", ""),
        "connectionRef": source.get("connectionRef", ""),
        "mode": mode,
        "window": window,
        "params": params or {},
        "ticketId": ticket_id,
        "status": "completed" if s3_ref else "accepted",
        "progress": 100.0 if s3_ref else 0.0,
        "outputRefs": [s3_ref] if s3_ref else [],
        "outputDatasetVersionId": output_dataset_version_id,
        "s3Ref": s3_ref,
        "logs": logs,
        "createdAt": _now_iso(),
    }
    reg["ingestions"][ingestion_id] = record
    _save_registry(reg)
    return record


def get_ingestion(ingestion_id: str) -> Optional[Dict[str, Any]]:
    return _load_registry()["ingestions"].get(ingestion_id)


def list_ingestions() -> List[Dict[str, Any]]:
    return list(_load_registry()["ingestions"].values())


# ── Dataset registry ─────────────────────────────────────────────────
def create_dataset(
    name: str,
    owner: Optional[str],
    domain: Optional[str],
    description: Optional[str],
    schema_ref: Optional[str],
) -> Dict[str, Any]:
    reg = _load_registry()
    dataset_id = _short_id()
    record = {
        "datasetId": dataset_id,
        "name": name,
        "owner": owner,
        "domain": domain,
        "description": description,
        "schemaRef": schema_ref,
        "createdAt": _now_iso(),
    }
    reg["datasets"][dataset_id] = record
    reg["dataset_versions"][dataset_id] = {}
    _save_registry(reg)
    return record


def list_datasets() -> List[Dict[str, Any]]:
    return list(_load_registry()["datasets"].values())


def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    return _load_registry()["datasets"].get(dataset_id)


def update_dataset(dataset_id: str, **kwargs) -> Optional[Dict[str, Any]]:
    reg = _load_registry()
    if dataset_id not in reg["datasets"]:
        return None
    reg["datasets"][dataset_id].update(
        {k: v for k, v in kwargs.items() if v is not None}
    )
    _save_registry(reg)
    return reg["datasets"][dataset_id]


def create_dataset_version(
    dataset_id: str,
    storage_ref: str,
    schema_ref: Optional[str],
    lineage: Optional[Dict[str, Any]],
    stats: Optional[Dict[str, Any]],
    ticket_id: Optional[str],
) -> Dict[str, Any]:
    """
    Register a new dataset version.
    - If storage_ref is a local CSV → split 80/20, save train/test to data-storage/,
      upload both splits to S3, track with DVC.
    - If storage_ref is already an S3 URI → register as-is.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split as _tts

    reg = _load_registry()
    if dataset_id not in reg["datasets"]:
        raise ValueError(f"Dataset '{dataset_id}' not found")

    # Sequential version number within this dataset: v1, v2, v3 …
    existing = reg["dataset_versions"].get(dataset_id, {})
    version_num = len(existing) + 1
    version_id = f"v{version_num}"

    digest: Optional[str] = None
    s3_ref = storage_ref
    dvc_tracked = False
    train_s3_ref: Optional[str] = None
    test_s3_ref: Optional[str] = None
    train_local_ref: Optional[str] = None
    test_local_ref: Optional[str] = None

    local_path = Path(storage_ref)
    if local_path.exists():
        digest = compute_digest(storage_ref)

        # ── 80/20 train/test split ───────────────────────────────────
        stem = local_path.stem
        df = pd.read_csv(storage_ref)
        train_df, test_df = _tts(df, test_size=0.2, random_state=42)

        train_file = _DATA_STORAGE / f"{stem}_train.csv"
        test_file  = _DATA_STORAGE / f"{stem}_test.csv"
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        train_local_ref = str(train_file)
        test_local_ref  = str(test_file)
        logger.info("Split %s → %d train / %d test rows", stem, len(train_df), len(test_df))

        # ── Upload raw + splits to S3 ────────────────────────────────
        filename = local_path.name
        s3_ref = upload_file_to_s3(
            storage_ref,
            f"datasets/{dataset_id}/versions/{version_id}/{filename}",
        )
        train_s3_ref = upload_file_to_s3(
            str(train_file),
            f"datasets/{dataset_id}/versions/{version_id}/train/{train_file.name}",
        )
        test_s3_ref = upload_file_to_s3(
            str(test_file),
            f"datasets/{dataset_id}/versions/{version_id}/test/{test_file.name}",
        )

        # ── DVC track all three files ────────────────────────────────
        # Only track the raw file if it lives inside the DVC project;
        # temp files (e.g. from /data/preprocess) are outside the project
        # and must not be passed to `dvc add`.
        dvc_raw: Dict[str, Any] = {"success": True}
        try:
            local_path.resolve().relative_to(_DVC_ROOT)
            dvc_raw = track_and_push(str(local_path.resolve()))
        except ValueError:
            pass  # file is outside DVC project — already uploaded to S3, skip DVC

        dvc_train = track_and_push(str(train_file.resolve()))
        dvc_test  = track_and_push(str(test_file.resolve()))
        dvc_tracked = (
            dvc_raw.get("success", False)
            and dvc_train.get("success", False)
            and dvc_test.get("success", False)
        )

    version = {
        "versionId": version_id,
        "datasetId": dataset_id,
        "storageRef": s3_ref,
        "localRef": storage_ref,
        "trainStorageRef": train_s3_ref,
        "testStorageRef": test_s3_ref,
        "trainLocalRef": train_local_ref,
        "testLocalRef": test_local_ref,
        "digest": digest,
        "schemaRef": schema_ref,
        "lineage": lineage or {},
        "stats": stats or {},
        "ticketId": ticket_id,
        "dvcTracked": dvc_tracked,
        "createdBy": "api",
        "createdAt": _now_iso(),
    }

    if dataset_id not in reg["dataset_versions"]:
        reg["dataset_versions"][dataset_id] = {}
    reg["dataset_versions"][dataset_id][version_id] = version
    _save_registry(reg)
    return version


def list_dataset_versions(dataset_id: str) -> List[Dict[str, Any]]:
    return list(_load_registry()["dataset_versions"].get(dataset_id, {}).values())


def get_dataset_version(dataset_id: str, version_id: str) -> Optional[Dict[str, Any]]:
    reg = _load_registry()
    return reg["dataset_versions"].get(dataset_id, {}).get(version_id)


def get_dataset_lineage(
    dataset_id: str,
    from_version: Optional[str] = None,
    depth: int = 3,
) -> Dict[str, Any]:
    reg = _load_registry()
    dataset = reg["datasets"].get(dataset_id, {})
    versions = reg["dataset_versions"].get(dataset_id, {})

    nodes: List[Dict[str, Any]] = [
        {"id": dataset_id, "type": "dataset", "name": dataset.get("name", dataset_id)}
    ]
    edges: List[Dict[str, Any]] = []

    target_versions = versions
    if from_version and from_version in versions:
        target_versions = {from_version: versions[from_version]}

    for vid, ver in target_versions.items():
        nodes.append({
            "id": vid,
            "type": "version",
            "storageRef": ver.get("storageRef", ""),
            "digest": ver.get("digest"),
            "createdAt": ver.get("createdAt", ""),
        })
        edges.append({"from": dataset_id, "to": vid, "type": "has_version"})

        lin = ver.get("lineage", {})
        for parent in lin.get("parentVersions", []):
            edges.append({"from": parent, "to": vid, "type": "derived_from"})
        if lin.get("transformRef"):
            nodes.append({"id": lin["transformRef"], "type": "transform"})
            edges.append({"from": lin["transformRef"], "to": vid, "type": "applied_to"})

    return {"datasetId": dataset_id, "nodes": nodes, "edges": edges}


# ── Feature set registry ──────────────────────────────────────────────
def create_feature_set(
    name: str,
    owner: Optional[str],
    entity_schema: Optional[str],
    description: Optional[str],
) -> Dict[str, Any]:
    reg = _load_registry()
    fs_id = _short_id()
    record = {
        "featureSetId": fs_id,
        "name": name,
        "owner": owner,
        "entitySchema": entity_schema,
        "description": description,
        "createdAt": _now_iso(),
    }
    reg["features"][fs_id] = record
    reg["feature_versions"][fs_id] = {}
    _save_registry(reg)
    return record


def list_feature_sets() -> List[Dict[str, Any]]:
    return list(_load_registry()["features"].values())


def get_feature_set(fs_id: str) -> Optional[Dict[str, Any]]:
    return _load_registry()["features"].get(fs_id)


def update_feature_set(fs_id: str, **kwargs) -> Optional[Dict[str, Any]]:
    reg = _load_registry()
    if fs_id not in reg["features"]:
        return None
    reg["features"][fs_id].update({k: v for k, v in kwargs.items() if v is not None})
    _save_registry(reg)
    return reg["features"][fs_id]


def create_feature_version(
    fs_id: str,
    storage_ref: str,
    schema_ref: Optional[str],
    computed_from: Optional[str],
    ticket_id: Optional[str],
) -> Dict[str, Any]:
    reg = _load_registry()
    if fs_id not in reg["features"]:
        raise ValueError(f"Feature set '{fs_id}' not found")

    # Sequential version number within this feature set: fv1, fv2, fv3 …
    existing = reg["feature_versions"].get(fs_id, {})
    version_num = len(existing) + 1
    version_id = f"fv{version_num}"

    digest: Optional[str] = None
    s3_ref = storage_ref
    dvc_tracked = False

    local_path = Path(storage_ref)
    if local_path.exists():
        digest = compute_digest(storage_ref)
        filename = local_path.name
        s3_key = f"features/{fs_id}/versions/{version_id}/{filename}"
        s3_ref = upload_file_to_s3(storage_ref, s3_key)

        dvc_result = track_and_push(str(local_path.resolve()))
        dvc_tracked = dvc_result.get("success", False)

    version = {
        "versionId": version_id,
        "featureSetId": fs_id,
        "storageRef": s3_ref,
        "schemaRef": schema_ref,
        "computedFrom": computed_from,
        "digest": digest,
        "ticketId": ticket_id,
        "dvcTracked": dvc_tracked,
        "createdAt": _now_iso(),
    }

    if fs_id not in reg["feature_versions"]:
        reg["feature_versions"][fs_id] = {}
    reg["feature_versions"][fs_id][version_id] = version
    _save_registry(reg)
    return version


def list_feature_versions(fs_id: str) -> List[Dict[str, Any]]:
    return list(_load_registry()["feature_versions"].get(fs_id, {}).values())


def get_feature_version(fs_id: str, version_id: str) -> Optional[Dict[str, Any]]:
    return _load_registry()["feature_versions"].get(fs_id, {}).get(version_id)
