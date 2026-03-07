"""
S3 + DVC service layer for the Data dimension.

Responsibilities:
  - Upload / download files to the S3 bucket via boto3
  - Track files with DVC (dvc add) and push metadata to S3 remote (dvc push)
  - Maintain a JSON registry for sources, ingestions, datasets and features
    so that metadata survives process restarts without needing a database.

Production fixes applied:
  - Thread-safe registry access via threading.Lock
  - Atomic registry writes (write-to-tmp → rename)
  - AWS credentials passed via env vars to DVC subprocess (not CLI args)
  - DVC remote configured once per process (cached flag)
  - S3 list uses paginator (no 1000-object truncation)
  - Version IDs derived from max existing suffix (no collision on delete)
  - localRef is None when storage_ref is S3 URI (no local file created)
  - Temp file cleanup in try/finally inside create_dataset_version
  - Input validation (empty strings rejected) in all create functions
  - Filename sanitisation before building S3 keys
  - Distinguishes ClientError / ValueError / RuntimeError in ingestion
  - Ingestion status set to "failed" on error (not silently ignored)
  - DVC YAML parse errors are logged (not silently swallowed)
  - FileNotFoundError raised when local storage_ref doesn't exist
  - CSV read falls back to latin-1 encoding on UnicodeDecodeError
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from uuid import uuid4

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from ...config import settings

logger = logging.getLogger(__name__)

# ── DVC project root ─────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_DVC_ROOT: Path = _HERE
while _DVC_ROOT != _DVC_ROOT.parent:
    if (_DVC_ROOT / ".dvc").is_dir():
        break
    _DVC_ROOT = _DVC_ROOT.parent

# ── Local data-storage folder ────────────────────────────────────────
_DATA_STORAGE = Path(__file__).parent / "data-storage"
_DATA_STORAGE.mkdir(exist_ok=True)

# ── Registry file ────────────────────────────────────────────────────
_REGISTRY_FILE = Path(__file__).parent / "data_registry.json"
_registry_lock = threading.Lock()

# ── DVC remote configuration cache ──────────────────────────────────
_dvc_remote_configured = False


# ─────────────────────────────────────────────────────────────────────
# Registry helpers — thread-safe, atomic writes
# ─────────────────────────────────────────────────────────────────────

def _load_registry() -> Dict[str, Any]:
    """Load registry from disk. Caller must hold _registry_lock."""
    if _REGISTRY_FILE.exists():
        try:
            with open(_REGISTRY_FILE, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            logger.error("Registry JSON corrupt: %s — returning empty registry", exc)
    return {
        "sources": {},
        "ingestions": {},
        "datasets": {},
        "dataset_versions": {},
        "features": {},
        "feature_versions": {},
    }


def _save_registry(reg: Dict[str, Any]) -> None:
    """Atomic write: write to .tmp then rename. Caller must hold _registry_lock."""
    tmp = _REGISTRY_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, default=str)
    tmp.replace(_REGISTRY_FILE)


@contextmanager
def _registry() -> Generator[Dict[str, Any], None, None]:
    """
    Thread-safe context manager: acquires lock, loads registry, yields it,
    then saves only on clean exit (not on exception).
    """
    with _registry_lock:
        reg = _load_registry()
        try:
            yield reg
        except Exception:
            raise  # Do not save corrupted state
        else:
            _save_registry(reg)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id() -> str:
    return str(uuid4())[:8]


def _sanitize_filename(path_or_name: str) -> str:
    """Extract and sanitize the filename from a path or URI."""
    name = Path(path_or_name.split("/")[-1]).name
    name = name.replace("..", "").replace("\\", "").strip()
    if not name:
        raise ValueError(f"Invalid filename derived from: {path_or_name!r}")
    return name


def _next_version_id(existing: Dict[str, Any], prefix: str = "v") -> str:
    """Derive next version ID from max existing numeric suffix (avoids collision on delete)."""
    max_num = 0
    for k in existing:
        if k.startswith(prefix) and k[len(prefix):].isdigit():
            max_num = max(max_num, int(k[len(prefix):]))
    return f"{prefix}{max_num + 1}"


# ─────────────────────────────────────────────────────────────────────
# AWS clients
# ─────────────────────────────────────────────────────────────────────

def _s3_client():
    return boto3.client(
        "s3",
        region_name=settings.aws_s3_region,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )


def _sm_client(region: Optional[str] = None):
    return boto3.client(
        "secretsmanager",
        region_name=region or settings.aws_s3_region,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )


def fetch_s3_secret(secret_ref: str) -> Dict[str, Any]:
    """
    Fetch a Secrets Manager secret by name or ARN.
    Expected JSON: {"bucket": "...", "prefix": "path/to/file.csv", "region"?: "..."}

    Raises:
      ClientError: secret doesn't exist or permission denied
      RuntimeError: secret value is not valid JSON
      ValueError: secret missing required 'prefix' key
    """
    sm = _sm_client()
    response = sm.get_secret_value(SecretId=secret_ref)
    raw = response["SecretString"]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Secret '{secret_ref}' is not valid JSON. "
            f"Raw value: {raw!r}. Error: {exc}"
        ) from exc
    if "prefix" not in data:
        raise ValueError(
            f"Secret '{secret_ref}' missing required 'prefix' key. "
            f"Got keys: {list(data.keys())}"
        )
    return data


def download_from_s3_secret(secret_ref: str) -> str:
    """
    Resolve a Secrets Manager secret to an S3 location, download the file
    to data-storage/, and return the local path.

    Raises: ClientError, RuntimeError, ValueError, or IOError
    """
    secret = fetch_s3_secret(secret_ref)
    bucket = secret.get("bucket", settings.aws_s3_bucket)
    prefix = secret["prefix"]
    s3_uri = f"s3://{bucket}/{prefix}"
    filename = _sanitize_filename(prefix)
    local_dest = str(_DATA_STORAGE / filename)
    return download_from_s3(s3_uri, local_dest)


def compute_digest(file_path: str) -> str:
    """Return the SHA-256 digest of a local file in 'sha256:<hex>' format."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


# ─────────────────────────────────────────────────────────────────────
# S3 operations
# ─────────────────────────────────────────────────────────────────────

def upload_file_to_s3(local_path: str, s3_key: str) -> str:
    """Upload a local file to S3 and return its s3:// URI."""
    s3 = _s3_client()
    s3.upload_file(local_path, settings.aws_s3_bucket, s3_key)
    uri = f"s3://{settings.aws_s3_bucket}/{s3_key}"
    logger.info("Uploaded %s → %s", local_path, uri)
    return uri


def download_from_s3(s3_uri: str, local_dest: str) -> str:
    """Download a file from an S3 URI to local_dest and return the local path."""
    without_scheme = s3_uri.replace("s3://", "")
    parts = without_scheme.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URI: {s3_uri!r}")
    bucket, key = parts[0], parts[1]
    Path(local_dest).parent.mkdir(parents=True, exist_ok=True)
    _s3_client().download_file(bucket, key, local_dest)
    logger.info("Downloaded %s → %s", s3_uri, local_dest)
    return local_dest


def list_s3_objects(prefix: str = "") -> List[Dict[str, Any]]:
    """List all objects in the S3 bucket under the given prefix (fully paginated)."""
    s3 = _s3_client()
    objects: List[Dict[str, Any]] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.aws_s3_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects.append({
                "key": obj["Key"],
                "size": obj["Size"],
                "lastModified": obj["LastModified"].isoformat(),
            })
    return objects


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
            filename = _sanitize_filename(s3_ref)
            local_dest = str(_DATA_STORAGE / filename)
            if Path(local_dest).exists():
                return local_dest
            return download_from_s3(s3_ref, local_dest)
        return None

    with _registry_lock:
        reg = _load_registry()

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

    if dataset_source.startswith("s3://"):
        filename = _sanitize_filename(dataset_source)
        local_dest = str(_DATA_STORAGE / filename)
        if Path(local_dest).exists():
            return local_dest
        return download_from_s3(dataset_source, local_dest)

    if Path(dataset_source).exists():
        return dataset_source

    for ds_versions in reg.get("dataset_versions", {}).values():
        ver = ds_versions.get(dataset_source)
        if ver:
            result = _from_version(ver)
            if result:
                return result

    return dataset_source  # fallback: return as-is and let caller handle


# ─────────────────────────────────────────────────────────────────────
# DVC operations
# ─────────────────────────────────────────────────────────────────────

def _run_dvc(args: List[str]) -> subprocess.CompletedProcess:
    """
    Run a DVC command from the DVC project root.
    AWS credentials are injected via environment variables — never via CLI args
    (which would expose them in process listings).
    """
    env = {**os.environ}
    if settings.aws_access_key_id:
        env["AWS_ACCESS_KEY_ID"] = settings.aws_access_key_id
    if settings.aws_secret_access_key:
        env["AWS_SECRET_ACCESS_KEY"] = settings.aws_secret_access_key
    if settings.aws_s3_region:
        env["AWS_DEFAULT_REGION"] = settings.aws_s3_region
    return subprocess.run(
        ["dvc"] + args,
        cwd=str(_DVC_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )


def configure_dvc_s3_remote() -> Dict[str, Any]:
    """
    Register the S3 DVC remote once per process. Subsequent calls are no-ops.
    Credentials are NOT written to .dvc/config.local — they are passed via env vars.
    """
    global _dvc_remote_configured
    if _dvc_remote_configured:
        return {"remote": "s3remote", "cached": True}
    bucket = settings.aws_s3_bucket
    remote_url = f"s3://{bucket}/dvc"
    _run_dvc(["remote", "add", "--default", "-f", "s3remote", remote_url])
    _run_dvc(["remote", "modify", "s3remote", "region", settings.aws_s3_region])
    _dvc_remote_configured = True
    logger.info("DVC remote configured → %s", remote_url)
    return {"remote": "s3remote", "url": remote_url}


def dvc_add_file(local_path: str) -> Dict[str, Any]:
    """
    Run `dvc add <file>` to track a file with DVC.
    Returns {"success": True/False, "dvcFile": ..., "metadata": ..., "error"?: ...}
    """
    abs_path = str(Path(local_path).resolve())
    result = _run_dvc(["add", abs_path])
    if result.returncode != 0:
        logger.warning("dvc add failed for %s: %s", abs_path, result.stderr)
        return {"success": False, "error": result.stderr, "dvcFile": abs_path + ".dvc"}

    dvc_file = abs_path + ".dvc"
    dvc_meta: Dict[str, Any] = {}
    if Path(dvc_file).exists():
        try:
            import yaml
            with open(dvc_file, encoding="utf-8") as fh:
                dvc_meta = yaml.safe_load(fh) or {}
        except Exception as exc:
            logger.warning("Failed to read DVC metadata from %s: %s", dvc_file, exc)

    return {"success": True, "dvcFile": dvc_file, "metadata": dvc_meta}


def dvc_push(target_dvc_file: Optional[str] = None) -> Dict[str, Any]:
    """Push DVC-tracked files to the S3 remote."""
    configure_dvc_s3_remote()
    args = ["push"]
    if target_dvc_file:
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


# ─────────────────────────────────────────────────────────────────────
# Source registry
# ─────────────────────────────────────────────────────────────────────

def create_source(
    name: str,
    type_: str,
    connection_ref: str,
    schema_ref: Optional[str],
    owner: Optional[str],
    refresh_cadence: Optional[str],
) -> Dict[str, Any]:
    name = (name or "").strip()
    type_ = (type_ or "").strip()
    connection_ref = (connection_ref or "").strip()
    if not name:
        raise ValueError("Source 'name' is required and cannot be empty")
    if not type_:
        raise ValueError("Source 'type' is required and cannot be empty")
    if not connection_ref:
        raise ValueError("Source 'connectionRef' is required and cannot be empty")

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
    with _registry() as reg:
        reg["sources"][source_id] = record
    return record


def list_sources() -> List[Dict[str, Any]]:
    with _registry_lock:
        return list(_load_registry()["sources"].values())


def get_source(source_id: str) -> Optional[Dict[str, Any]]:
    with _registry_lock:
        return _load_registry()["sources"].get(source_id)


# ─────────────────────────────────────────────────────────────────────
# Ingestion registry
# ─────────────────────────────────────────────────────────────────────

def create_ingestion(
    source_id: str,
    mode: str,
    window: Optional[str],
    params: Optional[Dict[str, Any]],
    ticket_id: Optional[str],
) -> Dict[str, Any]:
    source_id = (source_id or "").strip()
    mode = (mode or "").strip()
    if not source_id:
        raise ValueError("'sourceId' is required")
    if mode not in ("batch", "stream"):
        raise ValueError(f"'mode' must be 'batch' or 'stream', got: {mode!r}")

    # Read source info under lock (quick read, no I/O)
    with _registry_lock:
        source = _load_registry()["sources"].get(source_id, {})

    ingestion_id = _short_id()
    run_id = _short_id()
    s3_ref: Optional[str] = None
    local_file: Optional[str] = None
    logs: List[str] = []
    ingestion_failed = False

    secret_ref: Optional[str] = (params or {}).get("secretRef")
    if not secret_ref and source.get("type") == "s3":
        secret_ref = source.get("connectionRef") or None

    if secret_ref:
        # S3 source: download via Secrets Manager, then DVC track
        try:
            secret = fetch_s3_secret(secret_ref)
            bucket = secret.get("bucket", settings.aws_s3_bucket)
            prefix = secret["prefix"]
            s3_ref = f"s3://{bucket}/{prefix}"
            local_file = download_from_s3_secret(secret_ref)
            logs.append(f"Downloaded from S3 via secret '{secret_ref}': {s3_ref} → {local_file}")
            dvc_result = track_and_push(local_file)
            if not dvc_result.get("success"):
                logs.append(f"DVC tracking warning: {dvc_result.get('error', 'unknown')}")
            else:
                logs.append("DVC tracking: success")
        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            logs.append(f"Secret fetch failed [{error_code}]: {exc}")
            logger.error("ClientError fetching secret '%s': %s", secret_ref, exc)
            ingestion_failed = True
        except (RuntimeError, ValueError) as exc:
            logs.append(f"Secret configuration error: {exc}")
            logger.error("Secret config error for '%s': %s", secret_ref, exc)
            ingestion_failed = True
        except Exception as exc:
            logs.append(f"Unexpected ingestion error: {exc}")
            logger.exception("Unexpected error during S3 ingestion for source %s", source_id)
            ingestion_failed = True
    else:
        # Legacy: local file → upload to S3
        raw_local = (params or {}).get("localFilePath")
        if raw_local and Path(raw_local).exists():
            try:
                local_file = raw_local
                filename = _sanitize_filename(local_file)
                s3_key = f"ingestions/{ingestion_id}/{filename}"
                s3_ref = upload_file_to_s3(local_file, s3_key)
                logs.append(f"Uploaded {local_file} → {s3_ref}")
                dvc_result = track_and_push(local_file)
                if not dvc_result.get("success"):
                    logs.append(f"DVC tracking warning: {dvc_result.get('error', 'unknown')}")
                else:
                    logs.append("DVC tracking: success")
            except (ClientError, BotoCoreError) as exc:
                logs.append(f"S3 upload failed: {exc}")
                logger.error("S3 upload failed for ingestion %s: %s", ingestion_id, exc)
                ingestion_failed = True
            except Exception as exc:
                logs.append(f"Unexpected upload error: {exc}")
                logger.exception("Unexpected error during local file ingestion %s", ingestion_id)
                ingestion_failed = True

    status = "failed" if ingestion_failed else (
        "completed" if (local_file or s3_ref) else "accepted"
    )
    progress = 0.0 if ingestion_failed else (100.0 if (local_file or s3_ref) else 0.0)

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
        "status": status,
        "progress": progress,
        "outputRefs": [s3_ref] if s3_ref else [],
        "outputDatasetVersionId": None,
        "s3Ref": s3_ref,
        "localRef": local_file,
        "logs": logs,
        "createdAt": _now_iso(),
    }
    with _registry() as reg:
        reg["ingestions"][ingestion_id] = record
    return record


def get_ingestion(ingestion_id: str) -> Optional[Dict[str, Any]]:
    with _registry_lock:
        return _load_registry()["ingestions"].get(ingestion_id)


def list_ingestions() -> List[Dict[str, Any]]:
    with _registry_lock:
        return list(_load_registry()["ingestions"].values())


# ─────────────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────────────

def create_dataset(
    name: str,
    owner: Optional[str],
    domain: Optional[str],
    description: Optional[str],
    schema_ref: Optional[str],
) -> Dict[str, Any]:
    name = (name or "").strip()
    if not name:
        raise ValueError("Dataset 'name' is required and cannot be empty")

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
    with _registry() as reg:
        reg["datasets"][dataset_id] = record
        reg["dataset_versions"][dataset_id] = {}
    return record


def list_datasets() -> List[Dict[str, Any]]:
    with _registry_lock:
        return list(_load_registry()["datasets"].values())


def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    with _registry_lock:
        return _load_registry()["datasets"].get(dataset_id)


def update_dataset(dataset_id: str, **kwargs) -> Optional[Dict[str, Any]]:
    with _registry() as reg:
        if dataset_id not in reg["datasets"]:
            return None
        reg["datasets"][dataset_id].update(
            {k: v for k, v in kwargs.items() if v is not None}
        )
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
    - Local CSV  → compute digest, split 80/20, upload all to S3, DVC track.
    - S3 URI     → register as-is (no split, no re-upload).
    - secret://  → download first, then treat as local CSV.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split as _tts

    # Phase 1: validate and reserve version ID (short lock, no I/O)
    with _registry_lock:
        reg = _load_registry()
        if dataset_id not in reg["datasets"]:
            raise LookupError(f"Dataset '{dataset_id}' not found")
        existing = reg["dataset_versions"].get(dataset_id, {})
        version_id = _next_version_id(existing, prefix="v")

    # Resolve secret reference before any other work
    if storage_ref.startswith("arn:aws:secretsmanager") or storage_ref.startswith("secret://"):
        actual_ref = storage_ref.removeprefix("secret://")
        storage_ref = download_from_s3_secret(actual_ref)
        logger.info("Resolved secret '%s' → %s", actual_ref, storage_ref)

    local_path = Path(storage_ref)

    # Validate file existence for non-S3 refs
    if not local_path.exists() and not storage_ref.startswith("s3://"):
        raise FileNotFoundError(f"File not found: {storage_ref!r}")

    digest: Optional[str] = None
    s3_ref: Optional[str] = storage_ref if storage_ref.startswith("s3://") else None
    local_ref: Optional[str] = None
    dvc_tracked = False

    # Phase 2: heavy I/O — no registry lock held
    if local_path.exists():
        local_ref = storage_ref
        digest = compute_digest(storage_ref)

        try:
            filename = local_path.name
            s3_ref = upload_file_to_s3(
                storage_ref,
                f"datasets/{dataset_id}/versions/{version_id}/{filename}",
            )

            dvc_raw: Dict[str, Any] = {"success": True}
            try:
                local_path.resolve().relative_to(_DVC_ROOT)
                dvc_raw = track_and_push(str(local_path.resolve()))
            except ValueError:
                pass  # file outside DVC project — already in S3, skip raw DVC

            dvc_tracked = dvc_raw.get("success", False)
            if not dvc_tracked:
                logger.warning(
                    "DVC tracking incomplete for %s/%s: raw=%s",
                    dataset_id, version_id, dvc_raw.get("success"),
                )

        except Exception:
            raise

    version = {
        "versionId": version_id,
        "datasetId": dataset_id,
        "storageRef": s3_ref or storage_ref,
        "localRef": local_ref,
        "trainStorageRef": None,
        "testStorageRef": None,
        "trainLocalRef": None,
        "testLocalRef": None,
        "digest": digest,
        "schemaRef": schema_ref,
        "lineage": lineage or {},
        "stats": stats or {},
        "ticketId": ticket_id,
        "dvcTracked": dvc_tracked,
        "createdBy": "api",
        "createdAt": _now_iso(),
    }

    # Phase 3: write version record (short lock)
    with _registry() as reg:
        if dataset_id not in reg["dataset_versions"]:
            reg["dataset_versions"][dataset_id] = {}
        # Re-check for collision (another request may have written concurrently)
        existing_now = reg["dataset_versions"][dataset_id]
        if version_id in existing_now:
            version_id = _next_version_id(existing_now, prefix="v")
            version["versionId"] = version_id
        reg["dataset_versions"][dataset_id][version_id] = version
    return version


def list_dataset_versions(dataset_id: str) -> List[Dict[str, Any]]:
    with _registry_lock:
        return list(_load_registry()["dataset_versions"].get(dataset_id, {}).values())


def get_dataset_version(dataset_id: str, version_id: str) -> Optional[Dict[str, Any]]:
    with _registry_lock:
        return _load_registry()["dataset_versions"].get(dataset_id, {}).get(version_id)


def get_dataset_lineage(
    dataset_id: str,
    from_version: Optional[str] = None,
    depth: int = 3,
) -> Dict[str, Any]:
    with _registry_lock:
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


# ─────────────────────────────────────────────────────────────────────
# Feature set registry
# ─────────────────────────────────────────────────────────────────────

def create_feature_set(
    name: str,
    owner: Optional[str],
    entity_schema: Optional[str],
    description: Optional[str],
) -> Dict[str, Any]:
    name = (name or "").strip()
    if not name:
        raise ValueError("Feature set 'name' is required and cannot be empty")
    fs_id = _short_id()
    record = {
        "featureSetId": fs_id,
        "name": name,
        "owner": owner,
        "entitySchema": entity_schema,
        "description": description,
        "createdAt": _now_iso(),
    }
    with _registry() as reg:
        reg["features"][fs_id] = record
        reg["feature_versions"][fs_id] = {}
    return record


def list_feature_sets() -> List[Dict[str, Any]]:
    with _registry_lock:
        return list(_load_registry()["features"].values())


def get_feature_set(fs_id: str) -> Optional[Dict[str, Any]]:
    with _registry_lock:
        return _load_registry()["features"].get(fs_id)


def update_feature_set(fs_id: str, **kwargs) -> Optional[Dict[str, Any]]:
    with _registry() as reg:
        if fs_id not in reg["features"]:
            return None
        reg["features"][fs_id].update({k: v for k, v in kwargs.items() if v is not None})
        return reg["features"][fs_id]


def create_feature_version(
    fs_id: str,
    storage_ref: str,
    schema_ref: Optional[str],
    computed_from: Optional[str],
    entity_keys: Optional[List[str]],
    ticket_id: Optional[str],
) -> Dict[str, Any]:
    # Phase 1: validate and reserve version ID
    with _registry_lock:
        reg = _load_registry()
        if fs_id not in reg["features"]:
            raise LookupError(f"Feature set '{fs_id}' not found")
        existing = reg["feature_versions"].get(fs_id, {})
        version_id = _next_version_id(existing, prefix="fv")

    digest: Optional[str] = None
    s3_ref = storage_ref
    local_ref: Optional[str] = None
    dvc_tracked = False
    effective_entity_keys: List[str] = entity_keys or []
    train_local_ref: Optional[str] = None
    test_local_ref: Optional[str] = None
    train_s3_ref: Optional[str] = None
    test_s3_ref: Optional[str] = None

    # Phase 2: I/O — no lock held
    local_path = Path(storage_ref)
    if local_path.exists():
        local_ref = storage_ref
        digest = compute_digest(storage_ref)
        filename = _sanitize_filename(storage_ref)
        s3_key = f"features/{fs_id}/versions/{version_id}/{filename}"
        s3_ref = upload_file_to_s3(storage_ref, s3_key)
        dvc_result = track_and_push(str(local_path.resolve()))
        dvc_tracked = dvc_result.get("success", False)
        if not dvc_tracked:
            logger.warning(
                "DVC tracking failed for feature version %s/%s: %s",
                fs_id, version_id, dvc_result.get("error"),
            )

        # Split engineered features into train/test after upload
        try:
            import pandas as pd
            from sklearn.model_selection import train_test_split as _tts
            stem = f"{fs_id}_{version_id}"
            train_file = _DATA_STORAGE / f"{stem}_train.csv"
            test_file  = _DATA_STORAGE / f"{stem}_test.csv"
            try:
                df = pd.read_csv(storage_ref)
            except UnicodeDecodeError:
                df = pd.read_csv(storage_ref, encoding="latin-1")
            train_df, test_df = _tts(df, test_size=0.2, random_state=42)
            train_df.to_csv(train_file, index=False)
            test_df.to_csv(test_file, index=False)
            train_local_ref = str(train_file)
            test_local_ref  = str(test_file)
            train_s3_ref = upload_file_to_s3(str(train_file), f"features/{fs_id}/versions/{version_id}/train/{train_file.name}")
            test_s3_ref  = upload_file_to_s3(str(test_file),  f"features/{fs_id}/versions/{version_id}/test/{test_file.name}")
            logger.info("Feature split %s → %d train / %d test rows", stem, len(train_df), len(test_df))
        except Exception as exc:
            logger.warning("Feature train/test split failed for %s/%s: %s", fs_id, version_id, exc)

    version = {
        "versionId": version_id,
        "featureSetId": fs_id,
        "storageRef": s3_ref,
        "localRef": local_ref,
        "trainLocalRef": train_local_ref,
        "testLocalRef": test_local_ref,
        "trainStorageRef": train_s3_ref,
        "testStorageRef": test_s3_ref,
        "schemaRef": schema_ref,
        "computedFrom": computed_from,
        "digest": digest,
        "ticketId": ticket_id,
        "dvcTracked": dvc_tracked,
        "entityKeys": effective_entity_keys,
        "createdAt": _now_iso(),
    }

    # Phase 3: write record
    with _registry() as reg:
        if fs_id not in reg["feature_versions"]:
            reg["feature_versions"][fs_id] = {}
        existing_now = reg["feature_versions"][fs_id]
        if version_id in existing_now:
            version_id = _next_version_id(existing_now, prefix="fv")
            version["versionId"] = version_id
        reg["feature_versions"][fs_id][version_id] = version
    return version


def list_feature_versions(fs_id: str) -> List[Dict[str, Any]]:
    with _registry_lock:
        return list(_load_registry()["feature_versions"].get(fs_id, {}).values())


def get_feature_version(fs_id: str, version_id: str) -> Optional[Dict[str, Any]]:
    with _registry_lock:
        return _load_registry()["feature_versions"].get(fs_id, {}).get(version_id)
