"""
Redis online feature store — MLOps

Responsibilities:
  - Materialize a CSV feature table into Redis (one hash per row, keyed by entity values)
  - Serve feature vectors at serving time via O(1) Redis HGETALL
  - Gracefully degrade: callers get None when Redis is unavailable

Key schema:
  fsv:{featureSetId}:{versionId}:{col1=val1}&{col2=val2}  →  Redis Hash
                                                               field=feature_name
                                                               value=feature_value (string)

Example:
  fsv:fs-abc123:fv1:patient_id=42  →  { diagnosis: "M", radius_mean: "17.99", ... }
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ...config import settings

logger = logging.getLogger(__name__)

# ── Lazy singleton Redis client ───────────────────────────────────────
_redis_client = None


def get_redis_client():
    """Return a Redis client, creating it on first call."""
    global _redis_client
    if _redis_client is None:
        import redis as _redis
        _redis_client = _redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            db=settings.redis_db,
            decode_responses=True,   # all values come back as str
            socket_connect_timeout=0.5,  # fail fast when Redis is down
            socket_timeout=2,
        )
    return _redis_client


def is_redis_available() -> bool:
    """Return True if Redis can be reached."""
    try:
        return bool(get_redis_client().ping())
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────
# Key helpers
# ─────────────────────────────────────────────────────────────────────

def _build_key(fs_id: str, version_id: str, entity_kv: Dict[str, Any]) -> str:
    """
    Build a deterministic Redis key from entity key-value pairs.
    Pairs are sorted so the key is the same regardless of input ordering.

    Example: fsv:fs-abc:fv1:diagnosis=M&patient_id=42
    """
    sorted_pairs = "&".join(f"{k}={v}" for k, v in sorted(entity_kv.items()))
    return f"fsv:{fs_id}:{version_id}:{sorted_pairs}"


def _scan_key_pattern(fs_id: str, version_id: str) -> str:
    """Glob pattern to list all keys for a given feature set version."""
    return f"fsv:{fs_id}:{version_id}:*"


# ─────────────────────────────────────────────────────────────────────
# Write — materialize CSV → Redis
# ─────────────────────────────────────────────────────────────────────

def materialize_to_redis(
    csv_path: str,
    fs_id: str,
    version_id: str,
    entity_keys: List[str],
) -> Dict[str, Any]:
    """
    Load a CSV feature table and push every row into Redis as a hash.

    Each row becomes one Redis key keyed by its entity column values.
    Rows are written in batches of 500 via a pipeline to minimise RTTs.

    Args:
        csv_path:    Local path to the CSV feature table.
        fs_id:       Feature set ID (used in the Redis key namespace).
        version_id:  Version ID (used in the Redis key namespace).
        entity_keys: Column names that uniquely identify a row (e.g. ["patient_id"]).

    Returns:
        {"materialized": N, "errors": M, "skipped_missing_keys": K}

    Raises:
        ValueError  if entity_keys columns are absent from the CSV.
        FileNotFoundError if csv_path does not exist.
    """
    import pandas as pd

    client = get_redis_client()
    ttl = settings.redis_feature_ttl

    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")

    missing_cols = [k for k in entity_keys if k not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Entity key columns {missing_cols} not found in CSV columns: {list(df.columns)}."
        )

    materialized = 0
    errors = 0
    skipped = 0
    pipe = client.pipeline(transaction=False)
    batch_size = 0

    for _, row in df.iterrows():
        try:
            # Skip rows where any entity key is null
            if any(pd.isna(row.get(k)) for k in entity_keys):
                skipped += 1
                continue

            entity_kv = {k: str(row[k]) for k in entity_keys}
            redis_key = _build_key(fs_id, version_id, entity_kv)

            # Convert all feature values to strings (Redis hash values must be strings)
            row_dict = {
                col: "" if pd.isna(val) else str(val)
                for col, val in row.to_dict().items()
            }
            pipe.hset(redis_key, mapping=row_dict)
            if ttl > 0:
                pipe.expire(redis_key, ttl)

            materialized += 1
            batch_size += 1

            if batch_size >= 500:
                pipe.execute()
                pipe = client.pipeline(transaction=False)
                batch_size = 0

        except Exception as exc:
            logger.warning("Row materialization error (fs=%s v=%s): %s", fs_id, version_id, exc)
            errors += 1

    # Flush remaining commands
    if batch_size > 0:
        pipe.execute()

    logger.info(
        "Redis materialization complete: fs=%s v=%s materialized=%d errors=%d skipped=%d",
        fs_id, version_id, materialized, errors, skipped,
    )
    return {"materialized": materialized, "errors": errors, "skipped_missing_keys": skipped}


# ─────────────────────────────────────────────────────────────────────
# Read — serve features from Redis
# ─────────────────────────────────────────────────────────────────────

def get_features(
    fs_id: str,
    version_id: str,
    entity_kv: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a feature vector from Redis.

    Returns:
        A dict of feature_name → value (all strings), or None if not found
        or Redis is unavailable.
    """
    client = get_redis_client()
    redis_key = _build_key(fs_id, version_id, entity_kv)
    try:
        result = client.hgetall(redis_key)
        return result if result else None
    except Exception as exc:
        logger.warning("Redis HGETALL failed (key=%s): %s", redis_key, exc)
        return None


def delete_version(fs_id: str, version_id: str) -> int:
    """
    Delete all Redis keys for a specific feature set version.
    Returns the number of keys deleted.
    """
    client = get_redis_client()
    pattern = _scan_key_pattern(fs_id, version_id)
    deleted = 0
    try:
        cursor = 0
        while True:
            cursor, keys = client.scan(cursor, match=pattern, count=200)
            if keys:
                client.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
    except Exception as exc:
        logger.warning("Redis delete_version failed (fs=%s v=%s): %s", fs_id, version_id, exc)
    return deleted


# ─────────────────────────────────────────────────────────────────────
# Write — store a single entity's features (write-on-first-visit)
# ─────────────────────────────────────────────────────────────────────

def store_features(
    fs_id: str,
    version_id: str,
    entity_kv: Dict[str, Any],
    features: Dict[str, Any],
    overwrite: bool = True,
) -> bool:
    """
    Store a single entity's feature vector in Redis.

    Typically called after a model prediction for a new entity so that on
    subsequent visits the features can be retrieved without recomputing them.

    Args:
        fs_id:       Feature set ID.
        version_id:  Feature version ID.
        entity_kv:   Entity key-value pairs that identify this entity (e.g. {"patient_id": "42"}).
        features:    Feature name → value mapping to store.
        overwrite:   If False, skip when a key already exists (HSETNX semantics via exists check).

    Returns:
        True if the key was written, False if skipped (already exists and overwrite=False)
        or if Redis is unavailable.
    """
    try:
        client = get_redis_client()
        redis_key = _build_key(fs_id, version_id, entity_kv)

        if not overwrite and client.exists(redis_key):
            return False

        mapping = {k: "" if v is None else str(v) for k, v in features.items()}
        # Also store entity key columns so the full row is self-contained
        for k, v in entity_kv.items():
            mapping.setdefault(k, str(v))

        client.hset(redis_key, mapping=mapping)
        ttl = settings.redis_feature_ttl
        if ttl > 0:
            client.expire(redis_key, ttl)

        logger.info("Stored online features: key=%s fields=%d", redis_key, len(mapping))
        return True
    except Exception as exc:
        logger.warning("store_features failed (fs=%s v=%s entity=%s): %s", fs_id, version_id, entity_kv, exc)
        return False
