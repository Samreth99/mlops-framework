from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(env_path: Path) -> None:
    if not env_path.is_file():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.split("#")[0].strip()  # strip inline comments
            if key:
                os.environ[key] = value


# Walk up from this file to find the .env 
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # framework/
_load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    # ── MLflow ──
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/")
    )
    mlflow_artifact_root: str = field(
        default_factory=lambda: os.getenv("MLFLOW_ARTIFACT_ROOT", "s3://mlops-storage/mlflow")
    )
    mlflow_experiment_name: str = field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_NAME", "Default")
    )

    # ── Dataset ──
    default_dataset_path: str = field(
        default_factory=lambda: os.getenv("DEFAULT_DATASET_PATH", "src/mlops/dimensions/data/data-storage/breast_train.csv")
    )
    default_test_dataset_path: str = field(
        default_factory=lambda: os.getenv("DEFAULT_TEST_DATASET_PATH", "src/mlops/dimensions/data/data-storage/breast_test.csv")
    )

    # ── GitHub Actions Auto-Dispatch ──
    github_token: str = field(
        default_factory=lambda: os.getenv("GH_DISPATCH_TOKEN", "")
    )
    github_repo_owner: str = field(
        default_factory=lambda: os.getenv("GH_REPO_OWNER", "")
    )
    github_repo_name: str = field(
        default_factory=lambda: os.getenv("GH_REPO_NAME", "")
    )
    public_api_url: str = field(
        default_factory=lambda: os.getenv("PUBLIC_API_URL", "http://localhost:8000")
    )

    # ── Redis ──
    redis_host: str = field(
        default_factory=lambda: os.getenv("REDIS_HOST", "localhost")
    )
    redis_port: int = field(
        default_factory=lambda: int(os.getenv("REDIS_PORT", "6379"))
    )
    redis_password: str = field(
        default_factory=lambda: os.getenv("REDIS_PASSWORD", "").split("#")[0].strip()
    )
    redis_db: int = field(
        default_factory=lambda: int(os.getenv("REDIS_DB", "0"))
    )
    redis_feature_ttl: int = field(
        default_factory=lambda: int(os.getenv("REDIS_FEATURE_TTL", "0").split("#")[0].strip())
    )

    # ── AWS Configuration ──
    aws_access_key_id: str = field(
        default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID", "")
    )
    aws_secret_access_key: str = field(
        default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY", "")
    )
    aws_s3_bucket: str = field(
        default_factory=lambda: os.getenv("AWS_S3_BUCKET", "mlops-storage")
    )
    aws_s3_region: str = field(
        default_factory=lambda: os.getenv("AWS_S3_REGION", "eu-north-1")
    )
    aws_ecr_login_uri: str = field(
        default_factory=lambda: os.getenv("AWS_ECR_LOGIN_URI", "")
    )
    aws_ecr_repo_name: str = field(
        default_factory=lambda: os.getenv("AWS_ECR_REPO_NAME", "")
    )


settings = Settings()
