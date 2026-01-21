import json
from fastapi.testclient import TestClient
from mlops.main import app

ENDPOINTS = [
    "/plan/project-management",
    "/data/data-storage-versioning",
    "/data/data-acquisition",
    "/data/data-processing-feature-engineering",
    "/data/feature-storage-versioning",
    "/ops/mlops-process-orchestration",
    "/ops/deployment-monitoring",
    "/soft/api-development",
    "/soft/code-storage-versioning",
    "/model/model-development-training",
    "/model/model-storage-versioning",
]

def main():
    client = TestClient(app)
    logs = []
    for path in ENDPOINTS:
        r = client.get(path)
        r.raise_for_status()
        logs.append(r.json())

    # One combined JSON object as requested
    print(json.dumps({"run_id": logs[0]["trace_id"], "logs": logs}, indent=2))

if __name__ == "__main__":
    main()
