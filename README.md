# MLOps Framework

A RESTful MLOps framework built with **FastAPI** that covers the full machine learning lifecycle — from project planning and data ingestion to model training, software packaging, and workflow orchestration via BPMN.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Technology Stack](#technology-stack)
- [Running with Docker Compose](#running-with-docker-compose)
- [Environment Variables](#environment-variables)
- [API Dimensions](#api-dimensions)
  - [Plan](#plan-dimension-plan)
  - [Data](#data-dimension-data)
  - [Model](#model-dimension-model)
  - [Software](#software-dimension-soft)
  - [Orchestrator](#orchestrator-dimension-orchestrator)
  - [Ops _(in progress)_](#ops-dimension-ops)
- [API Documentation](#api-documentation)
- [Running the MLOps BPMN Workflow](#running-the-mlops-bpmn-workflow)

---

## Architecture Overview

```
MLOps Framework
│
├── /plan         → Project planning, repository bootstrap, ticket generation
├── /data         → Data acquisition, versioning, feature store (S3 + DVC + Redis)
├── /model        → MLflow experiments, training/tuning/evaluation, model registry
├── /soft         → Code, builds, packages, tests, releases (GitHub Actions, ECR)
├── /orchestrator → Tickets, contracts, runs, events, workflow orchestration
└── /ops          → Deployment & monitoring (in progress)
```

External services:

- **AWS S3** (`mlops-storage`, `eu-north-1`) — dataset and artifact storage
- **AWS ECR** - docker image storage
- **DVC** — data versioning and lineage
- **MLflow** — experiment tracking and model registry
- **Redis** — online feature serving
- **GitHub Actions** — CI/CD for builds and tests
- **Camunda 7** — BPMN process orchestration

---

## Technology Stack

| Layer                | Technology              |
| -------------------- | ----------------------- |
| API Framework        | FastAPI (Python 3.11)   |
| Model Tracking       | MLflow v3               |
| Data Versioning      | DVC + AWS S3            |
| Online Feature Store | Redis 7                 |
| Artifact Storage     | AWS S3                  |
| Container Registry   | AWS ECR                 |
| CI/CD                | GitHub Actions          |
| BPMN Orchestration   | Camunda 7               |
| Containerisation     | Docker + Docker Compose |

---

## Running with Docker Compose

### Prerequisites

- Docker Desktop installed and running
- AWS credentials — S3 (data versioning) and ECR (container image storage)
- A configured `framework/.env` file — see [Environment Variables](#environment-variables)

### Start all services

```bash
docker compose up --build
```

This starts three services:

| Service  | Container            | Port   | Description                          |
| -------- | -------------------- | ------ | ------------------------------------ |
| `redis`  | `mlops-redis`  | `6379` | Online feature store                 |
| `mlflow` | `mlops-mlflow` | `5000` | Experiment tracking & model registry |
| `api`    | `mlops-api`    | `8000` | FastAPI application                  |

### Verify services are up

```bash
# API health check
curl http://localhost:8000/health

# MLflow UI
open http://localhost:5000

# API interactive docs
open http://localhost:8000/docs
```

### Stop services

```bash
docker compose down
```

To also remove volumes (clears MLflow DB and Redis data):

```bash
docker compose down -v
```

### Running the API locally (without Docker)

```bash
cd framework
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e .

uvicorn mlops.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Environment Variables

Create `framework/.env` with the following variables:

```env
# AWS S3 (required for data versioning)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_S3_BUCKET=mlops-storage
AWS_S3_REGION=eu-north-1

# MLflow (auto-set in Docker Compose)
MLFLOW_TRACKING_URI=http://localhost:5000/
MLFLOW_ARTIFACT_ROOT=s3://mlops-storage/mlflow

# Redis (auto-set in Docker Compose)
REDIS_HOST=localhost
REDIS_PORT=6379

# GitHub Actions (for soft dimension CI/CD triggers)
GH_DISPATCH_TOKEN=ghp_your_token
GH_REPO_OWNER=your_github_username
GH_REPO_NAME=your_repo_name
PUBLIC_API_URL=http://localhost:8000

# AWS ECR (required for soft dimension — Docker image push after build)
AWS_ECR_LOGIN_URI=123456789.dkr.ecr.eu-north-1.amazonaws.com
AWS_ECR_REPO_NAME=mlops
```

---

## API Dimensions

All endpoints are available at `http://localhost:8000`. Interactive documentation is at `/docs`.

---

### Plan Dimension (`/plan`)

Manages ML project lifecycle from requirements to roadmap.

#### Project Management

| Method | Path                                      | Description                               |
| ------ | ----------------------------------------- | ----------------------------------------- |
| `POST` | `/plan/projects`                          | Create a new ML project                   |
| `GET`  | `/plan/projects`                          | List all projects                         |
| `GET`  | `/plan/projects/{projectId}`              | Get project details                       |
| `PUT`  | `/plan/projects/{projectId}`              | Update a project                          |
| `POST` | `/plan/projects/{projectId}/requirements` | Upsert project requirements               |
| `GET`  | `/plan/projects/{projectId}/requirements` | Get project requirements                  |
| `POST` | `/plan/projects/{projectId}/ml-problem`   | Define or update ML problem specification |
| `GET`  | `/plan/projects/{projectId}/ml-problem`   | Get ML problem spec                       |
| `POST` | `/plan/projects/{projectId}/data-sources` | Register planned data sources             |
| `GET`  | `/plan/projects/{projectId}/data-sources` | Get registered data sources               |
| `POST` | `/plan/projects/{projectId}/plan`         | Upsert project roadmap and milestones     |
| `GET`  | `/plan/projects/{projectId}/plan`         | Get project plan                          |

#### Repository Management

| Method | Path                           | Description                                             |
| ------ | ------------------------------ | ------------------------------------------------------- |
| `POST` | `/plan/repositories/bootstrap` | Bootstrap code, data, and model repositories/registries |

#### Ticket Generation

| Method | Path            | Description                                             |
| ------ | --------------- | ------------------------------------------------------- |
| `POST` | `/plan/tickets` | Generate MPO governance tickets from planning decisions |

---

### Data Dimension (`/data`)

End-to-end data management: ingestion, versioning, processing, and feature serving. Storage backed by **AWS S3 + DVC** for datasets and **Redis** for online features.

#### Data Acquisition

| Method | Path                                    | Description                                              |
| ------ | --------------------------------------- | -------------------------------------------------------- |
| `POST` | `/data/sources`                         | Register a data source connector                         |
| `GET`  | `/data/sources`                         | List all source connectors                               |
| `POST` | `/data/ingestions`                      | Trigger ingestion — uploads file to S3 and DVC-tracks it |
| `GET`  | `/data/ingestions`                      | List ingestion executions                                |
| `GET`  | `/data/ingestions/{ingestionId}`        | Get ingestion configuration                              |
| `GET`  | `/data/ingestions/{ingestionId}/status` | Poll ingestion progress/health                           |

#### Data Storage & Versioning

| Method | Path                                              | Description                                         |
| ------ | ------------------------------------------------- | --------------------------------------------------- |
| `POST` | `/data/datasets`                                  | Register a logical dataset                          |
| `GET`  | `/data/datasets`                                  | List all datasets                                   |
| `GET`  | `/data/datasets/{datasetId}`                      | Get dataset metadata                                |
| `POST` | `/data/datasets/{datasetId}`                      | Update dataset metadata                             |
| `POST` | `/data/datasets/{datasetId}/versions`             | Create dataset version (uploads to S3, DVC-tracked) |
| `GET`  | `/data/datasets/{datasetId}/versions`             | List all dataset versions                           |
| `GET`  | `/data/datasets/{datasetId}/versions/{versionId}` | Fetch specific version                              |
| `GET`  | `/data/datasets/{datasetId}/lineage`              | Query dataset provenance/lineage graph              |

#### Data Processing & Feature Engineering

| Method | Path               | Description                                                  |
| ------ | ------------------ | ------------------------------------------------------------ |
| `POST` | `/data/preprocess` | Execute preprocessing pipeline → creates new dataset version |
| `POST` | `/data/validate`   | Run data quality validation checks                           |
| `POST` | `/data/analyze`    | Profiling, EDA, and drift baseline computation               |
| `POST` | `/data/label`      | Apply labeling strategy → labeled dataset version            |
| `POST` | `/data/engineer`   | Build feature sets from raw datasets                         |

#### Feature Storage & Versioning

| Method | Path                                                 | Description                                     |
| ------ | ---------------------------------------------------- | ----------------------------------------------- |
| `GET`  | `/data/features/online/get`                          | Retrieve online features by entity keys (Redis) |
| `POST` | `/data/features/online/store`                        | Write entity feature values to Redis            |
| `POST` | `/data/features/online/materialize`                  | Bulk-load feature CSV into Redis                |
| `GET`  | `/data/features/offline/get`                         | Retrieve offline feature table reference (S3)   |
| `POST` | `/data/features`                                     | Register a logical feature set                  |
| `GET`  | `/data/features`                                     | List all feature sets                           |
| `GET`  | `/data/features/{featureSetId}`                      | Get feature set metadata                        |
| `POST` | `/data/features/{featureSetId}`                      | Update feature set metadata                     |
| `POST` | `/data/features/{featureSetId}/versions`             | Create/publish a feature version                |
| `GET`  | `/data/features/{featureSetId}/versions`             | List feature versions                           |
| `GET`  | `/data/features/{featureSetId}/versions/{versionId}` | Fetch specific feature version                  |

---

### Model Dimension (`/model`)

Model development and registry backed by **MLflow**.

#### Model Development & Training

| Method | Path                                 | Description                                            |
| ------ | ------------------------------------ | ------------------------------------------------------ |
| `POST` | `/model/experiments`                 | Create an MLflow experiment container                  |
| `GET`  | `/model/experiments`                 | List all experiments                                   |
| `GET`  | `/model/experiments/{experiment_id}` | Get experiment details                                 |
| `POST` | `/model/runs/execute-training`       | Start a training run (logs params, metrics, artifacts) |
| `POST` | `/model/runs/execute-tuning`         | Run hyperparameter tuning                              |
| `POST` | `/model/runs/execute-validation`     | Run model validation gates                             |
| `POST` | `/model/runs/execute-evaluation`     | Run final model evaluation/testing                     |

#### Model Storage & Versioning

| Method | Path                                                    | Description                                       |
| ------ | ------------------------------------------------------- | ------------------------------------------------- |
| `POST` | `/model/models`                                         | Register a model family                           |
| `GET`  | `/model/models`                                         | List all registered models                        |
| `GET`  | `/model/models/{modelId}`                               | Get model metadata                                |
| `POST` | `/model/models/{modelId}`                               | Update model metadata                             |
| `POST` | `/model/models/{modelId}/versions`                      | Create a new model version                        |
| `GET`  | `/model/models/{modelId}/versions`                      | List all model versions                           |
| `GET`  | `/model/models/{modelId}/versions/{versionId}`          | Fetch a specific model version                    |
| `POST` | `/model/models/{modelId}/register`                      | Register MLflow run artifact as a versioned model |
| `POST` | `/model/models/{modelId}/versions/{versionId}/evidence` | Attach validation reports/evidence                |
| `GET`  | `/model/models/{modelId}/versions/{versionId}/evidence` | Query attached evidence                           |

---

### Software Dimension (`/soft`)

CI/CD pipeline management — code commits, Docker builds, test runs, package artifacts, and releases. Build and test jobs are dispatched to **GitHub Actions** via `workflow_dispatch`. Status is polled back from the GitHub API on demand — no webhook/ngrok required.

#### CI/CD Workflow Details

Two GitHub Actions workflows are triggered automatically by the API:

---

**`docker-build-ci.yml` — Docker Build CI**

Triggered by: `POST /soft/builds`

Flow:
1. API registers the build record and dispatches `docker-build-ci.yml` via GitHub Actions `workflow_dispatch` (inputs: `buildId`, `imageTag`)
2. GitHub Actions builds `framework/Dockerfile`
3. Runs **unit tests** inside the container
4. Pushes the image to **AWS ECR** (`AWS_ECR_LOGIN_URI:imageTag`)
5. Status is polled via `GET /soft/builds/{buildId}/status` → syncs directly from the GitHub Actions API

Build states: `QUEUED` → `RUNNING` → `BUILT` / `FAILED`

---

**`docker-test-ci.yml` — Docker Test CI**

Triggered by: `POST /soft/tests`

Flow:
1. API looks up the `imageRef` from the linked build (must be `BUILT`) and dispatches `docker-test-ci.yml` (inputs: `testRunId`, `imageRef`)
2. GitHub Actions pulls the image from **AWS ECR**
3. Spins up a **Docker Compose** environment with MLflow and Redis
4. Runs **integration tests** against the live stack
5. Status is polled via `GET /soft/tests/{testRunId}` → syncs directly from the GitHub Actions API

Test states: `QUEUED` → `RUNNING` → `PASSED` / `FAILED`

---

**Typical call sequence:**

```
POST /soft/builds          → triggers docker-build-ci.yml
GET  /soft/builds/{id}/status  → poll until BUILT

POST /soft/tests           → triggers docker-test-ci.yml (pulls image from ECR)
GET  /soft/tests/{id}      → poll until PASSED / FAILED
```

Required env vars for CI dispatch: `GH_DISPATCH_TOKEN`, `GH_REPO_OWNER`, `GH_REPO_NAME`, `AWS_ECR_LOGIN_URI`

---

#### Code Management

| Method | Path                      | Description                       |
| ------ | ------------------------- | --------------------------------- |
| `POST` | `/soft/code/repositories` | Register a code repository        |
| `GET`  | `/soft/code/repositories` | List all repositories             |
| `POST` | `/soft/code/commits`      | Record a commit tied to a build   |
| `GET`  | `/soft/code/commits`      | List commits (filterable by repo) |
| `POST` | `/soft/code/tags`         | Create a tag/release pointer      |
| `GET`  | `/soft/code/tags`         | List tags (filterable by repo)    |

#### Build Management

| Method  | Path                            | Description                                            |
| ------- | ------------------------------- | ------------------------------------------------------ |
| `POST`  | `/soft/builds`                  | Trigger Docker image build via GitHub Actions          |
| `GET`   | `/soft/builds`                  | List builds (filterable by repo)                       |
| `GET`   | `/soft/builds/{buildId}`        | Get build metadata                                     |
| `GET`   | `/soft/builds/{buildId}/status` | Poll build progress and outcome                        |
| `PATCH` | `/soft/builds/{buildId}`        | CI callback from GitHub Actions to update build status |

#### Package Management

| Method | Path                         | Description                         |
| ------ | ---------------------------- | ----------------------------------- |
| `POST` | `/soft/packages`             | Create/register a packaged artifact |
| `GET`  | `/soft/packages`             | List all packaged artifacts         |
| `GET`  | `/soft/packages/{packageId}` | Get package metadata                |

#### Test Management

| Method  | Path                      | Description                                           |
| ------- | ------------------------- | ----------------------------------------------------- |
| `POST`  | `/soft/tests`             | Trigger test execution against a Docker image         |
| `GET`   | `/soft/tests`             | List test executions (filterable by package/build)    |
| `GET`   | `/soft/tests/{testRunId}` | Get test results summary                              |
| `PATCH` | `/soft/tests/{testRunId}` | CI callback from GitHub Actions to update test status |

#### Release Management

| Method | Path                         | Description                    |
| ------ | ---------------------------- | ------------------------------ |
| `POST` | `/soft/releases`             | Create a software release      |
| `GET`  | `/soft/releases`             | List all releases              |
| `GET`  | `/soft/releases/{releaseId}` | Get release details            |
| `POST` | `/soft/releases/{releaseId}` | Update release status or notes |

---

### Orchestrator Dimension (`/orchestrator`)

Governance and coordination layer — manages tickets, contracts, execution runs, and events across all dimensions.

#### Tickets

| Method | Path                                         | Description                                 |
| ------ | -------------------------------------------- | ------------------------------------------- |
| `POST` | `/orchestrator/tickets`                      | Create a governance ticket                  |
| `GET`  | `/orchestrator/tickets`                      | List tickets (filterable by type, priority) |
| `POST` | `/orchestrator/tickets/{ticketId}`           | Update ticket core fields                   |
| `POST` | `/orchestrator/tickets/{ticketId}/assign`    | Set or transfer assignee                    |
| `POST` | `/orchestrator/tickets/{ticketId}/state`     | Transition ticket status                    |
| `POST` | `/orchestrator/tickets/{ticketId}/evidence`  | Attach validation reports                   |
| `POST` | `/orchestrator/tickets/{ticketId}/artifacts` | Link dataset/model/service versions         |
| `GET`  | `/orchestrator/tickets/{ticketId}/history`   | Retrieve state transition log               |
| `POST` | `/orchestrator/tickets/{ticketId}/routing`   | Persist routing decision                    |

#### Contracts

| Method | Path                                            | Description                           |
| ------ | ----------------------------------------------- | ------------------------------------- |
| `POST` | `/orchestrator/contracts`                       | Register a coordination contract      |
| `GET`  | `/orchestrator/contracts`                       | List contracts (filterable by domain) |
| `GET`  | `/orchestrator/contracts/{contractId}`          | Read contract snapshot                |
| `POST` | `/orchestrator/contracts/{contractId}`          | Update contract metadata              |
| `POST` | `/orchestrator/contracts/{contractId}/versions` | Create a new contract version         |
| `POST` | `/orchestrator/contracts/{contractId}/validate` | Validate a payload against a contract |

#### Runs

| Method | Path                                   | Description                                |
| ------ | -------------------------------------- | ------------------------------------------ |
| `POST` | `/orchestrator/runs`                   | Create an orchestrated execution run       |
| `GET`  | `/orchestrator/runs`                   | List runs (filterable by ticketId, status) |
| `GET`  | `/orchestrator/runs/{runId}`           | Get run status and timeline                |
| `GET`  | `/orchestrator/runs/{runId}/trace`     | Fetch end-to-end execution trace           |
| `GET`  | `/orchestrator/runs/{runId}/artifacts` | List artifacts produced/consumed by run    |
| `GET`  | `/orchestrator/runs/{runId}/tickets`   | List tickets linked to run                 |

#### Events

| Method | Path                             | Description              |
| ------ | -------------------------------- | ------------------------ |
| `POST` | `/orchestrator/events/publish`   | Publish a typed event    |
| `POST` | `/orchestrator/events/subscribe` | Consume or stream events |

#### Orchestration

| Method | Path                                           | Description                             |
| ------ | ---------------------------------------------- | --------------------------------------- |
| `POST` | `/orchestrator/orchestrator/context`           | Set orchestration context variables     |
| `POST` | `/orchestrator/orchestrator/start`             | Bootstrap orchestrated runs             |
| `POST` | `/orchestrator/orchestrator/health-checks/run` | Trigger health check workflow           |
| `POST` | `/orchestrator/router/classify`                | Classify a ticket into a routing action |

---

### Ops Dimension (`/ops`)

> **Status: In progress** — currently exposes monitoring rule management only.

| Method | Path                    | Description                                |
| ------ | ----------------------- | ------------------------------------------ |
| `POST` | `/ops/monitoring/rules` | Create and evaluate an alerting/drift rule |
| `GET`  | `/ops/monitoring/rules` | List all monitoring rules                  |

---

## API Documentation

When the API is running, interactive documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **Root endpoint**: `http://localhost:8000/` — returns framework info
- **Health check**: `http://localhost:8000/health`

---

## Running the MLOps BPMN Workflow

The repository includes `mlops.bpmn` — a BPMN 2.0 process definition that orchestrates the full MLOps pipeline end-to-end using **Camunda 7**.

### Prerequisites

- [Camunda Modeler](https://camunda.com/download/modeler/) — desktop application to view, edit, deploy, and start BPMN processes

### Step 1 — Start Camunda 7

Run the Camunda BPM platform using Docker:

```bash
docker run -d --name camunda7 -p 8080:8080 camunda/camunda-bpm-platform:tomcat-7.23.0
```

### Step 2 — Access Camunda interfaces

| Interface                            | URL                                                          | Credentials     |
| ------------------------------------ | ------------------------------------------------------------ | --------------- |
| Camunda Cockpit (process monitoring) | `http://localhost:8080/camunda`                              | `demo` / `demo` |
| Camunda REST API                     | `http://localhost:8080/engine-rest`                          | —               |
| Process instance history (logs)      | `http://localhost:8080/engine-rest/history/process-instance` | —               |
| Tasklist                             | `http://localhost:8080/camunda/app/tasklist`                 | `demo` / `demo` |

### Step 3 — Deploy the BPMN process

1. Open **Camunda Modeler**
2. Open `mlops.bpmn` from the repository root
3. Click **Deploy** (rocket icon in the toolbar)
4. Set deployment endpoint to `http://localhost:8080/engine-rest`
5. Click **Deploy**

### Step 4 — Start a process instance

1. In Camunda Modeler, after deploying, click **Start Process Instance**
2. Provide the required variables (e.g. `projectId`, `experimentName`, `datasetName`, `owner`)
3. Click **Run**


### BPMN Process Overview

The `mlops.bpmn` process (`MPO — MLOps Process Orchestrator`) coordinates all six dimensions via message events:

| Message                   | Triggered by         | Target               |
| ------------------------- | -------------------- | -------------------- |
| `INIT_ORCHESTRATOR`       | Start event          | Orchestrator setup   |
| `START_TRAINING`          | Data dimension ready | Model dimension      |
| `MODEL_CANDIDATE_READY`   | Training complete    | Validation gate      |
| `SOFTWARE_ARTIFACT_READY` | Build complete       | Test execution       |
| `RELEASE_BUNDLE_READY`    | Tests pass           | Release creation     |
| `RETRAINING`              | Drift detected       | Retrain loop         |
| `DATA_BUG_FOUND`          | Validation failure   | Data correction loop |

The API endpoints (served at `http://localhost:8000`) are called as **service tasks** within the BPMN process. Make sure the API is running before starting a process instance.

### Running both together (recommended)

```bash
# Terminal 1 — start the MLOps API and its dependencies
docker compose up --build

# Terminal 2 — start Camunda 7
docker run -d --name camunda7 -p 8080:8080 camunda/camunda-bpm-platform:tomcat-7.23.0
```

Then open Camunda Modeler, deploy `mlops.bpmn` to `http://localhost:8080/engine-rest`, and start a process instance.
