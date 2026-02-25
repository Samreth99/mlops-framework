"""
Model Dimension CI Test Suite
testSuiteRef: tests/model-suite.yaml

Checks:
  Training completes without exception
  Accuracy >= minimum threshold (0.80)
  Model artifact stored in MLflow
  Evaluation passes on held-out test data
"""
import os
import pytest
import httpx

BASE_URL  = os.getenv("API_BASE_URL", "http://localhost:8000")
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.80"))

client = httpx.Client(base_url=BASE_URL, timeout=120.0)


# ─────────────────────────────────────────────
# Shared state across tests (set by fixtures)
# ─────────────────────────────────────────────
@pytest.fixture(scope="module")
def experiment_id():
    """Create one experiment for all tests in this module."""
    response = client.post("/model/experiments", json={
        "name": "ci-model-suite",
        "objective": "Automated CI test for breast cancer classifier"
    })
    assert response.status_code == 200, f"Experiment creation failed: {response.text}"
    data = response.json()
    assert "experimentId" in data
    return data["experimentId"]


@pytest.fixture(scope="module")
def training_result(experiment_id):
    """Run training once and share the result across tests."""
    response = client.post("/model/runs/execute-training", json={
        "experimentId": experiment_id,
        "trainingConfig": {
            "n_estimators": 100,
            "criterion": "gini"
        },
        "target_column": "Class",
        "cv_folds": 5,
        "seed": 42
    })
    assert response.status_code == 200, f"Training failed: {response.text}"
    return response.json()


# ─────────────────────────────────────────────
# TEST 1: Training completes without error
# ─────────────────────────────────────────────
def test_training_completes(training_result):
    """POST /runs/execute-training → must return status FINISHED."""
    assert training_result["status"] == "FINISHED", (
        f"Expected FINISHED, got: {training_result['status']}"
    )
    assert "runId" in training_result, "Missing runId in training response"
    assert training_result["runId"], "runId must not be empty"


# ─────────────────────────────────────────────
# TEST 2: Model artifact is saved in MLflow
# ─────────────────────────────────────────────
def test_model_artifact_exists(training_result):
    """modelArtifactRef must be a non-empty URI pointing to MLflow."""
    artifact_ref = training_result.get("modelArtifactRef", "")
    assert artifact_ref, "modelArtifactRef is empty — model was not saved"
    assert artifact_ref.startswith(("runs:/", "models:/")), (
        f"modelArtifactRef has unexpected format: {artifact_ref}"
    )


# ─────────────────────────────────────────────
# TEST 3: Accuracy meets minimum threshold
# ─────────────────────────────────────────────
def test_accuracy_threshold(training_result):
    """CV accuracy must be >= ACCURACY_THRESHOLD (default 0.80)."""
    metrics  = training_result.get("metrics", {})
    accuracy = metrics.get("accuracy")
    assert accuracy is not None, "accuracy metric missing from training response"
    assert accuracy >= ACCURACY_THRESHOLD, (
        f"Accuracy {accuracy:.4f} is below threshold {ACCURACY_THRESHOLD}"
    )


# ─────────────────────────────────────────────
# TEST 4: F1 macro score is acceptable
# ─────────────────────────────────────────────
def test_f1_macro_threshold(training_result):
    """F1 macro must be >= 0.75 (lower bar than accuracy for imbalanced classes)."""
    metrics  = training_result.get("metrics", {})
    f1_macro = metrics.get("f1_macro")
    assert f1_macro is not None, "f1_macro metric missing from training response"
    assert f1_macro >= 0.75, (
        f"f1_macro {f1_macro:.4f} is below minimum 0.75"
    )


# ─────────────────────────────────────────────
# TEST 5: Evaluation on held-out test data passes
# ─────────────────────────────────────────────
def test_evaluation_passes(training_result):
    """POST /runs/execute-evaluation → accuracy on test set must meet threshold."""
    artifact_ref = training_result["modelArtifactRef"]

    response = client.post("/model/runs/execute-evaluation", json={
        "modelCandidateRef": artifact_ref,
        "metrics": ["accuracy", "f1_macro", "precision_macro", "recall_macro"],
        "target_column": "Class"
    })
    assert response.status_code == 200, f"Evaluation failed: {response.text}"

    data    = response.json()
    metrics = data.get("metrics", {})

    assert "accuracy" in metrics, "accuracy missing from evaluation metrics"
    assert metrics["accuracy"] >= ACCURACY_THRESHOLD, (
        f"Evaluation accuracy {metrics['accuracy']:.4f} below threshold {ACCURACY_THRESHOLD}"
    )
    assert "evaluationReportRef" in data, "Missing evaluationReportRef"
    assert data["evaluationReportRef"], "evaluationReportRef must not be empty"


