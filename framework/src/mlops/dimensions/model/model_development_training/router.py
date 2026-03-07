"""
Model Development & Training API endpoints.
  POST   /experiments                 – Create an experiment container
  GET    /experiments                 – List experiment containers
  GET    /experiments/{experiment_id} – Get experiment details
  POST   /runs/execute-training       – Start a training run
  POST   /runs/execute-tuning         – Run hyperparameter tuning
  POST   /runs/execute-validation     – Run model validation gates
  POST   /runs/execute-evaluation     – Run evaluation / testing
"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from ....core.logs import make_log
from ..schemas import (
    ExperimentCreateRequest,
    ExperimentResponse,
    ExecuteTrainingRequest,
    ExecuteTrainingResponse,
    ExecuteTuningRequest,
    ExecuteTuningResponse,
    ExecuteValidationRequest,
    ExecuteValidationResponse,
    ExecuteEvaluationRequest,
    ExecuteEvaluationResponse,
)
from .. import mlflow_service as svc

router = APIRouter()


# ─────────────────────────────────────────────
# EXPERIMENTS
# ─────────────────────────────────────────────
@router.post("/experiments", response_model=ExperimentResponse,
             summary="Create an experiment container (group of runs)")
def create_experiment(req: ExperimentCreateRequest):
    """
    Create an MLflow experiment.

    **API spec inputs:** projectId, name, objective, ticketId
    **Returns:** experimentId, name, objective, artifact_location, lifecycle_stage, creation_time
    """
    log = make_log(area="Model", component="Model Development & Training", endpoint="/model/experiments", meta={"name": req.name})
    try:
        result = svc.create_experiment(
            name=req.name,
            objective=req.objective,
            ticket_id=req.ticketId,
        )
        return ExperimentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "experiment_creation_failed", "message": str(e)}
        })


@router.get("/experiments", response_model=List[ExperimentResponse],
            summary="List all experiment containers")
def list_experiments():
    """
    List all MLflow experiments visible to the tracking server.
    """
    log = make_log(area="Model", component="Model Development & Training", endpoint="/model/experiments")
    try:
        results = svc.list_experiments()
        return [ExperimentResponse(**r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "list_experiments_failed", "message": str(e)}
        })


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse,
            summary="Get experiment details")
def get_experiment(experiment_id: str):
    log = make_log(area="Model", component="Model Development & Training", endpoint=f"/model/experiments/{experiment_id}", meta={"experiment_id": experiment_id})
    try:
        result = svc.get_experiment(experiment_id)
        return ExperimentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "experiment_not_found", "message": str(e)}
        })


# ─────────────────────────────────────────────
# RUNS – EXECUTE TRAINING
# ─────────────────────────────────────────────
@router.post("/runs/execute-training", response_model=ExecuteTrainingResponse,
             summary="Start a training run with pinned data/features and config")
def execute_training(req: ExecuteTrainingRequest):
    """
    Execute a training run: loads data, trains a model, logs artefacts to MLflow.

    API spec inputs: experimentId, datasetVersion|featureSetVersion,
    trainingConfig, seed, ticketId

    Returns: runId, modelArtifactRef (when done), metrics refs
    """
    log = make_log(area="Model", component="Model Development & Training", endpoint="/model/runs/execute-training", meta={"experimentId": req.experimentId})
    dataset_source = req.datasetVersion or req.featureSetVersion
    try:
        result = svc.execute_training(
            experiment_id=req.experimentId,
            dataset_source=dataset_source,
            target_column=req.target_column,
            training_config=req.trainingConfig or {},
            cv_folds=req.cv_folds,
            seed=req.seed or 42,
            registered_model_name=req.registered_model_name,
        )
        return ExecuteTrainingResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={
            "error": {"code": "training_input_error", "message": str(e)}
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "training_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# RUNS – EXECUTE TUNING
# ─────────────────────────────────────────────
@router.post("/runs/execute-tuning", response_model=ExecuteTuningResponse,
             summary="Run hyperparameter tuning with tracked candidates")
def execute_tuning(req: ExecuteTuningRequest):
    """
    Grid-search over a search space. Each candidate is logged as a nested MLflow run.

    API spec inputs: baseConfig, searchSpace, budget,
    datasetVersion|featureSetVersion, ticketId

    Returns: runId, best params, candidate leaderboard ref
    """
    log = make_log(area="Model", component="Model Development & Training", endpoint="/model/runs/execute-tuning", meta={"experimentId": req.experimentId, "budget": req.budget})
    dataset_source = req.datasetVersion or req.featureSetVersion
    try:
        result = svc.execute_tuning(
            experiment_id=req.experimentId,
            base_config=req.baseConfig,
            search_space=req.searchSpace,
            budget=req.budget,
            dataset_source=dataset_source,
            target_column=req.target_column,
            test_size=req.test_size,
            seed=req.seed or 42,
        )
        return ExecuteTuningResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={
            "error": {"code": "tuning_input_error", "message": str(e)}
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "tuning_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# RUNS – EXECUTE VALIDATION
# ─────────────────────────────────────────────
@router.post("/runs/execute-validation", response_model=ExecuteValidationResponse,
             summary="Run model validation gates (checks, thresholds)")
def execute_validation(req: ExecuteValidationRequest):
    """
    Validate a model candidate against threshold rules.

    API spec inputs: modelCandidateRef, validationSuiteRef|rules,
    thresholds, ticketId

    Returns: passed=true|false, validation report ref
    """
    log = make_log(area="Model", component="Model Development & Training", endpoint="/model/runs/execute-validation", meta={"modelCandidateRef": req.modelCandidateRef})
    rules_dicts = [r.model_dump() for r in req.rules] if req.rules else None
    try:
        result = svc.execute_validation(
            model_candidate_ref=req.modelCandidateRef,
            rules=rules_dicts,
            thresholds=req.thresholds,
        )
        return ExecuteValidationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "validation_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# RUNS – EXECUTE EVALUATION
# ─────────────────────────────────────────────
@router.post("/runs/execute-evaluation", response_model=ExecuteEvaluationResponse,
             summary="Run evaluation/testing and produce evidence artifacts")
def execute_evaluation(req: ExecuteEvaluationRequest):
    """
    Evaluate a model on a held-out dataset and produce metric evidence.

    API spec inputs: modelCandidateRef, evalDatasetVersion, metrics[],
    fairnessChecks[], ticketId

    Returns: evaluation report ref, metrics, pass/fail per check
    """
    log = make_log(area="Model", component="Model Development & Training", endpoint="/model/runs/execute-evaluation", meta={"modelCandidateRef": req.modelCandidateRef, "metrics": req.metrics})
    try:
        result = svc.execute_evaluation(
            model_candidate_ref=req.modelCandidateRef,
            eval_dataset_source=req.evalDatasetVersion,
            metric_names=req.metrics,
            target_column=req.target_column,
            fairness_checks=req.fairnessChecks,
        )
        return ExecuteEvaluationResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={
            "error": {"code": "evaluation_input_error", "message": str(e)}
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "evaluation_failed", "message": str(e)}
        })
