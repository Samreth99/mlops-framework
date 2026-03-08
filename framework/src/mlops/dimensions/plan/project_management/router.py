"""
Project Management API endpoints.
  POST   /projects                              – Create a project
  GET    /projects                              – List all projects
  GET    /projects/{projectId}                  – Get project details
  PUT    /projects/{projectId}                  – Update a project record
  POST   /projects/{projectId}/requirements     – Upsert requirements for a project
  GET    /projects/{projectId}/requirements     – Get stored requirements
  POST   /projects/{projectId}/ml-problem       – Define or update the ML problem framing
  GET    /projects/{projectId}/ml-problem       – Get the ML problem spec
  POST   /projects/{projectId}/data-sources     – Register planned data sources
  GET    /projects/{projectId}/data-sources     – Get planned data sources
  POST   /projects/{projectId}/plan             – Upsert roadmap / milestones / deliverables
  GET    /projects/{projectId}/plan             – Get the project plan snapshot
"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from ....core.logs import make_log
from ..schemas import (
    ProjectCreateRequest,
    ProjectUpdateRequest,
    ProjectResponse,
    RequirementsRequest,
    RequirementsResponse,
    MLProblemRequest,
    MLProblemResponse,
    ProjectDataSourcesRequest,
    ProjectDataSourcesResponse,
    ProjectPlanRequest,
    ProjectPlanResponse,
)
from .. import plan_service as svc

router = APIRouter()


# ─────────────────────────────────────────────
# PROJECTS – CRUD
# ─────────────────────────────────────────────

@router.post("/projects", response_model=ProjectResponse,
             summary="Create a project and its high-level context")
def create_project(req: ProjectCreateRequest):
    """
    Create a new MLOps project record.

    **Inputs:** name, owner, domain, objective, stakeholders?[]
    **Returns:** projectId, metadata snapshot
    """
    log = make_log(
        area="Plan", component="Project Management",
        endpoint="/plan/projects", meta={"name": req.name},
    )
    try:
        result = svc.create_project(
            name=req.name,
            owner=req.owner,
            domain=req.domain,
            objective=req.objective,
            stakeholders=req.stakeholders,
        )
        return ProjectResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "project_creation_failed", "message": str(e)}
        })


@router.get("/projects", response_model=List[ProjectResponse],
            summary="List all projects")
def list_projects():
    """List all registered projects."""
    log = make_log(area="Plan", component="Project Management", endpoint="/plan/projects")
    try:
        results = svc.list_projects()
        return [ProjectResponse(**r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "list_projects_failed", "message": str(e)}
        })


@router.get("/projects/{projectId}", response_model=ProjectResponse,
            summary="Read a project record")
def get_project(projectId: str):
    """
    Retrieve a project by ID.

    **Inputs:** projectId (path)
    **Returns:** full project snapshot
    """
    log = make_log(
        area="Plan", component="Project Management",
        endpoint=f"/plan/projects/{projectId}", meta={"projectId": projectId},
    )
    try:
        result = svc.get_project(projectId)
        return ProjectResponse(**result)
    except KeyError as e:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "project_not_found", "message": str(e)}
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "get_project_failed", "message": str(e)}
        })


@router.put("/projects/{projectId}", response_model=ProjectResponse,
            summary="Update a project record")
def update_project(projectId: str, req: ProjectUpdateRequest):
    """
    Update fields of an existing project.

    **Inputs:** name?, objective?, stakeholders?[], status?
    **Returns:** updated project snapshot
    """
    log = make_log(
        area="Plan", component="Project Management",
        endpoint=f"/plan/projects/{projectId}", meta={"projectId": projectId},
    )
    try:
        result = svc.update_project(projectId, req.model_dump(exclude_none=True))
        return ProjectResponse(**result)
    except KeyError as e:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "project_not_found", "message": str(e)}
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "update_project_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# REQUIREMENTS
# ─────────────────────────────────────────────

@router.post("/projects/{projectId}/requirements", response_model=RequirementsResponse,
             summary="Upsert requirements, constraints, and acceptance criteria")
def upsert_requirements(projectId: str, req: RequirementsRequest):
    """
    Store or update the requirement set for a project.

    **Inputs:** requirements[] (id, text, priority, constraints?[], acceptanceCriteria?[]), ticketId?
    **Returns:** stored requirement set + version/diff
    """
    log = make_log(
        area="Plan", component="Project Management",
        endpoint=f"/plan/projects/{projectId}/requirements",
        meta={"projectId": projectId, "count": len(req.requirements)},
    )
    try:
        reqs_dicts = [r.model_dump() for r in req.requirements]
        result = svc.upsert_requirements(projectId, reqs_dicts)
        return RequirementsResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "requirements_upsert_failed", "message": str(e)}
        })


@router.get("/projects/{projectId}/requirements", response_model=RequirementsResponse,
            summary="Get stored requirements for a project")
def get_requirements(projectId: str):
    log = make_log(
        area="Plan", component="Project Management",
        endpoint=f"/plan/projects/{projectId}/requirements",
        meta={"projectId": projectId},
    )
    try:
        result = svc.get_requirements(projectId)
        return RequirementsResponse(**result)
    except KeyError as e:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "requirements_not_found", "message": str(e)}
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "get_requirements_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# ML PROBLEM DEFINITION
# ─────────────────────────────────────────────

@router.post("/projects/{projectId}/ml-problem", response_model=MLProblemResponse,
             summary="Capture ML framing: targets, metrics, and risk constraints")
def upsert_ml_problem(projectId: str, req: MLProblemRequest):
    """
    Define or update the ML problem specification for a project.

    **Inputs:** taskType, target, metrics[], fairnessOrSafetyConstraints?[], assumptions?[]
    **Returns:** ML problem spec + auto-generated gate identifiers
    """
    log = make_log(
        area="Plan", component="Project Management",
        endpoint=f"/plan/projects/{projectId}/ml-problem",
        meta={"projectId": projectId, "taskType": req.taskType},
    )
    try:
        result = svc.upsert_ml_problem(
            project_id=projectId,
            task_type=req.taskType,
            target=req.target,
            metrics=req.metrics,
            fairness_constraints=req.fairnessOrSafetyConstraints,
            assumptions=req.assumptions,
        )
        return MLProblemResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "ml_problem_upsert_failed", "message": str(e)}
        })


@router.get("/projects/{projectId}/ml-problem", response_model=MLProblemResponse,
            summary="Get the ML problem spec for a project")
def get_ml_problem(projectId: str):
    log = make_log(
        area="Plan", component="Project Management",
        endpoint=f"/plan/projects/{projectId}/ml-problem",
        meta={"projectId": projectId},
    )
    try:
        result = svc.get_ml_problem(projectId)
        return MLProblemResponse(**result)
    except KeyError as e:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "ml_problem_not_found", "message": str(e)}
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "get_ml_problem_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# PROJECT DATA SOURCES (planning-level)
# ─────────────────────────────────────────────

@router.post("/projects/{projectId}/data-sources", response_model=ProjectDataSourcesResponse,
             summary="Register data sources and collection assumptions for a project")
def register_data_sources(projectId: str, req: ProjectDataSourcesRequest):
    """
    Register the planned data sources associated with a project.

    **Inputs:** sources[] (name, owner, accessMode, refreshCadence, risk)
    **Returns:** registered sources + plan:// refs
    """
    log = make_log(
        area="Plan", component="Project Management",
        endpoint=f"/plan/projects/{projectId}/data-sources",
        meta={"projectId": projectId, "sourceCount": len(req.sources)},
    )
    try:
        sources_dicts = [s.model_dump() for s in req.sources]
        result = svc.register_project_data_sources(projectId, sources_dicts)
        return ProjectDataSourcesResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "data_sources_registration_failed", "message": str(e)}
        })


@router.get("/projects/{projectId}/data-sources", response_model=ProjectDataSourcesResponse,
            summary="Get registered data sources for a project")
def get_data_sources(projectId: str):
    log = make_log(
        area="Plan", component="Project Management",
        endpoint=f"/plan/projects/{projectId}/data-sources",
        meta={"projectId": projectId},
    )
    try:
        result = svc.get_project_data_sources(projectId)
        return ProjectDataSourcesResponse(**result)
    except KeyError as e:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "data_sources_not_found", "message": str(e)}
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "get_data_sources_failed", "message": str(e)}
        })


# ─────────────────────────────────────────────
# PROJECT PLAN (Roadmap / Milestones)
# ─────────────────────────────────────────────

@router.post("/projects/{projectId}/plan", response_model=ProjectPlanResponse,
             summary="Manage roadmap, milestones, and deliverables")
def upsert_project_plan(projectId: str, req: ProjectPlanRequest):
    """
    Create or update the project roadmap.

    **Inputs:** milestones[], deliverables[], timeline, roles?[]
    **Returns:** plan snapshot + milestoneIds
    """
    log = make_log(
        area="Plan", component="Project Management",
        endpoint=f"/plan/projects/{projectId}/plan",
        meta={"projectId": projectId, "milestones": len(req.milestones)},
    )
    try:
        milestones_dicts = [m.model_dump() for m in req.milestones]
        deliverables_dicts = [d.model_dump() for d in req.deliverables]
        result = svc.upsert_project_plan(
            project_id=projectId,
            milestones=milestones_dicts,
            deliverables=deliverables_dicts,
            timeline=req.timeline,
            roles=req.roles,
        )
        return ProjectPlanResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "plan_upsert_failed", "message": str(e)}
        })


@router.get("/projects/{projectId}/plan", response_model=ProjectPlanResponse,
            summary="Get the project plan snapshot")
def get_project_plan(projectId: str):
    log = make_log(
        area="Plan", component="Project Management",
        endpoint=f"/plan/projects/{projectId}/plan",
        meta={"projectId": projectId},
    )
    try:
        result = svc.get_project_plan(projectId)
        return ProjectPlanResponse(**result)
    except KeyError as e:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "plan_not_found", "message": str(e)}
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "get_plan_failed", "message": str(e)}
        })
