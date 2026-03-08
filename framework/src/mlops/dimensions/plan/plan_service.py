from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_REGISTRY_PATH = Path(__file__).parent / "plan_registry.json"


def _load_registry() -> Dict[str, Any]:
    if _REGISTRY_PATH.is_file():
        with open(_REGISTRY_PATH) as f:
            return json.load(f)
    return {
        "projects": {},
        "requirements": {},
        "ml_problems": {},
        "data_sources": {},
        "plans": {},
        "repos": {},
        "tickets": {},
    }


def _save_registry(reg: Dict[str, Any]) -> None:
    with open(_REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────
# PROJECTS
# ─────────────────────────────────────────────

def create_project(
    name: str,
    owner: str,
    domain: str,
    objective: str,
    stakeholders: Optional[List[str]] = None,
) -> Dict[str, Any]:
    reg = _load_registry()
    project_id = f"proj_{uuid.uuid4().hex[:8]}"
    project = {
        "projectId": project_id,
        "name": name,
        "owner": owner,
        "domain": domain,
        "objective": objective,
        "stakeholders": stakeholders or [],
        "status": "active",
        "createdAt": _now(),
    }
    reg["projects"][project_id] = project
    _save_registry(reg)
    return project


def list_projects() -> List[Dict[str, Any]]:
    return list(_load_registry()["projects"].values())


def get_project(project_id: str) -> Dict[str, Any]:
    reg = _load_registry()
    if project_id not in reg["projects"]:
        raise KeyError(f"Project {project_id} not found")
    return reg["projects"][project_id]


def update_project(project_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    reg = _load_registry()
    if project_id not in reg["projects"]:
        raise KeyError(f"Project {project_id} not found")
    project = reg["projects"][project_id]
    for k, v in updates.items():
        if v is not None:
            project[k] = v
    project["updatedAt"] = _now()
    _save_registry(reg)
    return project


# ─────────────────────────────────────────────
# REQUIREMENTS
# ─────────────────────────────────────────────

def upsert_requirements(
    project_id: str,
    requirements: List[Dict[str, Any]],
) -> Dict[str, Any]:
    reg = _load_registry()
    for req in requirements:
        if not req.get("id"):
            req["id"] = f"req_{uuid.uuid4().hex[:6]}"
    prev = reg["requirements"].get(project_id)
    version_num = (int(prev["version"].lstrip("v")) + 1) if prev else 1
    entry = {
        "projectId": project_id,
        "version": f"v{version_num}",
        "requirements": requirements,
        "storedAt": _now(),
        "diff": {"previousVersion": prev["version"]} if prev else None,
    }
    reg["requirements"][project_id] = entry
    _save_registry(reg)
    return entry


def get_requirements(project_id: str) -> Dict[str, Any]:
    reg = _load_registry()
    if project_id not in reg.get("requirements", {}):
        raise KeyError(f"No requirements found for project {project_id}")
    return reg["requirements"][project_id]


# ─────────────────────────────────────────────
# ML PROBLEM DEFINITION
# ─────────────────────────────────────────────

def upsert_ml_problem(
    project_id: str,
    task_type: str,
    target: str,
    metrics: List[str],
    fairness_constraints: Optional[List[str]] = None,
    assumptions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    reg = _load_registry()
    gate_ids = [
        {"gateId": f"gate_{m.replace(' ', '_')}", "gateType": "metric_threshold", "threshold": None}
        for m in metrics
    ]
    entry = {
        "projectId": project_id,
        "taskType": task_type,
        "target": target,
        "metrics": metrics,
        "fairnessOrSafetyConstraints": fairness_constraints or [],
        "assumptions": assumptions or [],
        "gateIdentifiers": gate_ids,
        "createdAt": _now(),
    }
    reg["ml_problems"][project_id] = entry
    _save_registry(reg)
    return entry


def get_ml_problem(project_id: str) -> Dict[str, Any]:
    reg = _load_registry()
    if project_id not in reg.get("ml_problems", {}):
        raise KeyError(f"No ML problem definition found for project {project_id}")
    return reg["ml_problems"][project_id]


# ─────────────────────────────────────────────
# PROJECT DATA SOURCES (planning-level)
# ─────────────────────────────────────────────

def register_project_data_sources(
    project_id: str,
    sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    reg = _load_registry()
    refs = [f"plan://projects/{project_id}/data-sources/{s['name']}" for s in sources]
    entry = {
        "projectId": project_id,
        "sources": sources,
        "refs": refs,
        "registeredAt": _now(),
    }
    reg["data_sources"][project_id] = entry
    _save_registry(reg)
    return entry


def get_project_data_sources(project_id: str) -> Dict[str, Any]:
    reg = _load_registry()
    if project_id not in reg.get("data_sources", {}):
        raise KeyError(f"No data sources found for project {project_id}")
    return reg["data_sources"][project_id]


# ─────────────────────────────────────────────
# PROJECT PLAN (Roadmap / Milestones)
# ─────────────────────────────────────────────

def upsert_project_plan(
    project_id: str,
    milestones: List[Dict[str, Any]],
    deliverables: List[Dict[str, Any]],
    timeline: Optional[str],
    roles: Optional[List[str]],
) -> Dict[str, Any]:
    reg = _load_registry()
    milestone_ids = []
    for m in milestones:
        if not m.get("id"):
            m["id"] = f"ms_{uuid.uuid4().hex[:6]}"
        milestone_ids.append(m["id"])
    for d in deliverables:
        if not d.get("id"):
            d["id"] = f"del_{uuid.uuid4().hex[:6]}"
    entry = {
        "projectId": project_id,
        "milestoneIds": milestone_ids,
        "milestones": milestones,
        "deliverables": deliverables,
        "timeline": timeline,
        "roles": roles or [],
        "snapshot": {
            "milestoneCount": len(milestones),
            "deliverableCount": len(deliverables),
        },
        "updatedAt": _now(),
    }
    reg["plans"][project_id] = entry
    _save_registry(reg)
    return entry


def get_project_plan(project_id: str) -> Dict[str, Any]:
    reg = _load_registry()
    if project_id not in reg.get("plans", {}):
        raise KeyError(f"No plan found for project {project_id}")
    return reg["plans"][project_id]


# ─────────────────────────────────────────────
# REPOSITORY BOOTSTRAP
# ─────────────────────────────────────────────

def bootstrap_repositories(
    project_id: str,
    repo_types: List[str],
    provider: str,
    naming: Optional[Dict[str, str]],
    access_policy: Optional[str],
) -> Dict[str, Any]:
    reg = _load_registry()
    repos = []
    for rtype in repo_types:
        repo_name = (naming or {}).get(rtype, f"{project_id}-{rtype}")
        if provider == "github":
            uri = f"https://github.com/mlops/{repo_name}"
        elif provider == "gitlab":
            uri = f"https://gitlab.com/mlops/{repo_name}"
        elif provider == "s3":
            uri = f"s3://mlops-storage/{project_id}/{rtype}"
        elif provider == "mlflow":
            uri = f"mlflow://models/{repo_name}"
        else:
            uri = f"{provider}://{repo_name}"
        repos.append({
            "repoType": rtype,
            "uri": uri,
            "id": f"repo_{uuid.uuid4().hex[:8]}",
        })
    entry = {
        "projectId": project_id,
        "repos": repos,
        "accessMetadata": {
            "provider": provider,
            "accessPolicy": access_policy or "private",
        },
        "bootstrappedAt": _now(),
    }
    reg["repos"][f"{project_id}_bootstrap"] = entry
    _save_registry(reg)
    return entry


# ─────────────────────────────────────────────
# TICKET GENERATION
# ─────────────────────────────────────────────

_ROUTING_KEYWORDS: Dict[str, List[str]] = {
    "data": ["dataset", "data", "feature", "ingestion", "source"],
    "model": ["train", "model", "tune", "evaluate", "validate", "experiment"],
    "ops": ["deploy", "monitor", "ops", "serving", "inference"],
    "soft": ["test", "build", "release", "code", "package", "api"],
}


def _route_hint(description: str) -> str:
    desc = description.lower()
    for dimension, keywords in _ROUTING_KEYWORDS.items():
        if any(kw in desc for kw in keywords):
            return dimension
    return "orchestrator"


def generate_tickets(
    project_id: str,
    work_items: List[Dict[str, Any]],
    target_contract: Optional[str],
) -> Dict[str, Any]:
    reg = _load_registry()
    tickets = []
    ticket_ids = []
    routing_hints: Dict[str, str] = {}

    for item in work_items:
        ticket_id = f"tkt_{uuid.uuid4().hex[:8]}"
        ticket_ids.append(ticket_id)
        hint = _route_hint(item["description"])
        routing_hints[ticket_id] = hint
        tickets.append({
            "ticketId": ticket_id,
            "workItemType": item["type"],
            "description": item["description"],
            "routingHint": hint,
        })

    entry = {
        "projectId": project_id,
        "ticketIds": ticket_ids,
        "tickets": tickets,
        "routingHints": routing_hints,
        "targetContract": target_contract,
        "generatedAt": _now(),
    }
    reg["tickets"][f"{project_id}_{uuid.uuid4().hex[:6]}"] = entry
    _save_registry(reg)
    return entry
