"""
Software – Test Management endpoints.

  POST  /tests               – Trigger a test execution (unit/integration/e2e)
  GET   /tests               – List test executions
  GET   /tests/{testRunId}   – Test results summary and evidence pointers
  PATCH /tests/{testRunId}   – CI callback: update real test result from GitHub Actions
"""
from __future__ import annotations

import logging
from typing import List, Optional

import httpx

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from ....core.logs import make_log
from ....config.settings import settings
from ..schemas import (
    TestTriggerRequest,
    TestRunResponse,
    TestRunDetailResponse,
)
from .. import software_service as svc


TEST_WORKFLOW_FILE = "docker-test-ci.yml"


class CICallbackRequest(BaseModel):
    status: str
    passed: Optional[bool] = None
    reportRef: Optional[str] = None
    coverageRef: Optional[str] = None


def _dispatch_github_test(
    test_run_id: str,
    image_ref: str,
) -> None:
    """
    Dispatch the Docker test workflow on GitHub Actions as a background task.
    After dispatch, polls GitHub API to find the workflow run_id and stores it.
    No callback/ngrok required — status is polled from GitHub API on demand.
    """
    import time

    token = settings.github_token
    owner = settings.github_repo_owner
    repo  = settings.github_repo_name

    if not all([token, owner, repo, image_ref]):
        svc.update_test_run(
            test_run_id,
            status="FAILED",
            passed=False,
            report_ref=None,
        )
        return

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept":        "application/vnd.github.v3+json",
    }

    try:
        resp = httpx.post(
            f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{TEST_WORKFLOW_FILE}/dispatches",
            headers=headers,
            json={
                "ref": "ks-software",
                "inputs": {
                    "testRunId": test_run_id,
                    "imageRef":  image_ref,
                },
            },
            timeout=10.0,
        )
        if resp.status_code != 204:
            logging.error("[Test Dispatch] GitHub returned %s: %s", resp.status_code, resp.text)
            svc.update_test_run(test_run_id, status="FAILED", passed=False)
            return

        # Poll until GitHub creates the run (up to 30s)
        run_id = None
        for _ in range(10):
            time.sleep(3)
            runs_resp = httpx.get(
                f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{TEST_WORKFLOW_FILE}/runs",
                headers=headers,
                params={"event": "workflow_dispatch", "per_page": 5},
                timeout=10.0,
            )
            if runs_resp.status_code == 200:
                runs = runs_resp.json().get("workflow_runs", [])
                for run in runs:
                    if run.get("name") or True:
                        run_id = run["id"]
                        break
            if run_id:
                break

        # Store run_id for polling
        logs_ref = (
            f"https://github.com/{owner}/{repo}/actions/runs/{run_id}"
            if run_id else None
        )
        svc.patch_test_run_github_run(test_run_id, run_id, logs_ref)

    except Exception as exc:
        logging.error("[Test Dispatch] Exception: %s", exc)
        svc.update_test_run(test_run_id, status="FAILED", passed=False)


def _sync_test_from_github(test_run_id: str) -> None:
    """Poll GitHub Actions API and sync test run status if run is complete."""
    token = settings.github_token
    owner = settings.github_repo_owner
    repo  = settings.github_repo_name

    test_run = svc.get_test_run(test_run_id)
    if not test_run or not test_run.get("githubRunId"):
        return

    run_id = test_run["githubRunId"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept":        "application/vnd.github.v3+json",
    }

    try:
        resp = httpx.get(
            f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}",
            headers=headers,
            timeout=10.0,
        )
        if resp.status_code != 200:
            return

        run = resp.json()
        gh_status     = run.get("status")     # queued | in_progress | completed
        gh_conclusion = run.get("conclusion") # success | failure | cancelled | None

        if gh_status != "completed":
            test_run["status"] = "RUNNING"
            return

        logs_ref = f"https://github.com/{owner}/{repo}/actions/runs/{run_id}"
        if gh_conclusion == "success":
            svc.update_test_run(
                test_run_id,
                status="PASSED",
                passed=True,
                report_ref=logs_ref,
            )
        else:
            svc.update_test_run(
                test_run_id,
                status="FAILED",
                passed=False,
                report_ref=logs_ref,
            )
    except Exception as exc:
        logging.error("[Test Sync] Exception: %s", exc)


router = APIRouter()


# ─────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────
@router.post("/tests", response_model=TestRunResponse,
             summary="Trigger a test execution against a Docker image from ECR")
def trigger_test(req: TestTriggerRequest, background_tasks: BackgroundTasks):
    """
    Queue a test run against a build's Docker image pulled from ECR.
    Dispatches `docker-test-ci.yml` which pulls the image, starts the container,
    runs unit tests and smoke tests, then the status is polled from GitHub API.

    **API spec inputs:** buildId, testSuiteRef, ticketId
    **Returns:** testRunId, queued status
    """
    make_log(
        area="Soft",
        component="Test Management",
        endpoint="/soft/tests",
        meta={
            "packageId": req.packageId,
            "buildId": req.buildId,
            "testSuiteRef": req.testSuiteRef,
        },
    )
    if not req.packageId and not req.buildId:
        raise HTTPException(status_code=400, detail={
            "error": {"code": "missing_test_target", "message": "Provide either packageId or buildId"}
        })

    # Resolve image ref from the linked build
    image_ref: Optional[str] = None
    if req.buildId:
        build = svc.get_build(req.buildId)
        if build:
            image_ref = build.get("environment", {}).get("imageRef")
    if not image_ref and req.packageId:
        pkg = svc.get_package(req.packageId)
        if pkg:
            image_ref = pkg.get("storageRef")

    if not image_ref:
        raise HTTPException(status_code=400, detail={
            "error": {"code": "missing_image_ref", "message": "No imageRef found for the given buildId/packageId. Ensure the build is BUILT first."}
        })

    try:
        result = svc.trigger_test(
            test_suite_ref=req.testSuiteRef,
            ticket_id=req.ticketId,
            package_id=req.packageId,
            build_id=req.buildId,
            env_ref=req.envRef,
        )
        background_tasks.add_task(
            _dispatch_github_test,
            result["testRunId"],
            image_ref,
        )
        return TestRunResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "test_trigger_failed", "message": str(e)}
        })


@router.get("/tests", response_model=List[TestRunResponse],
            summary="List test executions, optionally filtered by package or build")
def list_tests(
    packageId: Optional[str] = Query(None, description="Filter by package ID"),
    buildId: Optional[str] = Query(None, description="Filter by build ID"),
):
    make_log(
        area="Soft",
        component="Test Management",
        endpoint="/soft/tests",
        meta={"packageId": packageId, "buildId": buildId},
    )
    try:
        return [
            TestRunResponse(**r)
            for r in svc.list_test_runs(package_id=packageId, build_id=buildId)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": {"code": "list_tests_failed", "message": str(e)}
        })


@router.get("/tests/{testRunId}", response_model=TestRunDetailResponse,
            summary="Test results summary and evidence pointers")
def get_test_run(testRunId: str):
    """
    Poll the current test run state. Syncs from GitHub Actions API if still in progress.

    **API spec inputs:** none
    **Returns:** pass/fail, logs ref, report ref
    """
    make_log(
        area="Soft",
        component="Test Management",
        endpoint=f"/soft/tests/{testRunId}",
        meta={"testRunId": testRunId},
    )
    # Sync from GitHub Actions if still in progress
    run = svc.get_test_run(testRunId)
    if run and run.get("status") in ("QUEUED", "RUNNING"):
        _sync_test_from_github(testRunId)

    run = svc.get_test_run(testRunId)
    if run is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "test_run_not_found", "message": f"Test run '{testRunId}' not found"}
        })
    return TestRunDetailResponse(**run)


@router.patch("/tests/{testRunId}", response_model=TestRunDetailResponse,
              summary="CI callback: update real test result from GitHub Actions")
def ci_callback(testRunId: str, req: CICallbackRequest):
    """
    Called by GitHub Actions at the end of a CI run (optional — polling is primary).
    """
    make_log(
        area="Soft",
        component="Test Management",
        endpoint=f"/soft/tests/{testRunId}",
        meta={"testRunId": testRunId, "status": req.status},
    )
    result = svc.update_test_run(
        test_run_id=testRunId,
        status=req.status,
        passed=req.passed,
        report_ref=req.reportRef,
        coverage_ref=req.coverageRef,
    )
    if result is None:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "test_run_not_found", "message": f"Test run '{testRunId}' not found"}
        })
    return TestRunDetailResponse(**result)
