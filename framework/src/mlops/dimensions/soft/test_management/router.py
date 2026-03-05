"""
Software – Test Management endpoints.

  POST  /tests               – Trigger a test execution (unit/integration/e2e)
  GET   /tests               – List test executions
  GET   /tests/{testRunId}   – Test results summary and evidence pointers
  PATCH /tests/{testRunId}   – CI callback: update real test result from GitHub Actions
"""
from __future__ import annotations

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


# ─────────────────────────────────────────────
# testSuiteRef → workflow filename map
# Add new suites here as you create them
# ─────────────────────────────────────────────
SUITE_TO_WORKFLOW = {
    "tests/model-suite.yaml": "238673616",
}


class CICallbackRequest(BaseModel):
    status: str
    passed: Optional[bool] = None
    reportRef: Optional[str] = None
    coverageRef: Optional[str] = None


def _dispatch_github_actions(
    test_run_id: str,
    test_suite_ref: str,
    image_ref: Optional[str] = None,
) -> None:
    """
    Auto-trigger GitHub Actions via workflow_dispatch on ks-software branch.
    Runs as a background task — does not block the API response.
    Passes imageRef so the workflow can start the API as a Docker container.
    Marks the test run as FAILED if GitHub is not configured or dispatch fails.
    """
    import logging

    workflow_file = SUITE_TO_WORKFLOW.get(test_suite_ref)
    token         = settings.github_token
    owner         = settings.github_repo_owner
    repo          = settings.github_repo_name
    public_url    = settings.public_api_url

    if not all([workflow_file, token, owner, repo, public_url]):
        svc.update_test_run(test_run_id, status="FAILED", passed=False)
        return

    callback_url = f"{public_url}/soft/tests/{test_run_id}"

    try:
        resp = httpx.post(
            f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_file}/dispatches",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept":        "application/vnd.github.v3+json",
            },
            json={
                "ref": "ks-software",
                "inputs": {
                    "testRunId":   test_run_id,
                    "callbackUrl": callback_url,
                    "apiBaseUrl":  public_url,
                    "imageRef":    image_ref or "",
                },
            },
            timeout=10.0,
        )
        if resp.status_code != 204:
            logging.error(
                "[CI Dispatch] GitHub returned %s: %s",
                resp.status_code, resp.text,
            )
            svc.update_test_run(test_run_id, status="FAILED", passed=False)
    except Exception as exc:
        logging.error("[CI Dispatch] Exception: %s", exc)
        svc.update_test_run(test_run_id, status="FAILED", passed=False)


router = APIRouter()


# ─────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────
@router.post("/tests", response_model=TestRunResponse,
             summary="Trigger a test execution (unit/integration/e2e)")
def trigger_test(req: TestTriggerRequest, background_tasks: BackgroundTasks):
    """
    Queue a test run against a package or build.

    **API spec inputs:** packageId|buildId, testSuiteRef, envRef?, ticketId
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
    try:
        result = svc.trigger_test(
            test_suite_ref=req.testSuiteRef,
            ticket_id=req.ticketId,
            package_id=req.packageId,
            build_id=req.buildId,
            env_ref=req.envRef,
        )
        # Resolve the Docker image ref from the linked package's storageRef
        image_ref: Optional[str] = None
        pkg_id = result.get("packageId") or req.packageId
        if pkg_id:
            pkg = svc.get_package(pkg_id)
            if pkg:
                image_ref = pkg.get("storageRef")
        background_tasks.add_task(
            _dispatch_github_actions,
            result["testRunId"],
            req.testSuiteRef,
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
    """
    List all test run records, optionally filtered by packageId or buildId.
    """
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
    Retrieve pass/fail result, coverage reference, quality metric refs, and report ref.

    **API spec inputs:** none
    **Returns:** pass/fail, coverage/quality metric refs, report ref
    """
    make_log(
        area="Soft",
        component="Test Management",
        endpoint=f"/soft/tests/{testRunId}",
        meta={"testRunId": testRunId},
    )
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
    Called by GitHub Actions at the end of a real CI run.

    GitHub Actions sends:
      status    : PASSED | FAILED
      passed    : true | false
      reportRef : URL to test report artifact
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
