"""
Unit tests for orchestrator_service — no external dependencies.
"""
import pytest
from mlops.dimensions.orchestrator import orchestrator_service as svc


@pytest.fixture(autouse=True)
def reset_store():
    for key in svc._store:
        if isinstance(svc._store[key], dict):
            svc._store[key].clear()
        else:
            svc._store[key] = []
    yield


# ── Ticket ────────────────────────────────────────────────────────
def test_create_ticket():
    t = svc.create_ticket(
        type="FEATURE_SET", category="data", severity="low",
        priority="normal", title="New features", description="desc",
        requester="alice",
    )
    assert t["ticketId"].startswith("tkt-")
    assert t["type"] == "FEATURE_SET"
    assert t["state"] == "open"


def test_get_ticket_not_found():
    assert svc.get_ticket("nonexistent") is None


def test_list_tickets_empty():
    assert svc.list_tickets() == []


# ── Classify ticket routing ───────────────────────────────────────
def test_classify_feature_set_ticket():
    t = svc.create_ticket(
        type="FEATURE_SET", category="data", severity="low",
        priority="normal", title="t", description="d", requester="alice",
    )
    result = svc.classify_ticket(t["ticketId"], signals={})
    assert result["routingLabel"] == "NEW_DATA"
    assert result["confidence"] >= 0.8


def test_classify_model_candidate_ready():
    t = svc.create_ticket(
        type="MODEL_CANDIDATE_READY", category="model", severity="low",
        priority="normal", title="t", description="d", requester="alice",
    )
    result = svc.classify_ticket(t["ticketId"], signals={})
    assert result["routingLabel"] == "DEPLOY_CANDIDATE"


def test_classify_data_bug_found():
    t = svc.create_ticket(
        type="DATA_BUG_FOUND", category="data", severity="high",
        priority="high", title="t", description="d", requester="alice",
    )
    result = svc.classify_ticket(t["ticketId"], signals={})
    assert result["routingLabel"] == "DATA_QUALITY_ISSUE"


def test_classify_manual_override():
    t = svc.create_ticket(
        type="FEATURE_SET", category="data", severity="low",
        priority="normal", title="t", description="d", requester="alice",
    )
    result = svc.classify_ticket(t["ticketId"], signals={}, manual_override="CUSTOM_LABEL")
    assert result["routingLabel"] == "CUSTOM_LABEL"
    assert result["confidence"] == 1.0


def test_classify_unknown_type_returns_unsupported():
    t = svc.create_ticket(
        type="UNKNOWN_TYPE", category="other", severity="low",
        priority="normal", title="t", description="d", requester="alice",
    )
    result = svc.classify_ticket(t["ticketId"], signals={})
    assert result["routingLabel"] == "UNSUPPORTED_TICKET"
