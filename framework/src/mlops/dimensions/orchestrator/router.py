from fastapi import APIRouter

from .tickets.router import router as tickets_router
from .contracts.router import router as contracts_router
from .runs.router import router as runs_router
from .events.router import router as events_router
from .orchestration.router import router as orchestration_router

router = APIRouter()

router.include_router(tickets_router)
router.include_router(contracts_router)
router.include_router(runs_router)
router.include_router(events_router)
router.include_router(orchestration_router)
