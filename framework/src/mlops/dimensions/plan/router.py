from fastapi import APIRouter

from .project_management.router import router as project_management_router
from .repository_management.router import router as repository_management_router
from .ticket_generation.router import router as ticket_generation_router

# No prefix here — main.py already mounts this router at /plan
router = APIRouter()

router.include_router(project_management_router)
router.include_router(repository_management_router)
router.include_router(ticket_generation_router)
