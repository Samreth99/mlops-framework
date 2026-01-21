from fastapi import APIRouter

from .project_management.router import router as project_management_router

router = APIRouter(prefix="/plan", tags=["plan"])

router.include_router(project_management_router)
