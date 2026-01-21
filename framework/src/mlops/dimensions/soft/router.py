from fastapi import APIRouter

from .api_development.router import router as api_development_router
from .code_storage_versioning.router import router as code_storage_versioning_router

router = APIRouter(prefix="/soft", tags=["soft"])

router.include_router(api_development_router)
router.include_router(code_storage_versioning_router)
