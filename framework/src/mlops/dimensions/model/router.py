from fastapi import APIRouter

from .model_development_training.router import router as model_development_training_router
from .model_storage_versioning.router import router as model_storage_versioning_router

router = APIRouter()

router.include_router(model_development_training_router)
router.include_router(model_storage_versioning_router)
