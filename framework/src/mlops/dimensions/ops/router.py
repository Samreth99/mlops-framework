from fastapi import APIRouter

from .deployment_monitoring.router import router as deployment_monitoring_router

router = APIRouter(prefix="/ops", tags=["ops"])

router.include_router(deployment_monitoring_router)
