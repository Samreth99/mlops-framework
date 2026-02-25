from fastapi import APIRouter

from .code_management.router import router as code_management_router
from .build_management.router import router as build_management_router
from .package_management.router import router as package_management_router
from .test_management.router import router as test_management_router
from .release_management.router import router as release_management_router

router = APIRouter()

router.include_router(code_management_router)
router.include_router(build_management_router)
router.include_router(package_management_router)
router.include_router(test_management_router)
router.include_router(release_management_router)
