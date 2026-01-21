from fastapi import APIRouter
from ....core.logs import make_log

router = APIRouter()

@router.get("/feature-storage-versioning")
def endpoint():
    return make_log(
        area="Data",
        component="Feature Storage & Versioning",
        endpoint="/data/feature-storage-versioning",
        meta={'stores': ['selected_features']},
    )
