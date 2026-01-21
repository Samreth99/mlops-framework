from fastapi import APIRouter
from ....core.logs import make_log

router = APIRouter()

@router.get("/data-storage-versioning")
def endpoint():
    return make_log(
        area="Data",
        component="Data Storage & Versioning",
        endpoint="/data/data-storage-versioning",
        meta={'stores': ['collected_data', 'processed_data']},
    )
