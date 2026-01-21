from fastapi import APIRouter
from ....core.logs import make_log

router = APIRouter()

@router.get("/model-storage-versioning")
def endpoint():
    return make_log(
        area="Model",
        component="Model Storage & Versioning",
        endpoint="/model/model-storage-versioning",
        meta={'stores': ['generated_models']},
    )
