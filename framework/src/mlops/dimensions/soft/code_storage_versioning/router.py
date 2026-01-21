from fastapi import APIRouter
from ....core.logs import make_log

router = APIRouter()

@router.get("/code-storage-versioning")
def endpoint():
    return make_log(
        area="Soft",
        component="Code Storage & Versioning",
        endpoint="/soft/code-storage-versioning",
        meta={'typical_tools': ['Git', 'GitHub/GitLab']},
    )
