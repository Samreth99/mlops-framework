from fastapi import APIRouter
from ....core.logs import make_log

router = APIRouter()

@router.get("/api-development")
def endpoint():
    return make_log(
        area="Soft",
        component="API Development",
        endpoint="/soft/api-development",
        meta={'owns': ['service interfaces', 'contracts']},
    )
