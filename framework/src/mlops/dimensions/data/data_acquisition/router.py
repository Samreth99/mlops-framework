from fastapi import APIRouter
from ....core.logs import make_log

router = APIRouter()

@router.get("/data-acquisition")
def endpoint():
    return make_log(
        area="Data",
        component="Data Acquisition",
        endpoint="/data/data-acquisition",
        meta={'consumes': ['new_data'], 'produces': ['collected_data']},
    )
