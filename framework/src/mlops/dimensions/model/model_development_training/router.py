from fastapi import APIRouter
from ....core.logs import make_log

router = APIRouter()

@router.get("/model-development-training")
def endpoint():
    return make_log(
        area="Model",
        component="Model Development & Training",
        endpoint="/model/model-development-training",
        meta={'consumes': ['selected_features'], 'produces': ['generated_models']},
    )
