from fastapi import FastAPI

from .core.styles import mount_custom_docs

from .dimensions.plan.router import router as plan_router
from .dimensions.data.router import router as data_router
from .dimensions.ops.router import router as ops_router
from .dimensions.soft.router import router as soft_router
from .dimensions.model.router import router as model_router
from .dimensions.orchestrator.router import router as orchestrator_router

app = FastAPI(
    title="MLOps Framework",
    version="0.2.0",
    docs_url=None,
)

mount_custom_docs(app)

@app.get("/")
def root():
    return {
        "name": "MLOps Framework",
        "hint": "Open /docs or run `python main.py` to call every component endpoint.",
    }

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(plan_router, prefix="/plan", tags=["plan"])
app.include_router(data_router, prefix="/data", tags=["data"])
app.include_router(ops_router, prefix="/ops", tags=["ops"])
app.include_router(soft_router, prefix="/soft", tags=["soft"])
app.include_router(model_router, prefix="/model", tags=["model"])
app.include_router(orchestrator_router, prefix="/orchestrator", tags=["orchestrator"])