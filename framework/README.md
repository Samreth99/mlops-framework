# MLOps Component Framework (Refactored)

Refactor to **package-by-area** (Plan/Data/Ops/Soft/Model) and **package-by-component** inside each area.

Each component exposes **one endpoint** returning a JSON log that tells you which component you are in.

## Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run API
```bash
uvicorn mlops_component_framework.main:app --reload
```
Open: http://127.0.0.1:8000/docs

## Call everything (no server required)
```bash
python demo.py
```
