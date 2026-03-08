FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GIT_PYTHON_REFRESH=quiet

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir -e .

RUN dvc init --no-scm

EXPOSE 8000

CMD ["uvicorn", "mlops.main:app", "--host", "0.0.0.0", "--port", "8000", "--use-colors"]
