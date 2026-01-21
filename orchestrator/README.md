# mlops-test (refactored)

## What changed
- Added clean layers:
  - `domain/` : `MLOps` enum (one mapping table, explicit switch)
  - `integration/python/` : thin HTTP client (RestClient)
  - `service/` : orchestration service (fetch + JSON parsing)
  - `camunda/` : delegate + JSON variable helper (Spin)
  - `web/` : debug controller for quick manual calls

## Configure
`application.yaml`:
```yaml
mlops.framework:
  base-url: http://127.0.0.1:8000
```

## Run
```bash
mvn spring-boot:run
```

Debug:
- `GET http://localhost:8080/debug/skelty/DATA_ACQUISITION`


### Camunda field naming
Preferred: `skeltyComponent`
Legacy supported: `componentKey`
