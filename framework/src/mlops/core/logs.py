from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

@dataclass(frozen=True)
class ComponentLog:
    trace_id: str
    timestamp: str
    area: str
    component: str
    endpoint: str
    message: str
    meta: Dict[str, Any]

def make_log(*, area: str, component: str, endpoint: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    log = ComponentLog(
        trace_id=str(uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        area=area,
        component=component,
        endpoint=endpoint,
        message=f"Entered component '{component}' in area '{area}'.",
        meta=meta or {},
    )
    return asdict(log)
