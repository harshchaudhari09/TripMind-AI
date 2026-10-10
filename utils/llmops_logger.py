"""
utils/llmops_logger.py  – LLMOps observability layer

Stage 1: LangSmith (free, zero-config)
Stage 2: ELK Stack (Elasticsearch + Kibana dashboards)

Usage: wrap any LangChain call with log_run()
"""
from __future__ import annotations

import os
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ── LangSmith (Stage 1, free) ─────────────────────────────────────────────────
LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

# ── ELK (Stage 2) ─────────────────────────────────────────────────────────────
ELK_ENABLED   = bool(os.getenv("ELASTICSEARCH_URL"))
ELK_URL       = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ELK_INDEX     = os.getenv("ELASTICSEARCH_INDEX", "travel_planner_logs")

try:
    from elasticsearch import Elasticsearch
    _es = Elasticsearch(ELK_URL) if ELK_ENABLED else None
except ImportError:
    _es = None
    ELK_ENABLED = False


def log_run(
    run_type:    str,
    inputs:      Dict[str, Any],
    outputs:     Dict[str, Any],
    latency_ms:  float,
    error:       Optional[str] = None,
    metadata:    Optional[Dict] = None,
) -> None:
    """
    Log an LLM/tool run to ELK (if configured) and stdout.
    LangSmith logging happens automatically via env vars — no code needed.
    """
    doc = {
        "run_id":     str(uuid.uuid4()),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "run_type":   run_type,
        "inputs":     inputs,
        "outputs":    outputs,
        "latency_ms": latency_ms,
        "error":      error,
        "metadata":   metadata or {},
    }

    # Stdout (always)
    status = "ERROR" if error else "OK"
    print(f"  [OPS] {run_type} | {latency_ms:.0f}ms | {status}")

    # ELK
    if ELK_ENABLED and _es:
        try:
            _es.index(index=ELK_INDEX, document=doc)
        except Exception as e:
            print(f"  [OPS] ELK logging failed: {e}")


def timed_run(run_type: str, inputs: Dict, fn, **kwargs) -> Any:
    """
    Convenience wrapper: times `fn(**kwargs)`, logs to OPS, returns result.
    """
    t0 = time.perf_counter()
    error = None
    result = None
    try:
        result = fn(**kwargs)
    except Exception as e:
        error = str(e)
        raise
    finally:
        latency = (time.perf_counter() - t0) * 1000
        log_run(
            run_type=run_type,
            inputs=inputs,
            outputs={"result": str(result)[:500] if result else ""},
            latency_ms=latency,
            error=error,
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ELK Setup helper — run once to create index + Kibana dashboard
# ─────────────────────────────────────────────────────────────────────────────
def setup_elk_index():
    """Creates the Elasticsearch index with proper mappings."""
    if not ELK_ENABLED or not _es:
        print("ELK not configured — skipping index setup.")
        return

    mapping = {
        "mappings": {
            "properties": {
                "run_id":     {"type": "keyword"},
                "timestamp":  {"type": "date"},
                "run_type":   {"type": "keyword"},
                "latency_ms": {"type": "float"},
                "error":      {"type": "text"},
                "inputs":     {"type": "object", "enabled": False},
                "outputs":    {"type": "object", "enabled": False},
                "metadata":   {"type": "object", "enabled": False},
            }
        }
    }

    if not _es.indices.exists(index=ELK_INDEX):
        _es.indices.create(index=ELK_INDEX, body=mapping)
        print(f"[ELK] Created index: {ELK_INDEX}")
    else:
        print(f"[ELK] Index already exists: {ELK_INDEX}")


if __name__ == "__main__":
    setup_elk_index()
