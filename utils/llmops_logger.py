"""
utils/llmops_logger.py  – LLMOps observability layer

Stage 1: LangSmith — every graph run is traced automatically (each research node,
         the writer, the critic, every LLM call with its tokens and timing).
         Turn it on with LANGSMITH_TRACING=true + LANGSMITH_API_KEY in .env.
Stage 2: ELK Stack (Elasticsearch + Kibana dashboards) — optional, via log_run().

Usage:
    from utils.llmops_logger import trace_config
    final = travel_graph.invoke(state, config=trace_config(state, source="api"))
"""
from __future__ import annotations

import os
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Importing settings loads .env FIRST, so the checks below see its values.
# (Reading os.getenv before .env was loaded made tracing look "off" when it was on.)
import config.settings  # noqa: F401


# ── LangSmith (Stage 1, free tier) ────────────────────────────────────────────
def langsmith_enabled() -> bool:
    """True if tracing is switched on (new LANGSMITH_* names or legacy LANGCHAIN_* names)."""
    flag = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or "false"
    key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    return flag.lower() == "true" and bool(key)


def langsmith_project() -> str:
    return os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "tripmind"


def trace_config(trip: Dict[str, Any], source: str) -> Dict[str, Any]:
    """
    Run config for travel_graph.invoke(). LangGraph sends the run to LangSmith on its
    own when tracing is on; this just gives the trace a readable name, tags you can
    filter by in the LangSmith UI, and the trip details as searchable metadata.
    Harmless when tracing is off.
    """
    return {
        "run_name": f"Plan trip: {trip.get('origin', '?')} → {trip.get('destination', '?')}",
        "tags": ["tripmind", source],
        "metadata": {
            "destination": trip.get("destination"),
            "origin":      trip.get("origin"),
            "start_date":  trip.get("start_date"),
            "end_date":    trip.get("end_date"),
            "budget_inr":  trip.get("budget_inr"),
            "num_people":  trip.get("num_people"),
            "source":      source,          # api / cli / eval
        },
    }


def tracing_status() -> str:
    if langsmith_enabled():
        return f"LangSmith tracing ON → project '{langsmith_project()}' at https://smith.langchain.com"
    return "LangSmith tracing off (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to enable)"


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
    Log a run summary to stdout and ELK (if configured).
    Detailed step-by-step tracing goes to LangSmith automatically — see trace_config().
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
    extra = " | " + ", ".join(f"{k}={v}" for k, v in outputs.items()) if outputs else ""
    print(f"  [OPS] {run_type} | {latency_ms:.0f}ms | {status}{extra}")

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
    print(tracing_status())
    setup_elk_index()
