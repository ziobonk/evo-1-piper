"""Bounded-memory latency statistics and atomic JSON reports."""

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path


SERVER_FIELDS = (
    "server_json_decode_ms", "server_unpack_ms", "server_preprocess_ms", "model_inference_ms",
    "server_postprocess_ms", "server_processing_ms", "server_response_encode_ms",
)
CLIENT_FIELDS = (
    "capture_ms", "feedback_read_ms", "payload_build_ms", "request_encode_ms",
    "request_preparation_ms", "inference_latency_ms", "response_decode_ms",
    "client_data_processing_ms", "server_json_decode_ms", "server_unpack_ms", "server_preprocess_ms",
    "server_model_inference_ms", "server_postprocess_ms", "server_processing_ms",
    "communication_and_other_ms", "execution_latency_ms", "cycle_latency_ms",
)


class LatencyStats:
    def __init__(self, fields):
        self.fields = fields
        self.sample_count = 0
        self.metrics = {}

    def observe(self, record):
        self.sample_count += 1
        for field in self.fields:
            value = record.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                continue
            metric = self.metrics.setdefault(field, {"count": 0, "total_ms": 0.0,
                                                     "min_ms": value, "max_ms": value})
            metric["count"] += 1
            metric["total_ms"] += value
            metric["min_ms"] = min(metric["min_ms"], value)
            metric["max_ms"] = max(metric["max_ms"], value)

    def summary(self):
        result = {}
        for field in self.fields:
            metric = self.metrics.get(field)
            result[field] = ({**metric, "mean_ms": metric["total_ms"] / metric["count"]}
                             if metric else {"count": 0, "total_ms": 0.0, "mean_ms": None,
                                             "min_ms": None, "max_ms": None})
        return {"sample_count": self.sample_count, "metrics": result}

    def save(self, path, **metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "updated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            **metadata, **self.summary(),
        }
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
        temporary.replace(path)
