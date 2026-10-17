# evo1_server_json.py

import sys
import argparse
import os
import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
import uuid
import websockets
import numpy as np
import cv2
import json
import torch
from fvcore.nn import FlopCountAnalysis



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.Evo1 import EVO1
from dataset.lerobot_dataset_pretrain_mp import NormalizationType
try:
    from .inference_latency import LatencyStats, SERVER_FIELDS
    from .inference_transport import decode_request
except ImportError:
    from inference_latency import LatencyStats, SERVER_FIELDS
    from inference_transport import decode_request


def write_server_log(log_file, record):
    timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
    log_file.write(json.dumps({"timestamp": timestamp, **record}, ensure_ascii=False, allow_nan=False) + "\n")
    log_file.flush()


class InferenceTimingStats:
    def __init__(self):
        self.request_count = 0
        self.inference_count = 0
        self.failed_requests = 0
        self.interrupted_requests = 0
        self.total_model_inference_ms = 0.0

    def observe(self, record):
        self.request_count += 1
        if record["status"] == "error":
            self.failed_requests += 1
        elif record["status"] == "interrupted":
            self.interrupted_requests += 1
        latency = record.get("model_inference_ms")
        if record["status"] == "ok" and latency is not None and math.isfinite(latency) and latency >= 0:
            self.inference_count += 1
            self.total_model_inference_ms += latency

    def summary(self):
        return {
            "request_count": self.request_count,
            "inference_count": self.inference_count,
            "failed_requests": self.failed_requests,
            "interrupted_requests": self.interrupted_requests,
            "total_model_inference_ms": self.total_model_inference_ms,
            "mean_model_inference_ms": (
                self.total_model_inference_ms / self.inference_count if self.inference_count else None
            ),
        }


def save_server_summary(log_file, run_id, stats):
    summary = stats.summary()
    write_server_log(log_file, {"event": "session_end", "run_id": run_id, **summary})
    mean = summary["mean_model_inference_ms"]
    mean_text = f"{mean:.3f} ms" if mean is not None else "unavailable (no successful inference)"
    print(f"Server inference summary: count={summary['inference_count']} "
          f"mean_model_inference_ms={mean_text} failed={summary['failed_requests']} "
          f"interrupted={summary['interrupted_requests']}")


def timed_model_inference(model, metrics, *, device="cuda", **inputs):
    """Measure the model call with completed CUDA work at both boundaries."""
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        return model.run_inference(**inputs)
    finally:
        torch.cuda.synchronize(device)
        metrics["model_inference_ms"] = (time.perf_counter() - started) * 1000



class Normalizer:
    def __init__(self, stats_or_path, normalization_type: NormalizationType = NormalizationType.BOUNDS):
        if isinstance(stats_or_path, str):
            with open(stats_or_path, "r") as f:
                self.stats_map = json.load(f)
        else:
            self.stats_map = stats_or_path

        if isinstance(normalization_type, str):
            normalization_type = NormalizationType(normalization_type)
        self.normalization_type = normalization_type
        print(f"Using normalization type: {self.normalization_type}")
        self.target_dim = 24
        self._cache_stats = {}

    def _pad_vector(self, values, name):
        tensor = torch.tensor(values, dtype=torch.float32)
        length = tensor.shape[0]
        if length < self.target_dim:
            pad = torch.zeros(self.target_dim - length, dtype=torch.float32)
            tensor = torch.cat([tensor, pad], dim=0)
        elif length > self.target_dim:
            raise ValueError(f"{name} length {length} exceeds expected {self.target_dim}")
        return tensor

    def _prepare_stats(self, stats_dict, stats_name):
        prepared = {}
        for key, values in stats_dict.items():
            prepared[key] = self._pad_vector(values, f"{stats_name}.{key}")
        return prepared

    def _stat_to_device(self, stats_dict, key, device, dtype):
        tensor = stats_dict.get(key)
        if tensor is None:
            return None
        return tensor.to(device=device, dtype=dtype)
    
    def _get_stats_for(self, arm_key, dataset_key, stats_type):
        """get the relevant stats dict for the given arm/dataset and stat type (state/action)"""
        cache_key = (arm_key, dataset_key, stats_type)
        if cache_key in self._cache_stats:
            return self._cache_stats[cache_key]
        
        if arm_key not in self.stats_map:
             raise ValueError(f"Arm key '{arm_key}' not found in normalization stats.")
             
        if "observation.state" in self.stats_map[arm_key] or "action" in self.stats_map[arm_key]:
            raw_stats = self.stats_map[arm_key]
        else:
            if dataset_key not in self.stats_map[arm_key]:
                 raise ValueError(f"Dataset key '{dataset_key}' not found in normalization stats for arm '{arm_key}'.")
            raw_stats = self.stats_map[arm_key][dataset_key]
        
        dict_key = "observation.state" if stats_type == "state" else "action"
        
        if dict_key not in raw_stats:
            raise ValueError(f"Key '{dict_key}' not found in stats for {arm_key}/{dataset_key}")
            
        prepared = self._prepare_stats(raw_stats[dict_key], dict_key)
        self._cache_stats[cache_key] = prepared
        return prepared

    def _normalize_tensor(self, tensor: torch.Tensor, stats_dict, clamp: bool) -> torch.Tensor:
        eps = 1e-8
        device, dtype = tensor.device, tensor.dtype
        norm_type = self.normalization_type
        
        current_dim = tensor.shape[-1]

        if norm_type == NormalizationType.NORMAL:
            mean = self._stat_to_device(stats_dict, "mean", device, dtype)
            std = self._stat_to_device(stats_dict, "std", device, dtype)
            if mean is not None: mean = mean[..., :current_dim]
            if std is not None: std = std[..., :current_dim]
            
            if mean is None or std is None:
                raise ValueError("Normal normalization selected but mean/std are missing in norm_stats.json")
            return (tensor - mean) / (std + eps)

        low_key, high_key = ("min", "max")
        if norm_type == NormalizationType.BOUNDS_Q99:
            low_key, high_key = ("q01", "q99")

        low = self._stat_to_device(stats_dict, low_key, device, dtype)
        high = self._stat_to_device(stats_dict, high_key, device, dtype)

        if (low is None or high is None) and norm_type == NormalizationType.BOUNDS_Q99:
            logging.warning("Missing q01/q99 stats; falling back to min/max bounds normalization.")
            low = self._stat_to_device(stats_dict, "min", device, dtype)
            high = self._stat_to_device(stats_dict, "max", device, dtype)

        if low is None or high is None:
            raise ValueError("Bounds normalization selected but min/max stats are missing in norm_stats.json")

        low = low[..., :current_dim]
        high = high[..., :current_dim]

        normalized = 2 * (tensor - low) / (high - low + eps) - 1
        if clamp:
            normalized = torch.clamp(normalized, -1.0, 1.0)
        return normalized

    def _denormalize_tensor(self, tensor: torch.Tensor, stats_dict) -> torch.Tensor:
        eps = 1e-8
        device, dtype = tensor.device, tensor.dtype
        norm_type = self.normalization_type

        if norm_type == NormalizationType.NORMAL:
            mean = self._stat_to_device(stats_dict, "mean", device, dtype)
            std = self._stat_to_device(stats_dict, "std", device, dtype)
            if mean is None or std is None:
                raise ValueError("Normal denormalization requested but mean/std stats are missing")
            return tensor * (std + eps) + mean

        low_key, high_key = ("min", "max")
        if norm_type == NormalizationType.BOUNDS_Q99:
            low_key, high_key = ("q01", "q99")

        low = self._stat_to_device(stats_dict, low_key, device, dtype)
        high = self._stat_to_device(stats_dict, high_key, device, dtype)

        if (low is None or high is None) and norm_type == NormalizationType.BOUNDS_Q99:
            logging.warning("Missing q01/q99 stats; falling back to min/max bounds denormalization.")
            low = self._stat_to_device(stats_dict, "min", device, dtype)
            high = self._stat_to_device(stats_dict, "max", device, dtype)

        if low is None or high is None:
            raise ValueError("Bounds denormalization requested but min/max stats are missing")
        
        current_dim = tensor.shape[-1]
        if low.shape[-1] > current_dim:
            low = low[..., :current_dim]
            high = high[..., :current_dim]

        return (tensor + 1.0) / 2.0 * (high - low + eps) + low

    def normalize_state(self, state: torch.Tensor, arm_key: str, dataset_key: str) -> torch.Tensor:
        stats = self._get_stats_for(arm_key, dataset_key, "state")
        norm_state = self._normalize_tensor(state, stats, clamp=True)

        if norm_state.shape[-1] < self.target_dim:
            padding_size = self.target_dim - norm_state.shape[-1]
            pad_tensor = torch.zeros(
                (*norm_state.shape[:-1], padding_size), 
                dtype=norm_state.dtype, 
                device=norm_state.device
            )
            norm_state = torch.cat([norm_state, pad_tensor], dim=-1)
            
        return norm_state

    def denormalize_action(self, action: torch.Tensor, arm_key: str, dataset_key: str) -> torch.Tensor:
        if action.ndim == 1:
            action = action.view(1, -1)
        stats = self._get_stats_for(arm_key, dataset_key, "action")
        denorm_action = self._denormalize_tensor(action, stats)

        # Padding if action dim is less than target_dim
        if denorm_action.shape[-1] < self.target_dim:
            padding_size = self.target_dim - denorm_action.shape[-1]
            pad_tensor = torch.zeros(
                (*denorm_action.shape[:-1], padding_size), 
                dtype=denorm_action.dtype, 
                device=denorm_action.device
            )
            denorm_action = torch.cat([denorm_action, pad_tensor], dim=-1)

        return denorm_action


def load_model_and_normalizer(ckpt_dir):
    config_dict = json.load(open(os.path.join(ckpt_dir, "config.json")))
    stats = json.load(open(os.path.join(ckpt_dir, "norm_stats.json")))

    config_dict["finetune_vlm"] = False
    config_dict["finetune_action_head"] = False
    config_dict["num_inference_timesteps"] = 50

    from config import EvoConfig
    config = EvoConfig.from_dict(config_dict)
    model = EVO1(config, load_pretrained=False).eval()
    ds_path = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt")
    torch_path = os.path.join(ckpt_dir, "checkpoint.pt")
    if os.path.isfile(ds_path):
        checkpoint = torch.load(ds_path, map_location="cpu")
        state_dict = checkpoint["module"]
    elif os.path.isfile(torch_path):
        checkpoint = torch.load(torch_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]
    else:
        raise FileNotFoundError(f"No DeepSpeed or standard checkpoint in {ckpt_dir}")
    model.load_state_dict(state_dict, strict=True)
    model = model.to("cuda")

    normalization_type = config_dict.get("normalization_type", NormalizationType.BOUNDS.value)
    normalizer = Normalizer(stats, normalization_type=normalization_type)
    return model, normalizer




def prepare_image_batch(images, device="cuda"):
    """Convert three BGR inputs to RGB float tensors with one uint8 upload."""
    if len(images) != 3:
        raise ValueError("Must provide exactly 3 images")
    rgb_images = []
    for image in images:
        image = np.asarray(image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Images must have shape (height, width, 3)")
        if image.shape != (448, 448, 3):
            image = cv2.resize(image, (448, 448))
        rgb_images.append(image[..., ::-1])
    # Stack creates writable contiguous storage, including for frombuffer views.
    batch = np.stack(rgb_images)
    tensor = torch.from_numpy(batch).to(device=device)
    tensor = tensor.permute(0, 3, 1, 2).to(dtype=torch.float32).div_(255.0)
    return list(tensor.unbind(0))



def infer_from_json_dict(data: dict, model, normalizer, arm_key, dataset_key, *, metrics=None):
    device = "cuda"
    if metrics is None:
        metrics = {}
    preprocessing_started = time.perf_counter()
    # Prefer per-request arm_key/dataset_key from the client payload; fall back to the
    # server startup defaults. Needed for multi-task benchmarks (RoboTwin sends
    # arm_key=aloha_joint and a per-task dataset_key=robotwin_<task>, and the norm
    # stats are keyed per task).
    arm_key = data.get("arm_key", arm_key)
    dataset_key = data.get("dataset_key", dataset_key)

  
    images = prepare_image_batch(data["image"], device=device)

 
    state = torch.tensor(data["state"], dtype=torch.float32, device=device)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    # if state.shape[1] < 24:
        # state = torch.cat([state, torch.zeros((1, 24 - state.shape[1]), device=device)], dim=1)
    norm_state = normalizer.normalize_state(state, arm_key, dataset_key).to(dtype=torch.float32)

    
    prompt = data["prompt"]
    image_mask = torch.tensor(data["image_mask"], dtype=torch.int32, device=device)
    action_mask = torch.as_tensor(data["action_mask"], dtype=torch.int32, device=device)
    if action_mask.ndim == 1:
        action_mask = action_mask.unsqueeze(0)
    elif action_mask.ndim == 3 and action_mask.shape[:2] == (1, 1):
        action_mask = action_mask.squeeze(1)
    elif action_mask.ndim == 2 and action_mask.shape[0] != 1:
        action_mask = action_mask.unsqueeze(0)

    torch.cuda.synchronize(device)
    metrics["server_preprocess_ms"] = (time.perf_counter() - preprocessing_started) * 1000
    
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        action = timed_model_inference(
            model, metrics, device=device,
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            state_input=norm_state,
            action_mask=action_mask
        )
        postprocessing_started = time.perf_counter()
        action = action.reshape(1, -1, 24)
        action = normalizer.denormalize_action(action[0], arm_key, dataset_key)
        result = action.cpu().numpy().tolist()
        metrics["server_postprocess_ms"] = (time.perf_counter() - postprocessing_started) * 1000
        return result


async def handle_request(websocket, model, normalizer, arm_key, dataset_key, *, log_file=None, run_id=None,
                         stats=None, latency_stats=None, latency_path=None):
    print("Client connected")
    try:
        async for message in websocket:
            record = {
                "event": "inference", "run_id": run_id, "request_id": uuid.uuid4().hex,
                "model_inference_ms": None, "chunk_length": None,
                "status": "running", "error": None,
            }
            try:
                processing_started = time.perf_counter()
                json_data = decode_request(message, record)
                record["request_bytes"] = len(message) if isinstance(message, bytes) else len(message.encode("utf-8"))
                record["request_id"] = json_data.get("request_id", record["request_id"])
                record["arm_key"] = json_data.get("arm_key", arm_key)
                record["dataset_key"] = json_data.get("dataset_key", dataset_key)
                actions = infer_from_json_dict(
                    json_data, model, normalizer, arm_key, dataset_key, metrics=record,
                )
                record["chunk_length"] = len(actions)
                record["status"] = "ok"
                record["server_processing_ms"] = (time.perf_counter() - processing_started) * 1000
                if json_data.get("return_timing", False):
                    response = {
                        "actions": actions, "request_id": record["request_id"], "server_run_id": run_id,
                        "timings": {field: record[field] for field in SERVER_FIELDS if field in record},
                    }
                else:
                    response = actions
                encode_started = time.perf_counter()
                encoded_response = json.dumps(response)
                record["server_response_encode_ms"] = (time.perf_counter() - encode_started) * 1000
            except BaseException as exc:
                record["status"] = (
                    "interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "error"
                )
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                if stats is not None:
                    stats.observe(record)
                if latency_stats is not None and record["status"] == "ok":
                    latency_stats.observe(record)
                if log_file is not None:
                    write_server_log(log_file, record)
                if record["model_inference_ms"] is not None:
                    print(f"request={record['request_id']} model_inference_ms={record['model_inference_ms']:.3f} "
                          f"chunk_length={record['chunk_length']} status={record['status']}")
            await websocket.send(encoded_response)
            if latency_stats is not None and latency_path is not None:
                latency_stats.save(latency_path, role="server", run_id=run_id,
                                   scope="Successful inference requests; includes first request")
            print("Sent action chunk")


    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evo-1 inference server")
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--arm_key", default="piper_eef_absolute")
    parser.add_argument("--dataset_key", default="piper_sdk_raw_eef_next_pose")
    parser.add_argument("--log-path", type=Path,
                        help="Server model timing JSONL; default: logs/evo1_server_inference_<timestamp>.jsonl")
    parser.add_argument("--latency-json", type=Path,
                        help="Separate latency summary JSON; default: <log basename>.latency.json")
    args = parser.parse_args()
    ckpt_dir = args.ckpt_dir
    port = args.port
    arm_key = args.arm_key
    dataset_key = args.dataset_key

    print("Loading EVO_1 model...")
    model, normalizer = load_model_and_normalizer(ckpt_dir)
    run_id = uuid.uuid4().hex
    inference_stats = InferenceTimingStats()
    latency_stats = LatencyStats(SERVER_FIELDS)
    log_path = args.log_path or Path("logs") / (
        f"evo1_server_inference_{datetime.now(timezone(timedelta(hours=8))):%Y%m%d_%H%M%S_%f}.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    latency_path = args.latency_json or log_path.with_suffix(".latency.json")

    async def main():
        print(f"EVO_1 server running at ws://0.0.0.0:{port}")
        async with websockets.serve(
            lambda ws: handle_request(ws, model, normalizer, arm_key, dataset_key,
                                      log_file=inference_log, run_id=run_id, stats=inference_stats,
                                      latency_stats=latency_stats, latency_path=latency_path),
            "0.0.0.0", port, max_size=100_000_000, ping_interval=None, ping_timeout=None,
            compression=None,
        ):
            await asyncio.Future()

    with log_path.open("a", encoding="utf-8") as inference_log:
        print(f"Server inference log: {log_path.resolve()}")
        print(f"Server latency JSON: {latency_path.resolve()}")
        write_server_log(inference_log, {
            "event": "session_start", "run_id": run_id,
            "checkpoint": str(Path(ckpt_dir).resolve()), "port": port,
            "timing_scope": "CUDA-synchronized model.run_inference only",
            "websocket_compression": None,
        })
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("Server stopped by user")
        finally:
            save_server_summary(inference_log, run_id, inference_stats)
            latency_stats.save(latency_path, role="server", run_id=run_id,
                               scope="Successful inference requests; includes first request")
