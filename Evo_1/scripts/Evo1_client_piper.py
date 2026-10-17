"""Piper EEF inference client. Dry run is the default; --execute sends commands."""

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import itertools
import json
import math
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import websockets
from scipy.spatial.transform import Rotation, Slerp
try:
    from .inference_latency import CLIENT_FIELDS, LatencyStats
    from .inference_transport import encode_request
except ImportError:
    from inference_latency import CLIENT_FIELDS, LatencyStats
    from inference_transport import encode_request


ARM_KEY = "piper_eef_absolute"
DATASET_KEY = "piper_sdk_raw_eef_next_pose"
LOG_TIMEZONE = timezone(timedelta(hours=8))


def write_inference_log(log_file, record):
    """Flush each JSONL event; represent invalid model numbers as JSON null."""
    def json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    event = {"timestamp": datetime.now(LOG_TIMEZONE).isoformat(), **record}
    log_file.write(json.dumps(json_safe(event), ensure_ascii=False, allow_nan=False) + "\n")
    log_file.flush()


def camera_source(value):
    return int(value) if value.isdecimal() else value


class OpenCVCamera:
    def __init__(self, source, width, height, fps):
        self.source = source
        self.capture = cv2.VideoCapture(camera_source(source))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open OpenCV camera {source}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Frame unavailable from OpenCV camera {self.source}")
        return frame

    def close(self):
        self.capture.release()


class RealSenseCamera:
    def __init__(self, serial, width, height, fps):
        import pyrealsense2 as rs

        self.serial = serial
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        try:
            self.pipeline.start(config)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not start D435i {serial} at {width}x{height}@{fps} BGR8; "
                "check its supported color profile and USB connection"
            ) from exc
        self.closed = False

    def read(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError(f"Color frame unavailable from D435i {self.serial}")
        return np.asanyarray(color.get_data()).copy()

    def close(self):
        if not self.closed:
            self.pipeline.stop()
            self.closed = True


def list_realsense_cameras():
    import pyrealsense2 as rs

    devices = list(rs.context().query_devices())
    if not devices:
        print("No RealSense devices found")
    for device in devices:
        print(f"{device.get_info(rs.camera_info.name)}: {device.get_info(rs.camera_info.serial_number)}")


def make_payload(base_frame, wrist_frame, state, prompt, *, image_transport="json"):
    """The server expects BGR arrays and converts them to RGB before inference."""
    images = [frame if frame.shape == (448, 448, 3)
              else cv2.resize(frame, (448, 448), interpolation=cv2.INTER_CUBIC)
              for frame in (base_frame, wrist_frame)]
    if image_transport == "json":
        images.append(np.zeros((448, 448, 3), dtype=np.uint8))
    return {
        "image": [image.tolist() for image in images] if image_transport == "json" else images,
        "image_mask": [1, 1, 0],
        "state": np.asarray(state, dtype=float).tolist(),
        "action_mask": [1] * 7 + [0] * 17,
        "prompt": prompt,
        "arm_key": ARM_KEY,
        "dataset_key": DATASET_KEY,
    }


def interpolate_target(start, end, fraction):
    """Interpolate absolute TCP position, orientation and gripper width."""
    if not 0 <= fraction <= 1:
        raise ValueError("Interpolation fraction must be in [0, 1]")
    result = np.empty(7, dtype=np.float64)
    result[:3] = start[:3] + fraction * (end[:3] - start[:3])
    result[3:6] = Slerp([0, 1], Rotation.from_rotvec([start[3:6], end[3:6]]))([fraction]).as_rotvec()[0]
    result[6] = start[6] + fraction * (end[6] - start[6])
    return result


def read_state(piper, offset_m, open_width_mm):
    pose_msg = piper.GetArmEndPoseMsgs()
    grip_msg = piper.GetArmGripperMsgs()
    if pose_msg.time_stamp <= 0 or grip_msg.time_stamp <= 0:
        raise RuntimeError("Piper pose or gripper feedback is unavailable")
    ep = pose_msg.end_pose
    flange = np.array([ep.X_axis, ep.Y_axis, ep.Z_axis], dtype=np.float64) * 1e-6
    euler = np.array([ep.RX_axis, ep.RY_axis, ep.RZ_axis], dtype=np.float64) * (np.pi / 180000)
    rotation = Rotation.from_euler("xyz", euler)
    tcp = flange + rotation.apply(offset_m)
    raw_grip = float(grip_msg.gripper_state.grippers_angle)
    if not 0 <= raw_grip <= 71000:
        raise RuntimeError(f"Invalid Piper gripper feedback: {raw_grip}")
    width = np.clip(raw_grip / 70000, 0, 1) * open_width_mm
    return np.concatenate((tcp, rotation.as_rotvec(), [width]))


def validate_target(current, target, low, high, open_width_mm, max_step_m, max_step_rad):
    if target.shape != (7,) or not np.isfinite(target).all():
        raise RuntimeError("Model returned an invalid 7D target")
    if np.any(target[:3] < low[:3] - 0.02) or np.any(target[:3] > high[:3] + 0.02):
        raise RuntimeError("Target is outside the demonstrated position bounds")
    if not 0 <= target[6] <= open_width_mm:
        raise RuntimeError(
            f"Target gripper width {target[6]:.9g} mm is outside the robot range "
            f"[0, {open_width_mm:g}] mm (current: {current[6]:.9g} mm)"
        )
    pos_step = np.linalg.norm(target[:3] - current[:3])
    rot_step = (Rotation.from_rotvec(current[3:6]).inv() * Rotation.from_rotvec(target[3:6])).magnitude()
    if pos_step > max_step_m or rot_step > max_step_rad:
        raise RuntimeError(f"Target step too large: {pos_step:.3f} m, {rot_step:.3f} rad")


def send_target(piper, target, offset_m, open_width_mm):
    rotation = Rotation.from_rotvec(target[3:6])
    flange = target[:3] - rotation.apply(offset_m)
    euler = rotation.as_euler("xyz")
    raw_pose = [int(round(v * 1e6)) for v in flange]
    raw_pose += [int(round(v * 180000 / np.pi)) for v in euler]
    piper.EndPoseCtrl(*raw_pose)
    raw_grip = int(round(np.clip(target[6] / open_width_mm, 0, 1) * 70000))
    piper.GripperCtrl(raw_grip, 1000, 0x01, 0)


async def run_inference_cycle(args, ws, base, wrist, piper, offset, open_width,
                              prompt, low, high, substeps, command_hz, record):
    cycle = record["cycle"]
    capture_started = time.perf_counter()
    base_frame, wrist_frame = await asyncio.gather(
        asyncio.to_thread(base.read), asyncio.to_thread(wrist.read))
    record["capture_ms"] = (time.perf_counter() - capture_started) * 1000
    feedback_started = time.perf_counter()
    current = read_state(piper, offset, open_width)
    record["feedback_read_ms"] = (time.perf_counter() - feedback_started) * 1000
    record["state"] = current.tolist()
    preparation_started = time.perf_counter()
    transport = getattr(args, "image_transport", "binary")
    payload_data = make_payload(base_frame, wrist_frame, current, prompt, image_transport=transport)
    record["request_id"] = uuid.uuid4().hex
    payload_data.update({"return_timing": True, "request_id": record["request_id"]})
    record["payload_build_ms"] = (time.perf_counter() - preparation_started) * 1000
    encoding_started = time.perf_counter()
    payload = encode_request(payload_data, transport)
    record["image_transport"] = transport
    record["request_bytes"] = len(payload) if isinstance(payload, bytes) else len(payload.encode("utf-8"))
    record["request_encode_ms"] = (time.perf_counter() - encoding_started) * 1000
    record["request_preparation_ms"] = (time.perf_counter() - preparation_started) * 1000
    request_started = time.perf_counter()
    try:
        await ws.send(payload)
        response = await ws.recv()
        record["inference_latency_ms"] = (time.perf_counter() - request_started) * 1000
    finally:
        record["request_elapsed_ms"] = (time.perf_counter() - request_started) * 1000
    decoding_started = time.perf_counter()
    response_data = json.loads(response)
    if isinstance(response_data, dict):
        if response_data.get("request_id") != record["request_id"]:
            raise RuntimeError("Server response request_id does not match the inference request")
        record["server_run_id"] = response_data.get("server_run_id")
        for field, value in response_data.get("timings", {}).items():
            if field not in ("server_json_decode_ms", "server_unpack_ms", "server_preprocess_ms", "model_inference_ms",
                             "server_postprocess_ms", "server_processing_ms"):
                continue
            if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                record["server_model_inference_ms" if field == "model_inference_ms" else field] = value
        response_data = response_data["actions"]
    chunk = np.asarray(response_data, dtype=np.float64)
    record["response_decode_ms"] = (time.perf_counter() - decoding_started) * 1000
    record["client_data_processing_ms"] = record["request_preparation_ms"] + record["response_decode_ms"]
    if "server_processing_ms" in record:
        # This remainder also includes WebSocket compression, response encoding
        # and scheduling; it cannot identify pure network time.
        remainder = record["inference_latency_ms"] - record["server_processing_ms"]
        if remainder >= 0:
            record["communication_and_other_ms"] = remainder
    record["action_shape"] = list(chunk.shape)
    record["chunk_length"] = len(chunk) if chunk.ndim == 2 else None
    if chunk.ndim != 2 or chunk.shape[1] < 7 or len(chunk) < args.steps_per_chunk:
        raise RuntimeError(f"Invalid model action shape: {chunk.shape}")
    record["action_chunk"] = chunk[:, :7].tolist()
    execution_started = time.perf_counter()
    next_tick = time.monotonic()
    try:
        for keyframe_index, keyframe in enumerate(chunk[:args.steps_per_chunk, :7]):
            record["keyframe_index"] = keyframe_index
            print(f"cycle={cycle} keyframe={keyframe_index} "
                  f"target={np.array2string(keyframe, precision=6)} execute={args.execute}")
            if np.isfinite(keyframe[6]):
                clipped_width = float(np.clip(keyframe[6], 0, open_width))
                if clipped_width != keyframe[6]:
                    print(f"cycle={cycle} keyframe={keyframe_index} gripper clipped: "
                          f"{keyframe[6]:.9g} -> {clipped_width:g} mm")
                    record["gripper_clips"].append({
                        "keyframe_index": keyframe_index,
                        "before_mm": float(keyframe[6]), "after_mm": clipped_width,
                    })
                    keyframe = keyframe.copy()
                    keyframe[6] = clipped_width
            # Check the pose before interpolation so a large model jump
            # cannot be hidden by small intermediate steps.
            try:
                validate_target(current, keyframe, low, high, open_width,
                                args.max_step_m, args.max_step_rad)
            except RuntimeError as exc:
                record["status"] = "rejected"
                record["error"] = str(exc)
                if args.execute:
                    raise
                print(f"cycle={cycle} keyframe={keyframe_index} rejected: {exc}; "
                      "skipping the rest of this action chunk (dry run)")
                break
            record["accepted_targets"].append(keyframe.tolist())
            start = current.copy()
            previous = start
            for substep in range(1, substeps + 1):
                target = interpolate_target(start, keyframe, substep / substeps)
                validate_target(previous, target, low, high, open_width,
                                args.max_step_m, args.max_step_rad)
                if args.execute:
                    await asyncio.sleep(max(0, next_tick - time.monotonic()))
                    send_target(piper, target, offset, open_width)
                    record["sent_control_steps"] += 1
                next_tick += 1 / command_hz
                previous = target
            record["processed_steps"] += 1
            if args.execute:
                record["executed_steps"] += 1
                await asyncio.sleep(max(0, next_tick - time.monotonic()))
            current = read_state(piper, offset, open_width) if args.execute else keyframe.copy()
        if record["status"] == "running":
            record["status"] = "executed" if args.execute else "dry_run"
    finally:
        record["execution_latency_ms"] = (time.perf_counter() - execution_started) * 1000


async def run(args):
    from piper_sdk import C_PiperInterface_V2

    info = json.loads((args.dataset / "meta/info.json").read_text())
    if info.get("robot_type") != "piper_eef" or info.get("pose_coordinate_frame") != "robot_base_tcp":
        raise ValueError("Expected the raw Piper EEF dataset")
    offset = np.asarray(info["tcp_offset_m"], dtype=np.float64)
    open_width = float(info["gripper_mapping"]["robot_open_width_mm"])
    fps = float(info["fps"])
    prompt = getattr(args, "prompt", None)
    if prompt is None:
        prompt = json.loads((args.dataset / "meta/tasks.jsonl").read_text().splitlines()[0])["task"]
    print(f"Task prompt: {prompt}")
    stats = json.loads((args.dataset / "meta/stats.json").read_text())["observation.state"]
    low, high = np.asarray(stats["min"]), np.asarray(stats["max"])
    if offset.shape != (3,) or not np.isfinite(offset).all() or open_width <= 0 or fps <= 0:
        raise ValueError("Invalid dataset TCP offset, gripper width or FPS")

    if args.control_hz < fps:
        raise ValueError(f"--control-hz must be at least the dataset rate ({fps:g} Hz)")
    substeps = max(1, math.ceil(args.control_hz / fps))
    command_hz = fps * substeps

    run_id = uuid.uuid4().hex
    log_path = getattr(args, "log_path", None) or (
        Path("logs") / f"piper_inference_{datetime.now(LOG_TIMEZONE):%Y%m%d_%H%M%S_%f}.jsonl"
    )
    log_path = Path(log_path)
    latency_path = getattr(args, "latency_json", None) or log_path.with_suffix(".latency.json")
    latency_stats = LatencyStats(CLIENT_FIELDS)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    print(f"Inference log: {log_path.resolve()}")
    print(f"Latency JSON: {Path(latency_path).resolve()}")
    session_started = time.perf_counter()
    session_status, session_error = "completed", None
    totals = {"cycles": 0, "executed_steps": 0, "sent_control_steps": 0}
    base = wrist = piper = None
    try:
        write_inference_log(log_file, {
            "event": "session_start", "run_id": run_id, "execute": args.execute,
            "dataset": str(args.dataset.resolve()), "server": args.server,
            "requested_cycles": args.cycles, "steps_per_chunk": args.steps_per_chunk,
            "dataset_fps": fps, "command_hz": command_hz, "substeps": substeps,
            "max_step_m": args.max_step_m, "max_step_rad": args.max_step_rad,
            "image_transport": getattr(args, "image_transport", "binary"),
            "websocket_compression": None,
            "prompt": prompt,
        })
        cameras = (("image", args.base_serial if args.camera_backend == "realsense" else args.base_camera),
                   ("wrist_image", args.wrist_serial if args.camera_backend == "realsense" else args.wrist_camera))
        for key, source in cameras:
            height, width = info["features"][key]["shape"][:2]
            camera = (RealSenseCamera(source, width, height, args.camera_fps)
                      if args.camera_backend == "realsense"
                      else OpenCVCamera(source, width, height, args.camera_fps))
            if key == "image":
                base = camera
            else:
                wrist = camera
        for _ in range(args.camera_warmup_frames):
            await asyncio.gather(asyncio.to_thread(base.read), asyncio.to_thread(wrist.read))

        piper = C_PiperInterface_V2(can_name=args.can_name, judge_flag=False, can_auto_init=True, dh_is_offset=1)
        piper.ConnectPort(piper_init=False)
        if args.execute:
            deadline = time.monotonic() + 10
            while not piper.EnablePiper():
                if time.monotonic() > deadline:
                    raise TimeoutError("Piper enable timed out")
                await asyncio.sleep(0.02)
            piper.MotionCtrl_2(0x01, 0x00, args.speed_pct, 0x00)
        async with websockets.connect(args.server, max_size=100_000_000, compression=None) as ws:
            cycles = range(args.cycles) if args.cycles else itertools.count()
            for cycle in cycles:
                cycle_started = time.perf_counter()
                record = {
                    "event": "cycle", "run_id": run_id, "cycle": cycle,
                    "started_at": datetime.now(LOG_TIMEZONE).isoformat(),
                    "execute": args.execute, "status": "running", "error": None,
                    "inference_latency_ms": None, "chunk_length": None,
                    "requested_steps": args.steps_per_chunk, "processed_steps": 0,
                    "executed_steps": 0, "sent_control_steps": 0,
                    "gripper_clips": [], "accepted_targets": [],
                }
                try:
                    await run_inference_cycle(
                        args, ws, base, wrist, piper, offset, open_width, prompt,
                        low, high, substeps, command_hz, record,
                    )
                except BaseException as exc:
                    if record["status"] != "rejected":
                        record["status"] = (
                            "interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "error"
                        )
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    record["cycle_latency_ms"] = (time.perf_counter() - cycle_started) * 1000
                    totals["cycles"] += 1
                    totals["executed_steps"] += record["executed_steps"]
                    totals["sent_control_steps"] += record["sent_control_steps"]
                    write_inference_log(log_file, record)
                    if record["chunk_length"] is not None and record["inference_latency_ms"] is not None:
                        latency_stats.observe(record)
                        latency_stats.save(
                            latency_path, role="client", run_id=run_id,
                            scope="Decoded action responses, including responses rejected for execution",
                            note="communication_and_other_ms includes compression, encoding and scheduling; not pure network time",
                        )
                    latency = record["inference_latency_ms"]
                    latency_text = f"{latency:.1f}" if latency is not None else "unavailable"
                    print(f"cycle={cycle} inference_ms={latency_text} chunk_length={record['chunk_length']} "
                          f"executed_steps={record['executed_steps']} "
                          f"sent_control_steps={record['sent_control_steps']} status={record['status']}")
            print(f"Completed {args.cycles} inference cycles")
    except BaseException as exc:
        session_status = "interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "error"
        session_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if piper is not None:
                # Release the connection without changing the arm's enable state.
                piper.DisconnectPort()
            if base is not None:
                base.close()
            if wrist is not None:
                wrist.close()
        finally:
            try:
                write_inference_log(log_file, {
                    "event": "session_end", "run_id": run_id,
                    "status": session_status, "error": session_error, **totals,
                    "session_latency_ms": (time.perf_counter() - session_started) * 1000,
                })
            finally:
                log_file.close()
                latency_stats.save(
                    latency_path, role="client", run_id=run_id,
                    scope="Decoded action responses, including responses rejected for execution",
                    note="communication_and_other_ms includes compression, encoding and scheduling; not pure network time",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("piper_data"))
    parser.add_argument("--server", default="ws://localhost:9000")
    parser.add_argument("--prompt", help="Override task text; default: first task in dataset meta/tasks.jsonl")
    parser.add_argument("--image-transport", choices=("binary", "json"), default="binary",
                        help="Lossless binary images (default); json supports older servers")
    parser.add_argument("--can-name", default="can0")
    parser.add_argument("--camera-backend", choices=("realsense", "opencv"), default="realsense")
    parser.add_argument("--list-cameras", action="store_true", help="List RealSense names and serial numbers")
    parser.add_argument("--base-serial", help="Global D435i serial number")
    parser.add_argument("--wrist-serial", help="Wrist D435i serial number")
    parser.add_argument("--base-camera", help="OpenCV device number or V4L path")
    parser.add_argument("--wrist-camera", help="OpenCV device number or V4L path")
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-warmup-frames", type=int, default=10)
    parser.add_argument("--cycles", type=int, default=20,
                        help="Number of inference cycles; 0 runs until interrupted")
    parser.add_argument("--steps-per-chunk", type=int, default=3)
    parser.add_argument("--control-hz", type=float, default=100.0,
                        help="Command rate, rounded up to a multiple of dataset FPS")
    parser.add_argument("--max-step-m", type=float, default=0.02)
    parser.add_argument("--max-step-rad", type=float, default=0.15)
    parser.add_argument("--speed-pct", type=int, default=20)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--log-path", type=Path,
                        help="Append JSONL inference events here; default: logs/piper_inference_<timestamp>.jsonl")
    parser.add_argument("--latency-json", type=Path,
                        help="Separate latency summary JSON; default: <log basename>.latency.json")
    args = parser.parse_args()
    if args.list_cameras:
        list_realsense_cameras()
    else:
        if args.camera_backend == "realsense":
            if not args.base_serial or not args.wrist_serial or args.base_serial == args.wrist_serial:
                parser.error("Provide different --base-serial and --wrist-serial (see --list-cameras)")
        elif not args.base_camera or not args.wrist_camera:
            parser.error("OpenCV backend requires --base-camera and --wrist-camera")
        if (args.cycles < 0 or args.steps_per_chunk < 1 or args.camera_fps < 1
                or args.camera_warmup_frames < 0 or args.control_hz <= 0
                or args.max_step_m <= 0 or args.max_step_rad <= 0
                or not 20 <= args.speed_pct <= 100):
            parser.error("Invalid cycle, camera, control, safety or speed setting")
        try:
            asyncio.run(run(args))
        except KeyboardInterrupt:
            print("Stopped by user; no arm disable command sent")
