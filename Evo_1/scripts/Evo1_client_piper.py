"""Piper EEF client for a model trained with piper_eef_next_pose.

Dry run is the default. Pass --execute only after checking the printed targets.
Camera inputs are OpenCV device numbers or V4L paths in the same view order as
the training data (base camera, then wrist camera).
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import cv2
import numpy as np
import websockets
from scipy.spatial.transform import Rotation


def camera_source(value):
    return int(value) if value.isdecimal() else value


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
        raise RuntimeError("Target gripper width is outside the robot range")
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


async def run(args):
    from piper_sdk import C_PiperInterface_V2

    info = json.loads((args.dataset / "meta/info.json").read_text())
    if info.get("robot_type") != "piper_eef" or info.get("pose_coordinate_frame") != "robot_base_tcp":
        raise ValueError("Expected the raw Piper EEF dataset")
    offset = np.asarray(info["tcp_offset_m"], dtype=np.float64)
    open_width = float(info["gripper_mapping"]["robot_open_width_mm"])
    fps = float(info["fps"])
    prompt = json.loads((args.dataset / "meta/tasks.jsonl").read_text().splitlines()[0])["task"]
    stats = json.loads((args.dataset / "meta/stats.json").read_text())["observation.state"]
    low, high = np.asarray(stats["min"]), np.asarray(stats["max"])
    if offset.shape != (3,) or not np.isfinite(offset).all() or open_width <= 0 or fps <= 0:
        raise ValueError("Invalid dataset TCP offset, gripper width or FPS")

    base = cv2.VideoCapture(camera_source(args.base_camera))
    wrist = cv2.VideoCapture(camera_source(args.wrist_camera))
    if not base.isOpened() or not wrist.isOpened():
        raise RuntimeError("Could not open both cameras")
    for capture, key in ((base, "image"), (wrist, "wrist_image")):
        height, width = info["features"][key]["shape"][:2]
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
    piper = C_PiperInterface_V2(can_name=args.can_name, judge_flag=False, can_auto_init=True, dh_is_offset=1)
    piper.ConnectPort(piper_init=False)
    enabled = False
    try:
        if args.execute:
            deadline = time.monotonic() + 10
            while not piper.EnablePiper():
                if time.monotonic() > deadline:
                    raise TimeoutError("Piper enable timed out")
                await asyncio.sleep(0.02)
            enabled = True
            piper.MotionCtrl_2(0x01, 0x00, args.speed_pct, 0x00)
        async with websockets.connect(args.server, max_size=100_000_000) as ws:
            for cycle in range(args.cycles):
                ok_base, base_frame = base.read()
                ok_wrist, wrist_frame = wrist.read()
                if not ok_base or not ok_wrist:
                    raise RuntimeError("Camera frame unavailable")
                current = read_state(piper, offset, open_width)
                images = [cv2.resize(frame, (448, 448), interpolation=cv2.INTER_CUBIC)
                          for frame in (base_frame, wrist_frame)]
                images.append(np.zeros((448, 448, 3), dtype=np.uint8))
                payload = {
                    "image": [image.tolist() for image in images],
                    "image_mask": [1, 1, 0],
                    "state": current.astype(float).tolist(),
                    "action_mask": [1] * 7 + [0] * 17,
                    "prompt": prompt,
                    "arm_key": "piper_eef_absolute",
                    "dataset_key": "piper_sdk_raw_eef_next_pose",
                }
                await ws.send(json.dumps(payload))
                chunk = np.asarray(json.loads(await ws.recv()), dtype=np.float64)
                if chunk.ndim != 2 or chunk.shape[1] < 7:
                    raise RuntimeError(f"Invalid model action shape: {chunk.shape}")
                for target in chunk[:args.steps_per_chunk, :7]:
                    validate_target(current, target, low, high, open_width,
                                    args.max_step_m, args.max_step_rad)
                    print(f"cycle={cycle} target={np.array2string(target, precision=4)} execute={args.execute}")
                    if args.execute:
                        send_target(piper, target, offset, open_width)
                        await asyncio.sleep(1 / fps)
                        current = read_state(piper, offset, open_width)
                    else:
                        current = target
    finally:
        if enabled:
            piper.DisableArm()
        piper.DisconnectPort()
        base.release()
        wrist.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("piper_data"))
    parser.add_argument("--server", default="ws://localhost:9000")
    parser.add_argument("--can-name", default="can0")
    parser.add_argument("--base-camera", required=True)
    parser.add_argument("--wrist-camera", required=True)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--steps-per-chunk", type=int, default=3)
    parser.add_argument("--max-step-m", type=float, default=0.02)
    parser.add_argument("--max-step-rad", type=float, default=0.15)
    parser.add_argument("--speed-pct", type=int, default=20)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.cycles < 1 or args.steps_per_chunk < 1 or not 20 <= args.speed_pct <= 100:
        parser.error("cycles and steps-per-chunk must be positive; speed-pct must be 20..100")
    asyncio.run(run(args))
